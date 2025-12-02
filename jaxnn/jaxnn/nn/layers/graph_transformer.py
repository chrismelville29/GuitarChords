"""Graph Transformer-style attention layers.

This module provides two minimal yet functional building blocks:

* ``GraphTransformerLayer`` — a dense self-attention block that optionally
  accepts an ``(N, N)`` attention mask to restrict attention to graph edges.
* ``GraphAttentionTransformerLayer`` — a GAT-inspired sparse attention block
  operating on ``edge_index`` (COO) inputs. It normalizes coefficients per
  destination node and supports optional edge features.

Both layers keep the library's init/apply split and avoid any external graph
utilities so they can be used in pure-JAX settings or dropped into existing
pipelines that already manage graph batching.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import jax
import jax.numpy as jnp

from ... import types
from .. import activations
from ..init import bias_zeros, glorot_uniform
from . import base
from .layernorm import LayerNorm

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


# ------------------------------- helpers --------------------------------- #


def _parse_dense_inputs(inputs: Array | Mapping[str, Array]) -> tuple[Array, Array | None]:
    """Accept either a raw node feature array or a mapping with ``x``/``mask``.

    Returns (x, mask) where mask may be ``None``.
    """

    if isinstance(inputs, Mapping):
        if "x" not in inputs:
            raise ValueError("Mapping inputs must include key 'x' for node features")
        x = inputs["x"]
        mask = inputs.get("mask")
    else:
        x, mask = inputs, None
    return x, mask


def _parse_sparse_inputs(
    inputs: Mapping[str, Array]
) -> tuple[Array, Array, Array | None]:
    """Extract ``x``, ``edge_index`` (+ optional ``edge_attr``) from a mapping."""

    if not isinstance(inputs, Mapping):
        raise TypeError("GraphAttentionTransformerLayer expects a mapping input")
    try:
        x = inputs["x"]
        edge_index = inputs["edge_index"]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise KeyError("Input mapping must contain 'x' and 'edge_index'") from exc
    edge_attr = inputs.get("edge_attr")
    return x, edge_index, edge_attr


def _segment_softmax(logits: Array, dst: Array, num_nodes: int) -> Array:
    """Compute softmax over incoming edges per destination node.

    Args:
        logits: Attention scores per edge per head with shape (E, H).
        dst: Destination node indices for each edge with shape (E,).
        num_nodes: Total number of nodes; determines the scatter shape.
    """

    max_init = jnp.full((num_nodes, logits.shape[1]), -jnp.inf, logits.dtype)
    max_per_dst = max_init.at[dst].max(logits)
    stabilized = logits - max_per_dst[dst]
    exp = jnp.exp(stabilized)
    denom = jnp.zeros_like(max_per_dst).at[dst].add(exp)
    return exp / (denom[dst] + 1e-9)


# -------------------------- Dense graph attention ------------------------- #


@dataclass(frozen=True)
class GraphMultiHeadAttention(base.Layer):
    """Standard multi-head self-attention operating on dense graphs.

    Input formats
    ------------
    * ``x: Array`` of shape ``(N, d_model)``
    * or a mapping ``{"x": x, "mask": mask}`` where ``mask`` is a boolean or
      float array of shape ``(N, N)``. False/0 entries are treated as
      disallowed attention positions and receive a large negative bias.
    """

    embed_dim: int
    num_heads: int
    out_dim: int | None = None
    use_bias: bool = True

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

    # ------------------ parameter init & projection helpers ------------------ #

    def _init_proj(self, rng: PRNGKey, head_dim: int) -> Array:
        return glorot_uniform(rng, (self.embed_dim, self.num_heads, head_dim))

    def init(self, rng: PRNGKey) -> Params:
        head_dim = self.embed_dim // self.num_heads
        wq_key, wk_key, wv_key, wo_key, bq_key, bk_key, bv_key, bo_key = jax.random.split(rng, 8)

        w_q = self._init_proj(wq_key, head_dim)
        w_k = self._init_proj(wk_key, head_dim)
        w_v = self._init_proj(wv_key, head_dim)
        w_o = glorot_uniform(wo_key, (self.num_heads * head_dim, self.out_dim or self.embed_dim))

        params: Params = {"w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o}
        if self.use_bias:
            params.update(
                {
                    "b_q": bias_zeros(bq_key, (self.num_heads, head_dim)),
                    "b_k": bias_zeros(bk_key, (self.num_heads, head_dim)),
                    "b_v": bias_zeros(bv_key, (self.num_heads, head_dim)),
                    "b_o": bias_zeros(bo_key, (self.out_dim or self.embed_dim,)),
                }
            )
        return params

    # ------------------------------- forward -------------------------------- #

    def apply(
        self,
        params: Params,
        inputs: Array | Mapping[str, Array],
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng, is_training)  # unused
        x, mask = _parse_dense_inputs(inputs)
        if x.ndim != 2:
            raise ValueError("GraphMultiHeadAttention expects (N, d_model) inputs")

        head_dim = self.embed_dim // self.num_heads
        scale = 1.0 / jnp.sqrt(head_dim)

        # Linear projections
        q = jnp.einsum("nd,dhk->nhk", x, params["w_q"])
        k = jnp.einsum("nd,dhk->nhk", x, params["w_k"])
        v = jnp.einsum("nd,dhk->nhk", x, params["w_v"])
        if self.use_bias:
            q = q + params["b_q"]
            k = k + params["b_k"]
            v = v + params["b_v"]

        logits = jnp.einsum("nhk,mhk->hnm", q, k) * scale  # (H, N, N)
        if mask is not None:
            if mask.shape != (x.shape[0], x.shape[0]):
                raise ValueError("mask must have shape (N, N) matching node count")
            zero = jnp.array(0.0, dtype=logits.dtype)
            neg_inf = jnp.array(-1e9, dtype=logits.dtype)
            mask_bias = jnp.where(mask, zero, neg_inf)
            logits = logits + mask_bias[None, :, :]

        attn = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("hnm,mhk->nhk", attn, v)  # (N, H, head_dim)
        concat = attended.reshape(x.shape[0], -1)
        out = concat @ params["w_o"]
        if self.use_bias:
            out = out + params["b_o"]
        return out


@dataclass(frozen=True)
class GraphTransformerLayer(base.Layer):
    """Pre-LN transformer block specialized for node features.

    Inputs: same formats as :class:`GraphMultiHeadAttention`.
    Residual connections assume ``embed_dim`` on both input and output.
    """

    embed_dim: int
    num_heads: int
    ff_hidden_dim: int | None = None
    activation: str = "gelu"
    use_layer_norm: bool = True
    use_bias: bool = True

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.ff_hidden_dim is not None and self.ff_hidden_dim <= 0:
            raise ValueError("ff_hidden_dim must be positive when provided")

    # ------------------------------- init ----------------------------------- #

    def init(self, rng: PRNGKey) -> Params:
        attn_key, ln1_key, ln2_key, ff1_key, ff2_key, b1_key, b2_key = jax.random.split(rng, 7)
        attn_params = GraphMultiHeadAttention(
            self.embed_dim, self.num_heads, out_dim=self.embed_dim, use_bias=self.use_bias
        ).init(attn_key)

        ff_dim = self.ff_hidden_dim or 2 * self.embed_dim
        w1 = glorot_uniform(ff1_key, (self.embed_dim, ff_dim))
        b1 = bias_zeros(b1_key, (ff_dim,)) if self.use_bias else None
        w2 = glorot_uniform(ff2_key, (ff_dim, self.embed_dim))
        b2 = bias_zeros(b2_key, (self.embed_dim,)) if self.use_bias else None

        params: Params = {
            "attn": attn_params,
            "ffn": {"w1": w1, "b1": b1, "w2": w2, "b2": b2},
        }

        if self.use_layer_norm:
            params["ln1"] = LayerNorm(self.embed_dim).init(ln1_key)
            params["ln2"] = LayerNorm(self.embed_dim).init(ln2_key)
        return params

    # ------------------------------- forward -------------------------------- #

    def apply(
        self,
        params: Params,
        inputs: Array | Mapping[str, Array],
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = is_training  # no training-specific branches yet
        x, mask = _parse_dense_inputs(inputs)
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"GraphTransformerLayer expected features {self.embed_dim}, got {x.shape[-1]}"
            )

        # Attention sub-layer (pre-LN)
        h = LayerNorm(self.embed_dim).apply(params["ln1"], x) if self.use_layer_norm else x
        attn_out = GraphMultiHeadAttention(
            self.embed_dim, self.num_heads, out_dim=self.embed_dim, use_bias=self.use_bias
        ).apply(params["attn"], {"x": h, "mask": mask}, rng=rng)
        x = x + attn_out

        # Feed-forward sub-layer (pre-LN)
        h = LayerNorm(self.embed_dim).apply(params["ln2"], x) if self.use_layer_norm else x
        ff = h @ params["ffn"]["w1"]
        if self.use_bias and params["ffn"]["b1"] is not None:
            ff = ff + params["ffn"]["b1"]
        ff = activations.get_activation(self.activation)(ff)
        ff = ff @ params["ffn"]["w2"]
        if self.use_bias and params["ffn"]["b2"] is not None:
            ff = ff + params["ffn"]["b2"]
        return x + ff


# --------------------------- Sparse graph attention ------------------------ #


@dataclass(frozen=True)
class GraphAttentionTransformerLayer(base.Layer):
    """GAT-style sparse attention block with transformer feed-forward.

    Inputs must be a mapping with keys:
    * ``x``: node features shaped ``(N, in_features)``.
    * ``edge_index``: integer array shaped ``(2, E)`` in COO format.
    * optional ``edge_attr``: edge features shaped ``(E, edge_features)``.

    Residual connections assume ``in_features == out_features`` for simplicity.
    """

    in_features: int
    out_features: int
    num_heads: int
    concat_heads: bool = True
    add_self_loops: bool = True
    edge_features: int | None = None
    ff_hidden_dim: int | None = None
    activation: str = "gelu"
    use_layer_norm: bool = True
    negative_slope: float = 0.2
    use_bias: bool = True

    def __post_init__(self) -> None:
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.concat_heads and self.out_features % self.num_heads != 0:
            raise ValueError("out_features must be divisible by num_heads when concatenating heads")
        if not self.concat_heads and self.out_features <= 0:
            raise ValueError("out_features must be positive when averaging heads")
        if self.in_features != self.out_features:
            raise ValueError("Residual path assumes in_features == out_features for this layer")
        if self.ff_hidden_dim is not None and self.ff_hidden_dim <= 0:
            raise ValueError("ff_hidden_dim must be positive when provided")

    # ------------------------------- init ----------------------------------- #

    def init(self, rng: PRNGKey) -> Params:
        (
            attn_key,
            a_src_key,
            a_dst_key,
            a_edge_key,
            ln1_key,
            ln2_key,
            ff1_key,
            ff2_key,
            b1_key,
            b2_key,
            bv_key,
        ) = jax.random.split(rng, 11)
        head_dim = self.out_features // self.num_heads if self.concat_heads else self.out_features

        w = glorot_uniform(attn_key, (self.in_features, self.num_heads, head_dim))
        a_src = glorot_uniform(a_src_key, (self.num_heads, head_dim))
        a_dst = glorot_uniform(a_dst_key, (self.num_heads, head_dim))

        attn_params: Params = {"w": w, "a_src": a_src, "a_dst": a_dst}
        if self.edge_features is not None:
            a_edge = glorot_uniform(a_edge_key, (self.edge_features, self.num_heads))
            attn_params["a_edge"] = a_edge
        if self.use_bias:
            attn_params["b_v"] = bias_zeros(bv_key, (self.num_heads, head_dim))

        ff_dim = self.ff_hidden_dim or 2 * self.out_features
        w1 = glorot_uniform(ff1_key, (self.out_features, ff_dim))
        b1 = bias_zeros(b1_key, (ff_dim,)) if self.use_bias else None
        w2 = glorot_uniform(ff2_key, (ff_dim, self.out_features))
        b2 = bias_zeros(b2_key, (self.out_features,)) if self.use_bias else None

        params: Params = {
            "attn": attn_params,
            "ffn": {"w1": w1, "b1": b1, "w2": w2, "b2": b2},
        }

        if self.use_layer_norm:
            params["ln1"] = LayerNorm(self.in_features).init(ln1_key)
            params["ln2"] = LayerNorm(self.out_features).init(ln2_key)
        return params

    # ------------------------------- forward -------------------------------- #

    def _apply_attention(
        self,
        params: Params,
        x: Array,
        edge_index: Array,
        edge_attr: Array | None,
    ) -> Array:
        head_dim = self.out_features // self.num_heads if self.concat_heads else self.out_features
        w, a_src, a_dst = params["w"], params["a_src"], params["a_dst"]
        b_v = params.get("b_v")

        # Linear projection per head
        projected = jnp.einsum("nd,dhc->nhc", x, w)
        if self.use_bias and b_v is not None:
            projected = projected + b_v

        src, dst = edge_index
        Wh_src = projected[src]  # (E, H, C)
        Wh_dst = projected[dst]

        logits = (Wh_src * a_src).sum(-1) + (Wh_dst * a_dst).sum(-1)  # (E, H)
        if edge_attr is not None and "a_edge" in params:
            logits = logits + jnp.einsum("ef,fh->eh", edge_attr, params["a_edge"])

        logits = jax.nn.leaky_relu(logits, negative_slope=self.negative_slope)
        alpha = _segment_softmax(logits, dst, x.shape[0])  # normalize per destination

        messages = alpha[..., None] * Wh_src
        agg = jnp.zeros((x.shape[0], self.num_heads, head_dim), dtype=messages.dtype)
        agg = agg.at[dst].add(messages)

        if self.concat_heads:
            return agg.reshape(x.shape[0], self.num_heads * head_dim)
        return jnp.mean(agg, axis=1)

    def apply(
        self,
        params: Params,
        inputs: Mapping[str, Array],
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng, is_training)
        x, edge_index, edge_attr = _parse_sparse_inputs(inputs)

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        num_nodes = x.shape[0]

        # Optionally add self-loops to stabilize training
        if self.add_self_loops:
            loops = jnp.arange(num_nodes, dtype=edge_index.dtype)
            self_edges = jnp.stack([loops, loops], axis=0)
            edge_index = jnp.concatenate([edge_index, self_edges], axis=1)
            if edge_attr is not None:
                loop_attr = jnp.zeros((num_nodes, edge_attr.shape[1]), dtype=edge_attr.dtype)
                edge_attr = jnp.concatenate([edge_attr, loop_attr], axis=0)

        if edge_attr is not None and self.edge_features is not None:
            if edge_attr.shape[1] != self.edge_features:
                raise ValueError(
                    f"edge_attr second dim {edge_attr.shape[1]} must match edge_features {self.edge_features}"
                )

        # Attention + residual
        h = LayerNorm(self.in_features).apply(params["ln1"], x) if self.use_layer_norm else x
        attn_out = self._apply_attention(params["attn"], h, edge_index, edge_attr)
        x = x + attn_out

        # Feed-forward + residual
        h = LayerNorm(self.out_features).apply(params["ln2"], x) if self.use_layer_norm else x
        ff = h @ params["ffn"]["w1"]
        if self.use_bias and params["ffn"]["b1"] is not None:
            ff = ff + params["ffn"]["b1"]
        ff = activations.get_activation(self.activation)(ff)
        ff = ff @ params["ffn"]["w2"]
        if self.use_bias and params["ffn"]["b2"] is not None:
            ff = ff + params["ffn"]["b2"]
        return x + ff


__all__ = [
    "GraphMultiHeadAttention",
    "GraphTransformerLayer",
    "GraphAttentionTransformerLayer",
]
