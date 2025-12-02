from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxnn.nn.layers import (
    GraphAttentionTransformerLayer,
    GraphMultiHeadAttention,
    GraphTransformerLayer,
)


def test_graph_multi_head_attention_accepts_mask_and_shapes():
    rng = jax.random.PRNGKey(0)
    layer = GraphMultiHeadAttention(embed_dim=4, num_heads=2)
    params = layer.init(rng)

    x = jnp.ones((3, 4), dtype=jnp.float32)
    mask = jnp.eye(3, dtype=bool)  # only self-attend

    out = layer.apply(params, {"x": x, "mask": mask})
    assert out.shape == (3, 4)
    assert jnp.all(jnp.isfinite(out))


def test_graph_transformer_layer_round_trip_shape():
    rng = jax.random.PRNGKey(1)
    layer = GraphTransformerLayer(embed_dim=6, num_heads=2, ff_hidden_dim=8)
    params = layer.init(rng)

    x = jnp.arange(30, dtype=jnp.float32).reshape(5, 6) * 0.1
    full_mask = jnp.ones((5, 5), dtype=bool)

    out = layer.apply(params, {"x": x, "mask": full_mask})
    assert out.shape == (5, 6)
    assert jnp.all(jnp.isfinite(out))


def test_graph_attention_transformer_layer_runs_sparse_edges():
    rng = jax.random.PRNGKey(2)
    layer = GraphAttentionTransformerLayer(
        in_features=4,
        out_features=4,
        num_heads=2,
        concat_heads=True,
        add_self_loops=True,
    )
    params = layer.init(rng)

    x = jnp.ones((3, 4), dtype=jnp.float32)
    edge_index = jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32)

    out = layer.apply(params, {"x": x, "edge_index": edge_index})
    assert out.shape == (3, 4)
    assert jnp.all(jnp.isfinite(out))

