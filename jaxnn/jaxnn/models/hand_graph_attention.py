"""Graph Attention Network tailored for MediaPipe hand landmarks.

This module keeps the MediaPipe hand topology (21 landmarks + connections)
explicit so we can build GAT-style models without depending on the runtime
MediaPipe graph definitions. The graph is based on the connections documented
in MediaPipe Hands and exposed via ``mp.solutions.hands.HAND_CONNECTIONS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from jaxnn import types
from jaxnn.nn.layers import Dense, LayerNorm
from jaxnn.nn.layers import base as layer_base
from jaxnn.nn.layers.graph_transformer import GraphAttentionTransformerLayer

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey

# MediaPipe Hands landmark ordering.
HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

# Undirected MediaPipe hand graph edges.
_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 5),
    (0, 17),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (5, 9),
    (6, 7),
    (7, 8),
    (9, 10),
    (9, 13),
    (10, 11),
    (11, 12),
    (13, 14),
    (13, 17),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
)

NUM_LANDMARKS = len(HAND_LANDMARK_NAMES)
NUM_FINGERS = 5


def _build_edge_index(include_wrist: bool) -> jnp.ndarray:
    edges: list[tuple[int, int]] = []
    for src, dst in _HAND_CONNECTIONS:
        if not include_wrist:
            if src == 0 or dst == 0:
                continue
            src -= 1
            dst -= 1
        edges.append((src, dst))
        edges.append((dst, src))
    edge_arr = np.array(edges, dtype=np.int32)
    return jnp.asarray(edge_arr.T)


_EDGE_INDEX_WITH_WRIST = _build_edge_index(include_wrist=True)
_EDGE_INDEX_WITHOUT_WRIST = _build_edge_index(include_wrist=False)


def hand_edge_index(include_wrist: bool = True) -> jnp.ndarray:
    """Return a COO edge_index for the MediaPipe hand graph."""

    return _EDGE_INDEX_WITH_WRIST if include_wrist else _EDGE_INDEX_WITHOUT_WRIST


def _finger_ids() -> np.ndarray:
    ids = np.full(NUM_LANDMARKS, -1, dtype=np.int32)
    ids[1:5] = 0
    ids[5:9] = 1
    ids[9:13] = 2
    ids[13:17] = 3
    ids[17:21] = 4
    return ids


FINGER_IDS = _finger_ids()
FINGER_ONE_HOT = np.eye(NUM_FINGERS, dtype=np.float32)[np.clip(FINGER_IDS, 0, None)]
FINGER_ONE_HOT[FINGER_IDS < 0] = 0.0

_joint_depth = np.zeros(NUM_LANDMARKS, dtype=np.float32)
for start in (1, 5, 9, 13, 17):
    _joint_depth[start : start + 4] = np.linspace(0.0, 1.0, 4, dtype=np.float32)
JOINT_DEPTH = _joint_depth[:, None]

IS_WRIST = (FINGER_IDS < 0).astype(np.float32)[:, None]

HAND_GRAPH_FEATURE_DIM = 3 + 1 + NUM_FINGERS + 1 + 1  # xyz + radius + finger one-hot + depth + wrist flag


def _metadata(include_wrist: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if include_wrist:
        return FINGER_ONE_HOT, JOINT_DEPTH, IS_WRIST
    return FINGER_ONE_HOT[1:], JOINT_DEPTH[1:], IS_WRIST[1:]


def hand_graph_features(
    landmarks: np.ndarray,
    *,
    include_wrist: bool = True,
    norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Convert landmark coordinates into per-node features for the GAT.

    Args:
        landmarks: Array of shape ``(N, 3)`` where ``N`` is 20 (wrist removed)
            or 21 (full hand). When ``include_wrist`` is ``True`` and the input
            contains only 20 points we automatically prepend the wrist at the
            origin so feature order always matches ``HAND_LANDMARK_NAMES``.
        include_wrist: Whether the returned features should keep the wrist.
        norm_stats: Optional ``(mean, std)`` tuple for xyz standardization.
    """

    coords = np.asarray(landmarks, dtype=np.float32)
    if include_wrist:
        if coords.shape == (NUM_LANDMARKS - 1, 3):
            coords = np.vstack([np.zeros((1, 3), dtype=coords.dtype), coords])
        elif coords.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"Expected 20 or 21 points, got shape {coords.shape}")
    else:
        if coords.shape == (NUM_LANDMARKS, 3):
            coords = coords[1:]
        elif coords.shape != (NUM_LANDMARKS - 1, 3):
            raise ValueError(f"Expected 20 or 21 points, got shape {coords.shape}")

    coords_norm = coords
    if norm_stats is not None:
        mean, std = norm_stats
        coords_norm = (coords_norm - mean[None, :]) / (std[None, :] + 1e-6)

    radius = np.linalg.norm(coords_norm, axis=-1, keepdims=True)
    finger_one_hot, joint_depth, is_wrist = _metadata(include_wrist)
    features = np.concatenate(
        [coords_norm, radius, finger_one_hot.astype(np.float32), joint_depth, is_wrist],
        axis=-1,
    )
    return features.astype(np.float32)


@dataclass(frozen=True)
class HandGraphAttentionNetwork(layer_base.Layer):
    """Small GAT stack specialized for 21-node hand graphs."""

    num_classes: int
    hidden_dim: int = 96
    num_layers: int = 3
    num_heads: int = 4
    readout: Literal["mean", "max"] = "mean"
    include_wrist: bool = True
    activation: str = "gelu"
    node_feature_dim: int = HAND_GRAPH_FEATURE_DIM

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.readout not in {"mean", "max"}:
            raise ValueError("readout must be 'mean' or 'max'")

    def _input_layer(self) -> Dense:
        return Dense(self.node_feature_dim, self.hidden_dim, activation=self.activation)

    def _gat_layer(self) -> GraphAttentionTransformerLayer:
        return GraphAttentionTransformerLayer(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim,
            num_heads=self.num_heads,
            concat_heads=True,
            add_self_loops=True,
            ff_hidden_dim=2 * self.hidden_dim,
            activation=self.activation,
        )

    def _classifier_layer(self) -> Dense:
        return Dense(self.hidden_dim, self.num_classes, activation=None)

    def init(self, rng: PRNGKey) -> Params:
        key_count = 2 + self.num_layers
        keys = jax.random.split(rng, key_count)
        params: Params = {
            "input_proj": self._input_layer().init(keys[0]),
            "gat_layers": tuple(self._gat_layer().init(keys[idx + 1]) for idx in range(self.num_layers)),
            "norm": LayerNorm(self.hidden_dim).init(keys[-2]),
            "classifier": self._classifier_layer().init(keys[-1]),
        }
        return params

    def _forward_single(self, params: Params, sample: Array) -> Array:
        if sample.shape[-1] != self.node_feature_dim:
            raise ValueError(
                f"Expected node feature dim {self.node_feature_dim}, got {sample.shape[-1]}"
            )

        x = self._input_layer().apply(params["input_proj"], sample)
        edge_index = hand_edge_index(include_wrist=self.include_wrist)
        for layer_params in params["gat_layers"]:
            x = self._gat_layer().apply(layer_params, {"x": x, "edge_index": edge_index})

        x = LayerNorm(self.hidden_dim).apply(params["norm"], x)
        if self.readout == "mean":
            graph_embed = jnp.mean(x, axis=0)
        else:
            graph_embed = jnp.max(x, axis=0)
        logits = self._classifier_layer().apply(params["classifier"], graph_embed)
        return logits

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng, is_training)
        x = jnp.asarray(inputs)
        squeeze = False
        if x.ndim == 2:
            x = x[None, ...]
            squeeze = True
        elif x.ndim != 3:
            raise ValueError("Inputs must have shape (batch, nodes, feat) or (nodes, feat)")

        forward_fn = lambda sample: self._forward_single(params, sample)
        logits = jax.vmap(forward_fn)(x)
        return logits[0] if squeeze else logits


__all__ = [
    "HAND_LANDMARK_NAMES",
    "HAND_GRAPH_FEATURE_DIM",
    "hand_edge_index",
    "hand_graph_features",
    "HandGraphAttentionNetwork",
]
