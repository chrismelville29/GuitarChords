from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxnn.models.hand_graph_attention import (
    HAND_GRAPH_FEATURE_DIM,
    HandGraphAttentionNetwork,
    hand_edge_index,
    hand_graph_features,
)


def test_hand_graph_features_shapes():
    coords = np.zeros((21, 3), dtype=np.float32)
    feats = hand_graph_features(coords, include_wrist=True)
    assert feats.shape == (21, HAND_GRAPH_FEATURE_DIM)

    coords_no_wrist = np.zeros((20, 3), dtype=np.float32)
    feats_now = hand_graph_features(coords_no_wrist, include_wrist=False)
    assert feats_now.shape == (20, HAND_GRAPH_FEATURE_DIM)


def test_hand_edge_index_directed():
    edge_index = hand_edge_index(include_wrist=True)
    assert edge_index.shape[0] == 2
    # MediaPipe hand graph has 21 undirected edges -> 42 directed
    assert edge_index.shape[1] == 42


def test_hand_gat_forward_shapes():
    network = HandGraphAttentionNetwork(num_classes=5, hidden_dim=32, num_layers=2, num_heads=4)
    params = network.init(jax.random.PRNGKey(0))

    dummy = jnp.zeros((8, 21, HAND_GRAPH_FEATURE_DIM), dtype=jnp.float32)
    logits = network.apply(params, dummy, rng=None, is_training=False)
    assert logits.shape == (8, 5)

    single = jnp.zeros((21, HAND_GRAPH_FEATURE_DIM), dtype=jnp.float32)
    logits_single = network.apply(params, single, rng=None, is_training=False)
    assert logits_single.shape == (5,)
