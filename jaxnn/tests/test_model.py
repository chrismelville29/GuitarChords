from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxnn.nn import model


def test_mnist_cnn_forward_shape_matches_labels():
    rng = jax.random.PRNGKey(0)
    init_fn, apply_fn = model.build_mnist_cnn()
    params = init_fn(rng)
    x = jnp.ones((4, 28, 28, 1), dtype=jnp.float32)
    logits = apply_fn(params, x)
    logits_eval = apply_fn(params, x, is_training=False)
    assert logits.shape == (4, 10)
    assert logits_eval.shape == (4, 10)
