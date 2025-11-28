from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxnn.nn import layers


def test_mlp_forward_shape():
    rng = jax.random.PRNGKey(0)
    network = layers.make_mlp((4, 8, 2), activation="relu")
    params = network.init(rng)
    x = jnp.ones((3, 4))
    y = network.apply(params, x)
    assert y.shape == (3, 2)


def test_activation_layer_runs_registered_function():
    rng = jax.random.PRNGKey(0)
    layer = layers.Activation("relu")
    params = layer.init(rng)
    x = jnp.array([[-1.0, 2.5]], dtype=jnp.float32)
    y = layer.apply(params, x)
    assert jnp.allclose(y, jnp.array([[0.0, 2.5]], dtype=jnp.float32))


def test_flatten_layer_collapses_spatial_dims():
    rng = jax.random.PRNGKey(0)
    layer = layers.Flatten()
    params = layer.init(rng)
    x = jnp.ones((2, 3, 4), dtype=jnp.float32)
    y = layer.apply(params, x)
    assert y.shape == (2, 12)
