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


def test_maxpool2d_reduces_spatial_extent():
    layer = layers.MaxPool2D(kernel_size=(2, 2), strides=(2, 2), padding="VALID")
    params = layer.init(jax.random.PRNGKey(0))
    x = jnp.arange(1, 17, dtype=jnp.float32).reshape(1, 4, 4, 1)
    y = layer.apply(params, x)
    expected = jnp.array([[[[6.0], [8.0]], [[14.0], [16.0]]]])
    assert y.shape == (1, 2, 2, 1)
    assert jnp.allclose(y, expected)


def test_maxpool2d_allows_same_padding_and_default_stride():
    layer = layers.MaxPool2D(kernel_size=(2, 2), padding="SAME")
    params = layer.init(jax.random.PRNGKey(0))
    x = jnp.array(
        [
            [
                [[1.0], [2.0], [3.0]],
                [[4.0], [5.0], [6.0]],
                [[7.0], [8.0], [9.0]],
            ]
        ]
    )
    y = layer.apply(params, x)
    expected = jnp.array([[[[5.0], [6.0]], [[8.0], [9.0]]]])
    assert y.shape == (1, 2, 2, 1)
    assert jnp.allclose(y, expected)
