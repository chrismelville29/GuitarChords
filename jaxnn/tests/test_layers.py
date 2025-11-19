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
