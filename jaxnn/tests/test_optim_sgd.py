from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxnn.optim.sgd import SGD


def test_sgd_basic_update_without_momentum():
    optimizer = SGD(learning_rate=0.1, momentum=0.0)
    params = jnp.array([1.0, 2.0])
    grads = jnp.array([1.0, -1.0])
    state = optimizer.init(params)
    new_params, new_state = optimizer.update(grads, state, params)
    assert new_state is None
    expected = params - 0.1 * grads
    assert jnp.allclose(new_params, expected)


def test_sgd_with_momentum_accumulates_velocity():
    optimizer = SGD(learning_rate=0.1, momentum=0.9)
    params = jnp.array([1.0])
    grads = jnp.array([1.0])
    state = optimizer.init(params)

    # First step: velocity = grad
    params1, state1 = optimizer.update(grads, state, params)
    assert jnp.allclose(params1, jnp.array([0.9]))
    # Second step: velocity = 0.9 * prev_vel + grad = 1.9
    params2, state2 = optimizer.update(grads, state1, params1)
    assert jnp.allclose(params2, jnp.array([0.9 - 0.19]))
    assert "momentum" in state2
