from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxnn.optim.sgd import SGD
from jaxnn.optim.schedule import cosine_decay_schedule


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


def test_sgd_respects_cosine_lr_schedule():
    schedule = cosine_decay_schedule(initial_lr=0.2, final_lr=0.0, total_steps=2)
    optimizer = SGD(learning_rate=0.1, momentum=0.0, lr_schedule=schedule)
    params = jnp.array([1.0])
    grads = jnp.array([1.0])
    state = optimizer.init(params)

    params1, state1 = optimizer.update(grads, state, params)
    assert jnp.allclose(params1, jnp.array([0.8]))
    params2, state2 = optimizer.update(grads, state1, params1)
    assert jnp.allclose(params2, jnp.array([0.7]))
    assert state2 is not None and state2["step"] == 2


def test_cosine_schedule_clamps_to_final_lr():
    schedule = cosine_decay_schedule(initial_lr=0.1, final_lr=0.01, total_steps=1)
    start_lr = schedule(0)
    end_lr = schedule(10)
    assert jnp.allclose(start_lr, 0.1)
    assert jnp.allclose(end_lr, 0.01)
