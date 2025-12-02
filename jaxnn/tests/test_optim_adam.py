from __future__ import annotations

import jax.numpy as jnp

from jaxnn.optim.adam import Adam
from jaxnn.optim.schedule import cosine_decay_schedule


def test_adam_uses_lr_schedule_on_first_step():
    schedule = cosine_decay_schedule(initial_lr=0.2, final_lr=0.0, total_steps=5)
    optimizer = Adam(lr=0.1, lr_schedule=schedule)
    params = jnp.array([1.0])
    grads = jnp.array([1.0])

    updates, state = optimizer.update(grads, optimizer.init(params), params)
    assert jnp.allclose(updates, jnp.array([-0.2]), atol=1e-5)
    assert state.step == 1
