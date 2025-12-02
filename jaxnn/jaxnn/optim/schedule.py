"""Learning rate schedules."""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

LRSchedule = Callable[[int], jnp.ndarray]


def cosine_decay_schedule(initial_lr: float, final_lr: float, total_steps: int) -> LRSchedule:
    """
    Cosine decay from ``initial_lr`` to ``final_lr`` over ``total_steps``.

    Args:
        initial_lr: Starting learning rate at step 0.
        final_lr: Ending learning rate when ``step >= total_steps``.
        total_steps: Number of steps over which to anneal. Must be positive.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive for cosine_decay_schedule")

    def schedule(step: int) -> float:
        step_clipped = jnp.minimum(jnp.asarray(step), total_steps)
        cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * step_clipped / total_steps))
        return final_lr + (initial_lr - final_lr) * cosine_decay

    return schedule


__all__ = ["cosine_decay_schedule"]
