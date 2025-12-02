"""Stochastic Gradient Descent (SGD) optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .base import Optimizer, scale_updates
from . import base
from .schedule import LRSchedule
from .. import types

Params = types.Params
OptState = types.OptState


@dataclass(frozen=True)
class SGD(Optimizer):
    learning_rate: float = 1e-2
    momentum: float = 0.0
    lr_schedule: LRSchedule | None = None

    def init(self, params: Params) -> OptState:
        needs_state = self.momentum != 0.0 or self.lr_schedule is not None
        if not needs_state:
            return None

        state: dict[str, object] = {"step": 0}
        if self.momentum != 0.0:
            state["momentum"] = jax.tree_util.tree_map(jnp.zeros_like, params)
        return state

    def update(
        self,
        grads: Params,
        state: OptState,
        params: Params,   # unused, but required by interface
    ) -> tuple[Params, OptState]:

        step = 0 if state is None else state.get("step", 0)
        lr = self.learning_rate if self.lr_schedule is None else self.lr_schedule(step)
        lr = jnp.asarray(lr, dtype=jnp.result_type(lr, 0.0))

        momentum_buf = None if state is None else state.get("momentum")
        if self.momentum == 0.0:
            velocity = grads
        else:
            if momentum_buf is None:
                momentum_buf = jax.tree_util.tree_map(jnp.zeros_like, grads)
            velocity = jax.tree_util.tree_map(
                lambda m, g: self.momentum * m + g,
                momentum_buf,
                grads,
            )

        updates = scale_updates(velocity, lr)
        new_params = base.apply_updates(params, updates)
        if self.lr_schedule is None and self.momentum == 0.0:
            return new_params, None

        new_state = {"step": step + 1}
        if self.momentum != 0.0:
            new_state["momentum"] = velocity
        return new_params, new_state
