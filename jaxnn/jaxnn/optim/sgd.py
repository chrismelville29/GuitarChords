"""Stochastic Gradient Descent (SGD) optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .base import Optimizer, scale_updates
from . import base
from .. import types

Params = types.Params
OptState = types.OptState


@dataclass(frozen=True)
class SGD(Optimizer):
    learning_rate: float = 1e-2
    momentum: float = 0.0

    def init(self, params: Params) -> OptState:
        if self.momentum == 0.0:
            return None
        momentum_buf = jax.tree_util.tree_map(jnp.zeros_like, params)
        return {"momentum": momentum_buf}

    def update(
        self,
        grads: Params,
        state: OptState,
        params: Params,   # unused, but required by interface
    ) -> tuple[Params, OptState]:

        if self.momentum == 0.0 or state is None:
            velocity = grads
            new_state = state
        else:
            momentum_buf = state["momentum"]
            velocity = jax.tree_util.tree_map(
                lambda m, g: self.momentum * m + g,
                momentum_buf,
                grads,
            )
            new_state = {"momentum": velocity}

        updates = scale_updates(velocity, self.learning_rate)
        new_params = base.apply_updates(params, updates)
        return new_params, new_state
