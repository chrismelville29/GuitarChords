"""Adam optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .base import Optimizer, scale_updates
from .. import tree, types

Params = types.Params
OptState = types.OptState


@dataclass(frozen=True)
class AdamState:
    step: int
    m: Params
    v: Params


@dataclass(frozen=True)
class Adam(Optimizer):
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def init(self, params: Params) -> AdamState:
        return AdamState(
            step=0,
            m=tree.zeros_like(params),
            v=tree.zeros_like(params),
        )

    def update(
        self,
        grads: Params,
        state: AdamState,
        params: Params,   # unused, but required by interface
    ) -> tuple[Params, AdamState]:

        step = state.step + 1

        # Update biased first and second moments
        m = jax.tree_util.tree_map(
            lambda m, g: self.beta1 * m + (1.0 - self.beta1) * g,
            state.m,
            grads,
        )

        v = jax.tree_util.tree_map(
            lambda v, g: self.beta2 * v + (1.0 - self.beta2) * (g * g),
            state.v,
            grads,
        )

        # Bias correction
        m_hat = jax.tree_util.tree_map(
            lambda m: m / (1.0 - self.beta1**step),
            m,
        )

        v_hat = jax.tree_util.tree_map(
            lambda v: v / (1.0 - self.beta2**step),
            v,
        )

        # Compute parameter updates
        updates = jax.tree_util.tree_map(
            lambda m, v: -self.lr * m / (jnp.sqrt(v) + self.eps),
            m_hat,
            v_hat,
        )

        new_state = AdamState(
            step=step,
            m=m,
            v=v,
        )

        return updates, new_state