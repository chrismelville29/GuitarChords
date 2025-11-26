"""Stochastic Gradient Descent (SGD) optimizer implementation."""

from __future__ import annotations

from dataclasses import dataclass

from .base import Optimizer, scale_updates
from .. import types

Params = types.Params
OptState = types.OptState


@dataclass(frozen=True)
class SGD(Optimizer):
    lr: float = 1e-2

    def init(self, params: Params) -> OptState:
        # SGD has no internal state
        return None

    def update(
        self,
        grads: Params,
        state: OptState,
        params: Params,   # unused, but required by interface
    ) -> tuple[Params, OptState]:

        updates = scale_updates(grads, self.lr)
        return updates, state