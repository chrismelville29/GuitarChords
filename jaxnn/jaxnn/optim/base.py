"""Shared optimizer plumbing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import jax

from .. import tree, types

Params = types.Params
OptState = types.OptState


def apply_updates(params: Params, updates: Params) -> Params:
    return jax.tree_util.tree_map(lambda p, u: p + u, params, updates)


def scale_updates(updates: Params, lr: float) -> Params:
    return jax.tree_util.tree_map(lambda g: -lr * g, updates)


class Optimizer(Protocol):
    def init(self, params: Params) -> OptState: ...

    def update(self, grads: Params, state: OptState, params: Params) -> Tuple[Params, OptState]: ...


@dataclass(frozen=True)
class TrainState:
    params: Params
    opt_state: OptState

    def replace(self, *, params: Params | None = None, opt_state: OptState | None = None) -> "TrainState":
        return TrainState(params=params or self.params, opt_state=opt_state or self.opt_state)
