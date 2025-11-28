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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TrainState:
    params: Params
    opt_state: OptState

    def tree_flatten(self):
        return (self.params, self.opt_state), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, opt_state = children
        return cls(params=params, opt_state=opt_state)

    def replace(self, *, params: Params | None = None, opt_state: OptState | None = None) -> "TrainState":
        return TrainState(
            params=self.params if params is None else params,
            opt_state=self.opt_state if opt_state is None else opt_state,
        )
