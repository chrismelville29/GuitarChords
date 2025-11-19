"""Convenience helpers for composing layers into init/apply pairs."""
from __future__ import annotations

from typing import Callable, Tuple

from .. import types
from . import layers

InitFn = Callable[[types.PRNGKey], types.Params]
ApplyFn = Callable[[types.Params, types.Array], types.Array]


def build_mlp(layer_sizes: Tuple[int, ...], activation: str | None = "relu") -> Tuple[InitFn, ApplyFn]:
    network = layers.make_mlp(layer_sizes, activation=activation)

    def init_fn(rng: types.PRNGKey) -> types.Params:
        return network.init(rng)

    def apply_fn(params: types.Params, x: types.Array) -> types.Array:
        return network.apply(params, x)

    return init_fn, apply_fn
