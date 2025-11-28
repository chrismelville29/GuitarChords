"""Activation layer wrapper to slot non-linearities into Sequential."""
from __future__ import annotations

from dataclasses import dataclass

from ... import types
from .. import activations
from . import base

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


@dataclass(frozen=True)
class Activation(base.Layer):
    """Stateless layer that applies a named activation function."""

    name: str

    def init(self, rng: PRNGKey) -> Params:
        _ = rng  # no parameters to initialize
        return {}

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (params, rng, is_training)  # present for API symmetry; unused
        return activations.get_activation(self.name)(inputs)


__all__ = ["Activation"]
