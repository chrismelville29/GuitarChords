"""Flatten layer for bridging conv stacks into dense heads."""
from __future__ import annotations

from dataclasses import dataclass

from ... import types
from . import base

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


@dataclass(frozen=True)
class Flatten(base.Layer):
    """Reshapes ``(batch, ...)`` tensors to ``(batch, features)``."""

    def init(self, rng: PRNGKey) -> Params:
        _ = rng
        return {}

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (params, rng, is_training)
        if inputs.ndim < 2:
            raise ValueError("Flatten expects at least a batch dimension and one feature dimension")
        batch_size = inputs.shape[0]
        return inputs.reshape((batch_size, -1))


__all__ = ["Flatten"]
