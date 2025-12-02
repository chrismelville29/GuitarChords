"""Dropout layer for regularization."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ... import types
from . import base

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


@dataclass(frozen=True)
class Dropout(base.Layer):
    """Dropout layer that randomly zeros elements during training.

    Args:
        rate: Fraction of inputs to drop (between 0 and 1)
    """
    rate: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.rate < 1:
            raise ValueError(f"Dropout rate must be in [0, 1), got {self.rate}")

    def init(self, rng: PRNGKey) -> Params:
        _ = rng  # unused
        return {}

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = params  # no learnable parameters

        if not is_training or self.rate == 0:
            return inputs

        if rng is None:
            raise ValueError("Dropout requires rng during training")

        keep_prob = 1 - self.rate
        mask = jax.random.bernoulli(rng, keep_prob, shape=inputs.shape)
        return jnp.where(mask, inputs / keep_prob, 0)


__all__ = ["Dropout"]
