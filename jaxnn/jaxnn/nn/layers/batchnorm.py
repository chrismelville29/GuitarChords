"""Batch normalization layer for NHWC inputs."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ... import types
from . import base

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


@dataclass(frozen=True)
class BatchNorm(base.Layer):
    num_features: int
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.num_features <= 0:
            raise ValueError("num_features must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    def init(self, rng: PRNGKey) -> Params:
        _ = rng  # unused
        gamma = jnp.ones((self.num_features,), dtype=jnp.float32)
        beta = jnp.zeros((self.num_features,), dtype=jnp.float32)
        return {"gamma": gamma, "beta": beta}

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng, is_training)  # batch stats only; no running averages yet
        if inputs.ndim != 4:
            raise ValueError("BatchNorm expects NHWC inputs with rank 4")
        if inputs.shape[-1] != self.num_features:
            raise ValueError(
                f"Input channels ({inputs.shape[-1]}) must match num_features ({self.num_features})"
            )

        mean = jnp.mean(inputs, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(inputs, axis=(0, 1, 2), keepdims=True)
        normalized = (inputs - mean) / jnp.sqrt(var + self.epsilon)
        return normalized * params["gamma"] + params["beta"]


__all__ = ["BatchNorm"]
