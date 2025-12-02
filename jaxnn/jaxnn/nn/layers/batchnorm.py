"""Batch normalization layer for NHWC inputs with running statistics.

This implementation keeps running mean/variance inside the layer parameters and
updates them during training. The running stats are *not* intended to be
trained via gradients; callers should exclude them from weight decay /
optimizer updates (see train_hand_pose.py for an example mask-based skip).
"""
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
    momentum: float = 0.1

    def __post_init__(self) -> None:
        if self.num_features <= 0:
            raise ValueError("num_features must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0.0 < self.momentum < 1.0):
            raise ValueError("momentum must be in (0, 1)")

    def init(self, rng: PRNGKey) -> Params:
        _ = rng  # unused
        gamma = jnp.ones((self.num_features,), dtype=jnp.float32)
        beta = jnp.zeros((self.num_features,), dtype=jnp.float32)
        running_mean = jnp.zeros((self.num_features,), dtype=jnp.float32)
        running_var = jnp.ones((self.num_features,), dtype=jnp.float32)
        return {"gamma": gamma, "beta": beta, "running_mean": running_mean, "running_var": running_var}

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array | tuple[Array, Params]:
        _ = rng
        if inputs.ndim != 4:
            raise ValueError("BatchNorm expects NHWC inputs with rank 4")
        if inputs.shape[-1] != self.num_features:
            raise ValueError(
                f"Input channels ({inputs.shape[-1]}) must match num_features ({self.num_features})"
            )

        if is_training:
            mean = jnp.mean(inputs, axis=(0, 1, 2))  # shape (C,)
            var = jnp.var(inputs, axis=(0, 1, 2))

            running_mean = self.momentum * params["running_mean"] + (1.0 - self.momentum) * mean
            running_var = self.momentum * params["running_var"] + (1.0 - self.momentum) * var

            normalized = (inputs - mean) / jnp.sqrt(var + self.epsilon)
            out = normalized * params["gamma"] + params["beta"]

            updated_params = {
                **params,
                "running_mean": running_mean,
                "running_var": running_var,
            }
            return out, updated_params

        # Evaluation: use running statistics
        normalized = (inputs - params["running_mean"]) / jnp.sqrt(params["running_var"] + self.epsilon)
        return normalized * params["gamma"] + params["beta"]


__all__ = ["BatchNorm"]
