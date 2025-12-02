"""Lightweight Layer Normalization for transformer-style blocks."""
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
class LayerNorm(base.Layer):
    """Applies LayerNorm over the last dimension of the input.

    This keeps the library's pure init/apply API and matches the signature used
    by other layers. Gamma/beta parameters are one-dimensional and broadcast
    over any leading dimensions in the input.
    """

    features: int
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.features <= 0:
            raise ValueError("features must be positive for LayerNorm")
        if self.eps <= 0:
            raise ValueError("eps must be positive for numerical stability")

    def init(self, rng: PRNGKey) -> Params:
        _ = rng  # stateless initialization; keep signature uniform
        gamma = jnp.ones((self.features,), dtype=jnp.float32)
        beta = jnp.zeros((self.features,), dtype=jnp.float32)
        return {"gamma": gamma, "beta": beta}

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng, is_training)  # API symmetry; unused
        if inputs.shape[-1] != self.features:
            raise ValueError(
                f"LayerNorm expected last dimension {self.features}, got {inputs.shape[-1]}"
            )

        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)
        inv_std = jax.lax.rsqrt(var + self.eps)
        normalized = (inputs - mean) * inv_std
        return normalized * params["gamma"] + params["beta"]


__all__ = ["LayerNorm"]
