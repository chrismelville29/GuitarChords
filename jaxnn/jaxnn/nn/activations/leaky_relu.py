"""Leaky ReLU activation (Pytorch-style)."""
from __future__ import annotations

import jax.numpy as jnp

from .. import types

Array = types.Array


def leaky_relu(x: Array, negative_slope: float = 0.01) -> Array:
    if negative_slope < 0:
        raise ValueError("negative_slope must be non-negative")
    return jnp.where(x >= 0, x, negative_slope * x)


__all__ = ["leaky_relu"]
