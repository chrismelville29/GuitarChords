"""ReLU activation."""
from __future__ import annotations

import jax.numpy as jnp

from ... import types

Array = types.Array


def relu(x: Array) -> Array:
    return jnp.maximum(x, 0)


__all__ = ["relu"]
