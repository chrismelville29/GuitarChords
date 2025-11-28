"""tanh activation."""
from __future__ import annotations

import jax.numpy as jnp

from ... import types

Array = types.Array


def tanh(x: Array) -> Array:
    positive_exp = jnp.exp(x)
    negative_exp = jnp.exp(-1 * x)
    return (positive_exp - negative_exp) / (positive_exp + negative_exp)


__all__ = ["tanh"]
