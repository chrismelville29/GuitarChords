"""gelu activation."""
from __future__ import annotations

import jax.numpy as jnp

from ... import types
from .tanh import tanh

Array = types.Array


def gelu(x: Array) -> Array:
    tanh_input = jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1 + tanh(tanh_input))


__all__ = ["gelu"]
