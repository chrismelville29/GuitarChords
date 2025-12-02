"""Weight initializers."""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from .. import types

Array = types.Array
PRNGKey = types.PRNGKey


def glorot_uniform(rng: PRNGKey, shape: Sequence[int], dtype: jnp.dtype = jnp.float32) -> Array:
    fan_in, fan_out = shape[-2], shape[-1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(rng, shape, dtype=dtype, minval=-limit, maxval=limit)


def he_normal(rng: PRNGKey, shape: Sequence[int], dtype: jnp.dtype = jnp.float32) -> Array:
    fan_in = shape[-2]
    std = jnp.sqrt(2.0 / fan_in)
    return std * jax.random.normal(rng, shape, dtype=dtype)


def he_normal_conv2d(rng: PRNGKey, shape: Sequence[int], dtype: jnp.dtype = jnp.float32) -> Array:
    """Kaiming/MSRA init adapted for Conv2D kernels."""
    if len(shape) != 4:
        raise ValueError("he_normal_conv2d expects a 4D conv kernel shape (kh, kw, cin, cout)")
    kh, kw, in_channels, _ = shape
    fan_in = kh * kw * in_channels
    std = jnp.sqrt(2.0 / fan_in)
    return std * jax.random.normal(rng, shape, dtype=dtype)


def bias_zeros(_: PRNGKey, shape: Sequence[int], dtype: jnp.dtype = jnp.float32) -> Array:
    return jnp.zeros(shape, dtype=dtype)
