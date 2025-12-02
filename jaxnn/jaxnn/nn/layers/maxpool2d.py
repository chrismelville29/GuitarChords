"""2D max pooling layer for NHWC inputs."""
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
class MaxPool2D(base.Layer):
    kernel_size: tuple[int, int]
    strides: tuple[int, int] | None = None
    padding: str = "VALID"

    def __post_init__(self) -> None:
        if len(self.kernel_size) != 2 or any(k <= 0 for k in self.kernel_size):
            raise ValueError("kernel_size must be a 2-tuple of positive ints")

        if self.strides is None:
            object.__setattr__(self, "strides", self.kernel_size)
        else:
            if len(self.strides) != 2 or any(s <= 0 for s in self.strides):
                raise ValueError("strides must be a 2-tuple of positive ints")

        if isinstance(self.padding, str):
            pad_type = self.padding.upper()
            if pad_type not in {"SAME", "VALID"}:
                raise ValueError("padding must be 'SAME' or 'VALID'")
        else:
            raise TypeError("padding must be a string ('SAME' or 'VALID')")

    def init(self, rng: PRNGKey) -> Params:
        _ = rng
        return {}

    def _init_value_for_dtype(self, dtype) -> Array:
        if jnp.issubdtype(dtype, jnp.inexact):
            return jnp.array(-jnp.inf, dtype=dtype)
        info = jnp.iinfo(dtype)
        return jnp.array(info.min, dtype=dtype)

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (params, rng, is_training)
        if inputs.ndim != 4:
            raise ValueError("MaxPool2D expects NHWC inputs with rank 4 (batch, h, w, c)")

        window = (1, self.kernel_size[0], self.kernel_size[1], 1)
        strides = (1, self.strides[0], self.strides[1], 1)
        init_value = self._init_value_for_dtype(inputs.dtype)

        return jax.lax.reduce_window(
            inputs,
            init_value,
            jax.lax.max,
            window_dimensions=window,
            window_strides=strides,
            padding=self.padding.upper(),
        )


__all__ = ["MaxPool2D"]
