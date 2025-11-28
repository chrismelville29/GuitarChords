"""2D convolution layer implemented from first principles (no ``lax.conv``)."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ... import types
from .. import init as initializers
from . import base

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


@dataclass(frozen=True)
class Conv2D(base.Layer):
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    strides: tuple[int, int] = (1, 1)
    padding: str = "SAME"

    def __post_init__(self) -> None:
        if len(self.kernel_size) != 2:
            raise ValueError("kernel_size must be a 2-tuple of ints")
        if len(self.strides) != 2:
            raise ValueError("strides must be a 2-tuple of ints")
        if any(k <= 0 for k in self.kernel_size):
            raise ValueError("kernel_size entries must be positive")
        if any(s <= 0 for s in self.strides):
            raise ValueError("stride entries must be positive")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")

    def _conv_kernel_init(self, rng: PRNGKey) -> Array:
        """Glorot-uniform for conv kernels that accounts for spatial extent."""
        kh, kw = self.kernel_size
        fan_in = kh * kw * self.in_channels
        fan_out = kh * kw * self.out_channels
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(
            rng,
            (kh, kw, self.in_channels, self.out_channels),
            minval=-limit,
            maxval=limit,
        )

    def init(self, rng: PRNGKey) -> Params:
        w_key, b_key = jax.random.split(rng)
        w = self._conv_kernel_init(w_key)
        b = initializers.bias_zeros(b_key, (self.out_channels,))
        return {"w": w, "b": b}

    def _compute_padding(self, in_h: int, in_w: int) -> tuple[int, int, int, int, int, int]:
        """Return (pad_top, pad_bottom, pad_left, pad_right, out_h, out_w)."""
        kh, kw = self.kernel_size
        sh, sw = self.strides
        if isinstance(self.padding, str):
            pad_type = self.padding.upper()
            if pad_type not in {"SAME", "VALID"}:
                raise ValueError("padding must be 'SAME' or 'VALID'")
            if pad_type == "SAME":
                out_h = (in_h + sh - 1) // sh
                out_w = (in_w + sw - 1) // sw
                pad_along_height = max((out_h - 1) * sh + kh - in_h, 0)
                pad_along_width = max((out_w - 1) * sw + kw - in_w, 0)
                pad_top = pad_along_height // 2
                pad_bottom = pad_along_height - pad_top
                pad_left = pad_along_width // 2
                pad_right = pad_along_width - pad_left
            else:  # VALID
                out_h = (in_h - kh) // sh + 1
                out_w = (in_w - kw) // sw + 1
                pad_top = pad_bottom = pad_left = pad_right = 0
        else:
            raise TypeError("padding must be a string ('SAME' or 'VALID')")
        if out_h <= 0 or out_w <= 0:
            raise ValueError("Kernel bigger than input after padding/strides")
        return pad_top, pad_bottom, pad_left, pad_right, out_h, out_w

    def _convolve_single_image(self, params: Params, x: Array) -> Array:
        """Convolve a single image (H, W, C_in) -> (H_out, W_out, C_out)."""
        kh, kw = self.kernel_size
        sh, sw = self.strides

        pad_top, pad_bottom, pad_left, pad_right, out_h, out_w = self._compute_padding(
            x.shape[0], x.shape[1]
        )
        x_padded = jnp.pad(
            x,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

        def convolve_at(i: int, j: int) -> Array:
            h_start = i * sh
            w_start = j * sw
            # Use dynamic slicing to keep indices JIT-compatible inside vmaps.
            window = jax.lax.dynamic_slice(x_padded, (h_start, w_start, 0), (kh, kw, x.shape[-1]))
            # Sum over spatial + in_channels to produce one vector per out_channel
            return jnp.tensordot(window, params["w"], axes=([0, 1, 2], [0, 1, 2])) + params["b"]

        # Vectorize across spatial locations
        js = jnp.arange(out_w)

        def row_fn(i: int) -> Array:
            return jax.vmap(lambda j: convolve_at(i, j))(js)

        is_ = jnp.arange(out_h)
        return jax.vmap(row_fn)(is_)

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng, is_training)  # present for API symmetry; unused here

        if inputs.ndim != 4:
            raise ValueError("Conv2D expects NHWC inputs with rank 4 (batch, h, w, c)")
        if inputs.shape[-1] != self.in_channels:
            raise ValueError(
                f"Input channels ({inputs.shape[-1]}) must match in_channels ({self.in_channels})"
            )

        # Vectorize over batch dimension.
        return jax.vmap(lambda img: self._convolve_single_image(params, img))(inputs)


__all__ = ["Conv2D"]
