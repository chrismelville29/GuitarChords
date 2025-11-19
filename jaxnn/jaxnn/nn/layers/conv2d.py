"""Conv2D layer placeholder."""
from __future__ import annotations

from dataclasses import dataclass

from ... import types
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

    def init(self, rng: PRNGKey) -> Params:  # pragma: no cover - placeholder
        raise NotImplementedError("Conv2D.init is not implemented yet")

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:  # pragma: no cover - placeholder
        raise NotImplementedError("Conv2D.apply is not implemented yet")


__all__ = ["Conv2D"]
