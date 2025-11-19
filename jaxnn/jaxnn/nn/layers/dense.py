"""Dense/fully-connected layer."""
from __future__ import annotations

from dataclasses import dataclass

import jax

from ... import types
from .. import activations
from ..init import bias_zeros, glorot_uniform
from . import base

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


@dataclass(frozen=True)
class Dense(base.Layer):
    in_features: int
    out_features: int
    activation: str | None = "relu"
    w_init: types.Initializer = glorot_uniform
    b_init: types.Initializer = bias_zeros

    def init(self, rng: PRNGKey) -> Params:
        w_key, b_key = jax.random.split(rng)
        w = self.w_init(w_key, (self.in_features, self.out_features))
        b = self.b_init(b_key, (self.out_features,))
        return {"w": w, "b": b}

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng, is_training)  # unused but keep signature uniform
        outputs = inputs @ params["w"] + params["b"]
        if self.activation is None:
            return outputs
        return activations.get_activation(self.activation)(outputs)


__all__ = ["Dense"]
