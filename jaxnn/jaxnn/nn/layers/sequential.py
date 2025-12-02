"""Sequential container layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import jax

from ... import types
from . import base

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


@dataclass(frozen=True)
class Sequential(base.Layer):
    layers: Tuple[base.Layer, ...]
    split_rngs: bool = False

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("Sequential requires at least one sub-layer")

    def init(self, rng: PRNGKey) -> Params:
        keys = jax.random.split(rng, len(self.layers))
        params = tuple(layer.init(key) for layer, key in zip(self.layers, keys))
        return params

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
        return_updated_params: bool = False,
    ) -> Array | tuple[Array, Params]:
        if not isinstance(params, (list, tuple)):
            raise TypeError("Sequential params must be a sequence aligned with sub-layers")
        if len(params) != len(self.layers):
            raise ValueError("Params sequence must match number of layers")

        rngs = None
        if rng is not None and self.split_rngs:
            rngs = jax.random.split(rng, len(self.layers))

        outputs = inputs
        updated_params_list = []
        any_updates = False

        for idx, (layer, layer_params) in enumerate(zip(self.layers, params)):
            layer_rng = rngs[idx] if rngs is not None else None
            out = layer.apply(layer_params, outputs, rng=layer_rng, is_training=is_training)

            if isinstance(out, tuple):
                outputs, layer_params_updated = out
                any_updates = True
            else:
                outputs = out
                layer_params_updated = layer_params

            updated_params_list.append(layer_params_updated)

        if return_updated_params or any_updates:
            return outputs, tuple(updated_params_list)
        return outputs


__all__ = ["Sequential"]
