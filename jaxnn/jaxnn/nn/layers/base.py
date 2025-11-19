"""Abstract base class for layers."""
from __future__ import annotations

import abc
from typing import Any

from ... import types

Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


class Layer(abc.ABC):
    """Base class every functional layer should extend."""

    name: str | None = None

    @abc.abstractmethod
    def init(self, rng: PRNGKey) -> Params:
        """Produce initial parameters for this layer."""

    @abc.abstractmethod
    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        """Run the layer's forward pass."""

    def __call__(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        return self.apply(params, inputs, rng=rng, is_training=is_training)

    def init_and_apply(
        self,
        init_rng: PRNGKey,
        inputs: Array,
        *,
        apply_rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> tuple[Params, Array]:
        params = self.init(init_rng)
        outputs = self.apply(params, inputs, rng=apply_rng, is_training=is_training)
        return params, outputs


class PlaceholderLayer(Layer):
    """Convenience base for layers we still need to implement."""

    def init(self, rng: PRNGKey) -> Params:  # pragma: no cover - placeholder logic
        raise NotImplementedError("Implement init() in subclasses")

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:  # pragma: no cover - placeholder logic
        raise NotImplementedError("Implement apply() in subclasses")


__all__ = ["Layer", "PlaceholderLayer"]
