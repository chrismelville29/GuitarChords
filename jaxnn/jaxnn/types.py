"""Shared typing aliases for the jaxnn library."""
from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Protocol, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp

Array = jax.Array
PRNGKey = jax.Array
PyTree = Any

Params = PyTree
State = PyTree
OptState = PyTree
Batch = Mapping[str, Array]
MutableBatch = MutableMapping[str, Array]

T = TypeVar("T")

LossFn = Callable[[Params, Batch], Array]
ApplyFn = Callable[[Params, Array], Array]


class Initializer(Protocol):
    def __call__(self, rng: PRNGKey, shape: Sequence[int], dtype: jnp.dtype = jnp.float32) -> Array: ...


class Optimizer(Protocol):
    def init(self, params: Params) -> OptState: ...

    def update(self, grads: Params, state: OptState, params: Params) -> Tuple[Params, OptState]: ...
