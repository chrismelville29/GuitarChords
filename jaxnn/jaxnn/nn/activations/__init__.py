"""Activation registry with one-function-per-file organization."""
from __future__ import annotations

from typing import Callable, Mapping

from .. import types
from .gelu import gelu
from .leaky_relu import leaky_relu
from .relu import relu
from .tanh import tanh

Array = types.Array
Activation = Callable[[Array], Array]

NAMED_ACTIVATIONS: Mapping[str, Activation] = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "gelu": gelu,
    "tanh": tanh,
}


def get_activation(name: str) -> Activation:
    try:
        return NAMED_ACTIVATIONS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown activation '{name}'. Registered: {tuple(NAMED_ACTIVATIONS)}") from exc


__all__ = ["gelu", "relu", "leaky_relu", "tanh", "NAMED_ACTIVATIONS", "get_activation"]
