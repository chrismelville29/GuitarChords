"""Layer package with one-file-per-layer organization."""
from __future__ import annotations

from typing import Sequence

from .activation import Activation
from .base import Layer
from .batchnorm import BatchNorm
from .conv2d import Conv2D
from .dense import Dense
from .dropout import Dropout
from .flatten import Flatten
from .maxpool2d import MaxPool2D
from .sequential import Sequential

__all__ = [
    "Layer",
    "Activation",
    "Dense",
    "BatchNorm",
    "Dropout",
    "Flatten",
    "Sequential",
    "Conv2D",
    "MaxPool2D",
    "make_mlp",
]


def make_mlp(layer_sizes: Sequence[int], activation: str | None = "relu") -> Sequential:
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must contain at least input and output dims")
    dense_layers = []
    for idx, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        is_last = idx == len(layer_sizes) - 2
        act = None if is_last else activation
        dense_layers.append(Dense(fan_in, fan_out, activation=act))
    return Sequential(tuple(dense_layers))
