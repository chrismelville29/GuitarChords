"""Layer package with one-file-per-layer organization."""
from __future__ import annotations

from typing import Sequence

from .base import Layer
from .conv2d import Conv2D
from .dense import Dense
from .sequential import Sequential

__all__ = [
    "Layer",
    "Dense",
    "Sequential",
    "Conv2D",
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
