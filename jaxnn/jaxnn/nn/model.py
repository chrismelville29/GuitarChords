"""Convenience helpers for composing layers into init/apply pairs."""
from __future__ import annotations

from typing import Callable, Tuple

from .. import types
from . import layers

InitFn = Callable[[types.PRNGKey], types.Params]
ApplyFn = Callable[[types.Params, types.Array], types.Array]


def build_mlp(layer_sizes: Tuple[int, ...], activation: str | None = "relu") -> Tuple[InitFn, ApplyFn]:
    network = layers.make_mlp(layer_sizes, activation=activation)

    def init_fn(rng: types.PRNGKey) -> types.Params:
        return network.init(rng)

    def apply_fn(params: types.Params, x: types.Array) -> types.Array:
        return network.apply(params, x)

    return init_fn, apply_fn


def build_mnist_cnn() -> Tuple[InitFn, ApplyFn]:
    """Small CNN tuned for 28x28 MNIST digits (NHWC inputs)."""
    network = layers.Sequential(
        (
            layers.Conv2D(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
            ),
            layers.BatchNorm(num_features=32),
            layers.Activation("relu"),
            layers.Conv2D(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
            ),
            layers.BatchNorm(num_features=64),
            layers.Activation("relu"),
            layers.Flatten(),
            layers.Dense(in_features=7 * 7 * 64, out_features=128, activation="relu"),
            layers.Dense(in_features=128, out_features=10, activation=None),
        )
    )

    def init_fn(rng: types.PRNGKey) -> types.Params:
        return network.init(rng)

    def apply_fn(params: types.Params, x: types.Array, *, is_training: bool = True) -> types.Array:
        return network.apply(params, x, is_training=is_training)

    return init_fn, apply_fn
