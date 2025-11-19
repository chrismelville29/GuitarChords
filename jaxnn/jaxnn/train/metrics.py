"""Evaluation helpers."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import types

Array = types.Array


def accuracy(logits: Array, labels: Array) -> Array:
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)
