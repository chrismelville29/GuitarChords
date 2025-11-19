"""Standard task losses."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import types

Array = types.Array


def cross_entropy_logits(logits: Array, labels: Array) -> Array:
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def mse(predictions: Array, targets: Array) -> Array:
    return jnp.mean(jnp.square(predictions - targets))
