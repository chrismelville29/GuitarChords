"""Standard task losses."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import types

Array = types.Array


def cross_entropy_logits(logits: Array, labels: Array, *, label_smoothing: float = 0.0) -> Array:
    """Mean cross-entropy for integer labels and unnormalized class logits.

    Args:
        logits: Array of shape ``(..., num_classes)`` containing raw model outputs.
        labels: Integer array broadcastable to ``logits.shape[:-1]``.
        label_smoothing: Optional smoothing factor in ``[0, 1)``. When non-zero,
            the one-hot labels are mixed with a uniform distribution.

    Returns:
        Scalar loss averaged across the leading dimensions.
    """
    if label_smoothing < 0.0 or label_smoothing >= 1.0:
        raise ValueError("label_smoothing must be in the range [0.0, 1.0)")

    num_classes = logits.shape[-1]
    one_hot = jax.nn.one_hot(labels, num_classes, dtype=logits.dtype)

    if label_smoothing:
        smooth = label_smoothing / num_classes
        one_hot = one_hot * (1.0 - label_smoothing) + smooth

    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


def mse(predictions: Array, targets: Array) -> Array:
    """Mean squared error between predictions and targets."""
    return jnp.mean(jnp.square(predictions - targets))


def nll_loss(log_probs: Array, targets: Array, reduction: str = "mean") -> Array:
    """Negative log-likelihood loss for integer class labels.

    Args:
        log_probs: Log-probabilities of shape ``(..., num_classes)`` (e.g., output of ``log_softmax``).
        targets: Integer labels broadcastable to ``log_probs.shape[:-1]``.
        reduction: One of ``\"mean\"``, ``\"sum\"``, or ``\"none\"``.

    Returns:
        Scalar loss (for ``mean``/``sum``) or per-example loss (for ``none``).
    """
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of {'mean', 'sum', 'none'}")

    n_classes = log_probs.shape[-1]
    if jnp.any((targets < 0) | (targets >= n_classes)):
        raise ValueError("targets must be within [0, num_classes)")

    gathered = jnp.take_along_axis(
        log_probs, jnp.expand_dims(targets, axis=-1), axis=-1
    ).squeeze(axis=-1)
    losses = -gathered

    if reduction == "none":
        return losses
    if reduction == "sum":
        return jnp.sum(losses)
    return jnp.mean(losses)


__all__ = ["cross_entropy_logits", "mse", "nll_loss"]
