"""Composable train/eval steps."""
from __future__ import annotations

from typing import Callable, Mapping

import jax

from .. import types
from ..optim import base
from . import metrics

Batch = types.Batch
Params = types.Params
ApplyFn = Callable[[Params, types.Array], types.Array]
Loss = Callable[[types.Array, types.Array], types.Array]


def make_train_step(apply_fn: ApplyFn, loss_fn: Loss, optimizer: base.Optimizer) -> Callable[[base.TrainState, Batch], tuple[base.TrainState, types.Array]]:
    @jax.jit
    def train_step(state: base.TrainState, batch: Batch) -> tuple[base.TrainState, types.Array]:
        def loss_with_params(params: Params) -> types.Array:
            logits = apply_fn(params, batch["x"])
            return loss_fn(logits, batch["y"])

        loss_value, grads = jax.value_and_grad(loss_with_params)(state.params)
        new_params, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_state = base.TrainState(new_params, new_opt_state)
        return new_state, loss_value

    return train_step


def make_eval_step(apply_fn: ApplyFn, loss_fn: Loss) -> Callable[[Params, Batch], Mapping[str, types.Array]]:
    @jax.jit
    def eval_step(params: Params, batch: Batch) -> Mapping[str, types.Array]:
        logits = apply_fn(params, batch["x"])
        loss_value = loss_fn(logits, batch["y"])
        acc = metrics.accuracy(logits, batch["y"])
        return {"loss": loss_value, "accuracy": acc}

    return eval_step
