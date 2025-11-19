"""Minimal example wiring the building blocks together."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxnn.nn import losses, model
from jaxnn.optim import base
from jaxnn.optim.sgd import SGD
from jaxnn.train import loop


def main() -> None:
    rng = jax.random.PRNGKey(0)
    init_fn, apply_fn = model.build_mlp((784, 256, 256, 10))
    params = init_fn(rng)

    optimizer = SGD(learning_rate=1e-2, momentum=0.9)
    opt_state = optimizer.init(params)
    state = base.TrainState(params, opt_state)

    train_step = loop.make_train_step(apply_fn, losses.cross_entropy_logits, optimizer)
    eval_step = loop.make_eval_step(apply_fn, losses.cross_entropy_logits)

    dummy_batch = {
        "x": jnp.ones((32, 784), dtype=jnp.float32),
        "y": jnp.zeros((32,), dtype=jnp.int32),
    }

    state, loss_value = train_step(state, dummy_batch)
    metrics = eval_step(state.params, dummy_batch)

    print("loss", float(loss_value))
    print({k: float(v) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
