# Optimizer Implementation Guide

This library keeps optimizers small, functional, and pytree-friendly. Each optimizer must expose two pure methods:

- `init(params) -> opt_state`: Build any auxiliary state you need to update parameters (e.g., momentum buffers, EMA statistics, step counters). The returned `opt_state` must be a pytree whose leaves mirror the structure of `params` so JAX transforms (jit, grad, pmap) work seamlessly.
- `update(grads, opt_state, params) -> (new_params, new_opt_state)`: Compute parameter updates from gradients and return both the updated parameters **and** the updated optimizer state. Avoid in-place mutation—always return new pytrees.

## What belongs in `opt_state`?

`opt_state` should contain everything required to reproduce the optimizer's behavior on the next step, typically:

- **Step counter:** An integer tracking how many updates have been applied (needed for bias correction in Adam/RMSProp-style methods).
- **Per-parameter statistics:** Trees that match `params` storing extra accumulators. Examples: momentum buffers for SGD with momentum, first/second moment estimates for Adam, running average of gradient magnitudes for AdaGrad, etc.
- **Hyperparameters that may change over time:** Learning rate schedules, weight decay rates, or clipping thresholds if they are updated internally.

Keep `opt_state` lightweight and deterministic—no random keys or external references.

## Minimal template

```python
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from jaxnn.optim import base

@dataclass(frozen=True)
class MyOptimizer(base.Optimizer):
    learning_rate: float

    def init(self, params):
        # Mirror the parameter tree with zeros for any accumulators
        return {"acc": jax.tree_util.tree_map(jnp.zeros_like, params), "step": 0}

    def update(self, grads, state, params):
        updates = base.scale_updates(grads, self.learning_rate)  # -lr * grad
        new_params = base.apply_updates(params, updates)
        new_acc = jax.tree_util.tree_map(lambda a, g: a + g, state["acc"], grads)
        new_state = {"acc": new_acc, "step": state["step"] + 1}
        return new_params, new_state
```

`base.scale_updates` and `base.apply_updates` are convenience helpers that already handle pytree arithmetic and broadcasting.

## Example: SGD with momentum

- **Opt state:** `{"momentum": tree_like_params, "step": int}`
- **Update rule:**
  1. `velocity = momentum * beta + grads`
  2. `params = params - lr * velocity`
  3. Increment `step` and store `velocity` back into the state.

## Example: Adam (suggested fields)

- **Opt state:** `{"m": first_moment, "v": second_moment, "step": int}`
- **Update rule:**
  1. Update `m` and `v` with exponential moving averages of the gradients and squared gradients.
  2. Compute bias-corrected estimates using `step`.
  3. Form updates `-lr * m_hat / (sqrt(v_hat) + eps)` and apply with `base.apply_updates`.

## Learning rate schedules

Both `SGD` and `Adam` accept an optional `lr_schedule` callable with signature `(step: int) -> lr`. When provided, the schedule output overrides the static `learning_rate`/`lr` for that step and increments the stored `step` counter automatically.

Use `jaxnn.optim.schedule.cosine_decay_schedule` to anneal from an initial LR to a final LR:

```python
from jaxnn.optim.sgd import SGD
from jaxnn.optim.schedule import cosine_decay_schedule

schedule = cosine_decay_schedule(initial_lr=0.1, final_lr=0.0, total_steps=10_000)
optimizer = SGD(learning_rate=0.1, momentum=0.9, lr_schedule=schedule)
```

## Testing checklist

- **Shapes:** `opt_state` leaves match `params` leaves for all supported parameter shapes.
- **Determinism:** Re-running `update` with the same inputs yields identical outputs.
- **Numerical sanity:** On a tiny model (e.g., the MNIST MLP example), the loss decreases for a few steps.
- **JIT compatibility:** Wrap `update` in `jax.jit` during tests to ensure it works with transformations.

Following this pattern keeps optimizers interchangeable and easy to reason about across the rest of the `jaxnn` codebase.
