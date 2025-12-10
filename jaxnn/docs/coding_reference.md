# Team Guide

This document lives next to the code so we always have a shared source of truth for how we want to build and maintain `jaxnn`.

## Our design pillars

1. **Everything important is a pytree.** Parameters, optimizer state, training batch dictionaries—if JAX can tree-map over it, so can we. That keeps transformations like `jit`, `grad`, `vmap`, and `pmap` happy.
2. **Functions over objects.** Public APIs return or accept plain pytrees, not mutable classes. Side effects live at the edges (data loading, logging, checkpoints), never inside JIT-compiled code paths.
3. **Pure init/apply split.** Every model exposes `init_fn(rng) -> params` and `apply_fn(params, inputs, **kwargs) -> outputs`. When stateful layers eventually show up, the functions will return `(outputs, new_state)` and keep the same spirit.
4. **RNGs are explicit.** If a layer needs randomness, it receives an RNG key and splits it locally. No hidden globals or implicit randomness inside JIT regions.

## Coding rules

- **Type hints everywhere.** Public functions must be annotated. Internal helpers should be annotated whenever the signature isn't obvious.
- **Docstrings for modules and public functions.** Keep them short and practical.
- **Treat params/state as immutable.** Copy or return new pytrees rather than mutating existing ones in-place. JAX transformations expect referential transparency.
- **Small, composable files.** Each layer lives in its own module under `jaxnn/nn/layers/`. Keep optimizers self-contained. Push shared helpers (tree math, typing) into `jaxnn/tree.py` and `jaxnn/types.py` so they can be reused.
- **No hiding control flow inside models.** If you need Python-side loops (curriculum logic, logging, etc.), keep them outside `jax.jit`.
- **Tests for every public API.** Add minimal `pytest` coverage under `tests/` whenever you launch a new feature. Focus on shape checking and numerical sanity.
- **Use _ for private functions.** Ensure private functions, variables and classes start with _ , avoid camel case for functions.

## Working style

- **Keep examples runnable.** Anything under `examples/` should be executable with `python -m examples.my_script` and use small dummy data unless otherwise noted.
- **Document new patterns.** If you introduce a new convention (say, handling mutable model state), record it here so the next teammate doesn't have to reverse engineer the workflow.
- **Prefer readable comments to clever code.** When logic isn't obvious, drop a short comment explaining the why, not the what.
- **Linting/formatting.** Stick to `ruff format` defaults (or Black-style formatting) and run `ruff check` before pushing once we add configuration. Until then, mirror the existing style—80-100 column width and double quotes for strings.
- **Dependencies.** Only add libraries we absolutely need. Everything in the main package should import fast; heavy dependencies belong in `examples/` or optional extras.

## How to add a new feature

1. **Sketch the pytree.** Decide what the parameters and any auxiliary state look like. Dig through `jaxnn/tree.py` to see if an existing helper does what you need.
2. **Write `init_*` and `apply_*`.** Follow the Dense/Sequential examples in `jaxnn/nn/layers/`. Keep RNG handling explicit.
3. **Add tests.** Place shape/value checks in `tests/`. Dummy RNG keys plus predictable inputs (e.g., zeros or ones) usually suffice.
4. **Update docs.** If the feature adds a new concept, mention it in `README.md` and extend this guide if it affects workflow expectations.

## PyTree quick reference

- Use `jax.tree_util.tree_map` for parameter-wise math (updates, weight decay, clipping).
- Reach for helpers in `jaxnn/tree.py` to count parameters, compute global norms, or clip gradients.
- When flattening manually, prefer `jax.tree_util.tree_flatten` + `tree_unflatten` rather than re-implementing recursion.
