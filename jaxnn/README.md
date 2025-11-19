# jaxnn


## Project layout

```
jaxnn/
  __init__.py
  types.py          # shared type aliases
  tree.py           # pytree helpers
  nn/
    __init__.py
    activations/
      __init__.py
      relu.py
      gelu.py
      tanh.py
    layers/
      __init__.py
      base.py
      dense.py
      conv2d.py
      sequential.py
    init.py         # weight initializers
    losses.py       # task losses
    model.py        # convenience helpers for stacking layers
  optim/
    __init__.py
    base.py         # Optimizer protocol + utilities
    sgd.py          # SGD implementation
  train/
    __init__.py
    loop.py         # reusable train/eval steps
    metrics.py      # accuracy and other metrics
examples/
  mnist_mlp.py      # example script tying everything together
docs/
  team_guide.md     # collaboration + coding expectations
tests/
  ...               # pytest-based regression/unit tests
```

See `docs/team_guide.md` for coding rules, workflow conventions, and guidelines on how to extend the library with new modules.
