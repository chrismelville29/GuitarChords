"""Tiny convenience wrappers around JAX tree utilities."""
from __future__ import annotations

from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
from jax import tree_map, tree_util

from . import types

PyTree = types.PyTree
Array = types.Array


def tree_size(tree: PyTree) -> int:
    """Count leaves in a pytree."""
    return len(tree_util.tree_leaves(tree))


def tree_apply(fn: Callable[..., PyTree], *trees: PyTree) -> PyTree:
    """Map ``fn`` over one or more pytrees with shared structure."""
    return tree_map(fn, *trees)


def tree_l2_norm(tree: PyTree) -> Array:
    """Compute the L2 norm across all leaves in ``tree``."""
    leaves = tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def clip_by_global_norm(tree: PyTree, max_norm: float) -> PyTree:
    """Scale all leaves so the pytree has at most ``max_norm`` norm."""
    norm = tree_l2_norm(tree)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return tree_map(lambda x: x * scale, tree)
