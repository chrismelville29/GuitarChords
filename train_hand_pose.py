"""Hand-pose classifier training script using jaxnn.

The model consumes wrist-normalized landmark .npy files produced by
``hand_pose.py``. Each file contains 20 (x, y, z) points (wrist removed).
We reshape them into a 4x5 grid (joints x fingers) with 3 channels to feed a
small CNN or an optional tiny ResNet-style model that uses leaky ReLU
activations and Adam.

The script also balances uneven chord data by (a) oversampling each class to
the same per-epoch count and (b) applying class-weighted cross-entropy.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore

import jax
import jax.numpy as jnp
import numpy as np

from jaxnn import types
from jaxnn.nn import activations
from jaxnn.nn import layers
from jaxnn.nn.layers import base as layer_base
from jaxnn.nn.layers.batchnorm import BatchNorm
from jaxnn.optim import base as optim_base
from jaxnn.optim.adam import Adam
from jaxnn.optim.schedule import cosine_decay_schedule
from jaxnn.models.hand_graph_attention import HandGraphAttentionNetwork, hand_graph_features

try:  # Prefer PyTorch's writer when available.
    from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    TorchSummaryWriter = None  # type: ignore

try:
    from tensorboardX import SummaryWriter as TensorboardXSummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    TensorboardXSummaryWriter = None  # type: ignore

try:
    from tensorboard.summary.writer.event_file_writer import EventFileWriter as TensorBoardEventFileWriter
    from tensorboard.compat.proto import event_pb2, summary_pb2
except ImportError:  # pragma: no cover - optional dependency
    TensorBoardEventFileWriter = None  # type: ignore
    event_pb2 = summary_pb2 = None  # type: ignore


Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


class _EventFileSummaryWriter:
    """Fallback writer that directly emits TensorBoard event files."""

    def __init__(self, log_dir: Path):
        if TensorBoardEventFileWriter is None or event_pb2 is None or summary_pb2 is None:
            raise RuntimeError("TensorBoard event writer backend is unavailable.")
        self._writer = TensorBoardEventFileWriter(str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        event = event_pb2.Event(wall_time=time.time(), step=int(step))
        event.summary.value.add(tag=tag, simple_value=float(value))
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def _create_summary_writer(log_dir: Path | None):
    if log_dir is None:
        raise RuntimeError(
            "TensorBoard logging is mandatory; log_dir must be resolved before training."
        )

    resolved = log_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)

    writer = None
    if TorchSummaryWriter is not None:
        writer = TorchSummaryWriter(log_dir=str(resolved))
    elif TensorboardXSummaryWriter is not None:
        writer = TensorboardXSummaryWriter(log_dir=str(resolved))
    elif TensorBoardEventFileWriter is not None:
        writer = _EventFileSummaryWriter(resolved)
    if writer is None:
        raise RuntimeError(
            "TensorBoard logging requires `tensorboard`, `tensorboardX`, or `torch` to be installed."
        )

    print(f"Logging TensorBoard summaries to {resolved}")
    print(f"Launch TensorBoard with: tensorboard --logdir {resolved}")
    return writer


def _resolve_log_dir(cli_value: Path | None) -> Path:
    """Select the TensorBoard log directory with CLI > env > timestamp priority."""

    if cli_value is not None:
        return cli_value

    env_override = os.environ.get("TENSORBOARD_LOGDIR")
    if env_override:
        return Path(env_override)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("runs") / timestamp


def _save_checkpoint(
    path: Path,
    state: optim_base.TrainState,
    *,
    epoch: int,
    global_step: int,
    best_val_acc: float,
    best_epoch: int,
    class_names: Sequence[str],
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    model_type: str,
    representation: str,
    include_wrist: bool,
    data_rng_state: dict | None,
    swa_params: Params | None,
    swa_count: int,
) -> None:
    """Persist training state for reproducible resumption/evaluation."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "params": jax.device_get(state.params),
        "opt_state": jax.device_get(state.opt_state),
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "class_names": list(class_names),
        "channel_mean": np.array(channel_mean),
        "channel_std": np.array(channel_std),
        "model_type": model_type,
        "representation": representation,
        "include_wrist": include_wrist,
        "data_rng_state": data_rng_state,
        "swa_params": jax.device_get(swa_params) if swa_params is not None else None,
        "swa_count": int(swa_count),
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint dictionary from disk."""

    with open(path.expanduser().resolve(), "rb") as f:
        return pickle.load(f)


# ----------------------------- Data utilities ----------------------------- #


def _augment_landmarks(landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply rotation/scale/translation/noise jitter to landmark coordinates."""

    augmented = np.array(landmarks, dtype=np.float32, copy=True)

    angle = rng.uniform(-0.15, 0.15)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    augmented = augmented @ rot_matrix.T

    scale = rng.uniform(0.95, 1.05)
    augmented = augmented * scale

    translation = rng.uniform(-0.02, 0.02, size=3)
    augmented = augmented + translation

    noise = rng.normal(0, 0.005, size=augmented.shape)
    augmented = augmented + noise

    return augmented.astype(np.float32)


def _finger_grid(
    landmarks: np.ndarray,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Reshape 20x3 landmarks into (4, 5, 3): 4 joints per finger x 5 fingers.

    The MediaPipe landmark order (after dropping the wrist) is grouped by finger
    in contiguous blocks of four points: thumb (1-4), index (5-8), middle
    (9-12), ring (13-16), pinky (17-20).

    Args:
        landmarks: Hand landmarks of shape (20, 3)
        augment: Whether to apply data augmentation
        rng: Random number generator for augmentation
    """

    if landmarks.shape != (20, 3):
        raise ValueError(f"Expected landmarks with shape (20, 3), got {landmarks.shape}")

    coords = np.array(landmarks, dtype=np.float32, copy=True)
    if augment and rng is not None:
        coords = _augment_landmarks(coords, rng)

    joints_per_finger = []
    for finger_idx in range(5):
        start = finger_idx * 4
        joints_per_finger.append(coords[start : start + 4])

    # Stack fingers along the second axis: (4 joints, 5 fingers, 3 channels)
    grid = np.stack(joints_per_finger, axis=1).astype(np.float32)

    if norm_stats is not None:
        mean, std = norm_stats
        grid = (grid - mean) / (std + 1e-6)

    return grid


def _hand_graph_inputs(
    landmarks: np.ndarray,
    *,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
    include_wrist: bool = True,
) -> np.ndarray:
    """Prepare (N, F) graph features compatible with HandGraphAttentionNetwork."""

    coords = np.array(landmarks, dtype=np.float32, copy=True)
    if include_wrist:
        if coords.shape == (20, 3):
            coords = np.vstack([np.zeros((1, 3), dtype=coords.dtype), coords])
        elif coords.shape != (21, 3):
            raise ValueError(f"Expected 20 or 21 points, got shape {coords.shape}")
    else:
        if coords.shape == (21, 3):
            coords = coords[1:]
        elif coords.shape != (20, 3):
            raise ValueError(f"Expected 20 or 21 points, got shape {coords.shape}")

    if augment and rng is not None:
        coords = _augment_landmarks(coords, rng)

    return hand_graph_features(coords, include_wrist=include_wrist, norm_stats=norm_stats)


def _prepare_landmark_sample(
    landmarks: np.ndarray,
    *,
    representation: str,
    augment: bool,
    rng: np.random.Generator | None,
    norm_stats: tuple[np.ndarray, np.ndarray] | None,
    include_wrist: bool,
) -> np.ndarray:
    if representation == "grid":
        return _finger_grid(landmarks, augment=augment, rng=rng, norm_stats=norm_stats)
    if representation == "graph":
        return _hand_graph_inputs(
            landmarks,
            augment=augment,
            rng=rng,
            norm_stats=norm_stats,
            include_wrist=include_wrist,
        )
    raise ValueError(f"Unknown representation: {representation}")


def _collect_split(split_dir: Path, class_names: Sequence[str] | None = None) -> tuple[List[tuple[Path, int]], List[str]]:
    """Return (samples, class_names) for a split directory of .npy files."""

    split_dir = split_dir.expanduser().resolve()

    discovered = [p.name for p in split_dir.iterdir() if p.is_dir()]
    if class_names is None:
        class_names = sorted(discovered)
    # IMPORTANT: Don't filter class_names! Keep all classes even if some have 0 samples
    # to maintain consistent label indices across train/valid/test splits

    label_map = {name: idx for idx, name in enumerate(class_names)}
    samples: list[tuple[Path, int]] = []

    # Only iterate over classes that actually exist in this split
    for cls in class_names:
        cls_dir = split_dir / cls
        if cls_dir.is_dir():
            for npy_file in sorted(cls_dir.glob("*.npy")):
                samples.append((npy_file, label_map[cls]))

    return samples, list(class_names)


def _compute_class_weights(counts: Counter) -> np.ndarray:
    """Inverse frequency weights normalized to mean 1.0."""

    if not counts:
        return np.array([], dtype=np.float32)

    num_classes = len(counts)
    total = sum(counts.values())
    weights = {cls: total / (num_classes * count) for cls, count in counts.items()}
    weights_arr = np.array([weights[i] for i in range(num_classes)], dtype=np.float32)
    return weights_arr / np.mean(weights_arr)


def _compute_channel_mean_std(
    samples: Sequence[tuple[Path, int]],
    *,
    include_wrist: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel (x,y,z) mean/std with optional wrist reinsertion."""

    total_points = 0
    sum_channels = np.zeros(3, dtype=np.float64)
    sumsq_channels = np.zeros(3, dtype=np.float64)

    for path, _ in samples:
        arr = np.load(path)  # shape (20, 3) unless dataset exported with wrist
        if include_wrist:
            if arr.shape[0] == 20:
                arr = np.vstack([np.zeros((1, arr.shape[1]), dtype=arr.dtype), arr])
            elif arr.shape[0] != 21:
                raise ValueError(f"Expected 20 or 21 landmarks, got {arr.shape}")
        sum_channels += arr.sum(axis=0)
        sumsq_channels += np.square(arr).sum(axis=0)
        total_points += arr.shape[0]

    if total_points == 0:
        raise ValueError("No samples available to compute normalization stats.")

    mean = sum_channels / total_points
    var = sumsq_channels / total_points - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def _effective_num_class_weights(counts: Counter, beta: float = 0.99) -> np.ndarray:
    """Class-balanced weights from 'Effective Number of Samples' (Cui et al.).

    beta close to 1.0 smooths the weights and avoids the extreme ratios of pure
    inverse-frequency weighting.
    """
    if not counts:
        return np.array([], dtype=np.float32)

    weights = []
    for idx in range(len(counts)):
        n = counts[idx]
        eff_num = 1.0 - (beta ** n)
        weight = (1.0 - beta) / max(eff_num, 1e-8)
        weights.append(weight)

    weights_arr = np.array(weights, dtype=np.float32)
    return weights_arr / np.mean(weights_arr)


def _balanced_epoch_samples(class_to_files: Dict[int, List[Path]], rng: np.random.Generator) -> list[tuple[Path, int]]:
    """Oversample classes to the same count for a balanced epoch."""

    max_count = max(len(files) for files in class_to_files.values())
    epoch_samples: list[tuple[Path, int]] = []

    for label, files in class_to_files.items():
        if not files:
            continue
        reps = math.ceil(max_count / len(files))
        oversampled = rng.choice(files, size=max_count, replace=True)
        epoch_samples.extend([(Path(p), label) for p in oversampled])

    rng.shuffle(epoch_samples)
    return epoch_samples


def batch_iterator(
    samples: Sequence[tuple[Path, int]],
    class_weights: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
    oversample: bool = True,
    augment: bool = False,
    seed: int = 0,
    rng: np.random.Generator | None = None,
    norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
    representation: str = "grid",
    include_wrist: bool = False,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield batches with inputs, labels, and per-example weights."""

    rng = rng or np.random.default_rng(seed)

    # Organize files by label for oversampling
    class_to_files: dict[int, list[Path]] = defaultdict(list)
    for path, label in samples:
        class_to_files[label].append(path)

    while True:
        if oversample:
            working_set = _balanced_epoch_samples(class_to_files, rng)
        else:
            working_set = list(samples)
            if shuffle:
                rng.shuffle(working_set)

        for start in range(0, len(working_set), batch_size):
            batch_paths = working_set[start : start + batch_size]
            if not batch_paths:
                continue

            xs = []
            ys = []
            ws = []
            for path, label in batch_paths:
                arr = np.load(path)
                xs.append(
                    _prepare_landmark_sample(
                        arr,
                        representation=representation,
                        augment=augment,
                        rng=rng,
                        norm_stats=norm_stats,
                        include_wrist=include_wrist,
                    )
                )
                ys.append(label)
                ws.append(class_weights[label])

            batch_x = np.stack(xs, axis=0)
            batch_y = np.array(ys, dtype=np.int32)
            batch_w = np.array(ws, dtype=np.float32)

            yield {"x": batch_x, "y": batch_y, "w": batch_w}


# ----------------------------- Model definition --------------------------- #


@dataclass(frozen=True)
class ResidualBlock(layer_base.Layer):
    """Two 3x3 convs with an optional projection skip connection."""

    in_channels: int
    out_channels: int
    strides: tuple[int, int] = (1, 1)
    use_lax_conv: bool = False

    def __post_init__(self) -> None:
        if len(self.strides) != 2:
            raise ValueError("strides must be a 2-tuple")
        if not isinstance(self.use_lax_conv, bool):
            raise ValueError("use_lax_conv must be a bool")

    def init(self, rng: PRNGKey) -> Params:
        k1, k2, kproj = jax.random.split(rng, 3)

        conv1 = layers.Conv2D(self.in_channels, self.out_channels, (3, 3), strides=self.strides, use_lax=self.use_lax_conv)
        bn1 = BatchNorm(self.out_channels)
        conv2 = layers.Conv2D(self.out_channels, self.out_channels, (3, 3), strides=(1, 1), use_lax=self.use_lax_conv)
        bn2 = BatchNorm(self.out_channels)

        params = {
            "conv1": conv1.init(k1),
            "bn1": bn1.init(k1),
            "conv2": conv2.init(k2),
            "bn2": bn2.init(k2),
        }

        if self.in_channels != self.out_channels or self.strides != (1, 1):
            proj = layers.Conv2D(self.in_channels, self.out_channels, (1, 1), strides=self.strides, use_lax=self.use_lax_conv)
            bn_proj = BatchNorm(self.out_channels)
            params["proj"] = proj.init(kproj)
            params["bn_proj"] = bn_proj.init(kproj)

        return params

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng,)  # no stochastic layers inside

        conv1 = layers.Conv2D(
            self.in_channels,
            self.out_channels,
            (3, 3),
            strides=self.strides,
            use_lax=self.use_lax_conv,
            padding="SAME",
        )
        bn1 = BatchNorm(self.out_channels)
        conv2 = layers.Conv2D(
            self.out_channels,
            self.out_channels,
            (3, 3),
            strides=(1, 1),
            use_lax=self.use_lax_conv,
            padding="SAME",
        )
        bn2 = BatchNorm(self.out_channels)

        out = conv1.apply(params["conv1"], inputs)
        bn1_out = bn1.apply(params["bn1"], out, is_training=is_training)
        if isinstance(bn1_out, tuple):
            out, bn1_params = bn1_out
        else:
            out, bn1_params = bn1_out, params["bn1"]
        out = activations.leaky_relu(out)

        out = conv2.apply(params["conv2"], out)
        bn2_out = bn2.apply(params["bn2"], out, is_training=is_training)
        if isinstance(bn2_out, tuple):
            out, bn2_params = bn2_out
        else:
            out, bn2_params = bn2_out, params["bn2"]

        shortcut = inputs
        if "proj" in params:
            proj = layers.Conv2D(
                self.in_channels,
                self.out_channels,
                (1, 1),
                strides=self.strides,
                use_lax=self.use_lax_conv,
                padding="SAME",
            )
            bn_proj = BatchNorm(self.out_channels)
            shortcut = proj.apply(params["proj"], inputs)
            bn_proj_out = bn_proj.apply(params["bn_proj"], shortcut, is_training=is_training)
            if isinstance(bn_proj_out, tuple):
                shortcut, bn_proj_params = bn_proj_out
            else:
                shortcut, bn_proj_params = bn_proj_out, params["bn_proj"]

        out = activations.leaky_relu(out + shortcut)

        if not is_training:
            return out

        # Merge updated BN running stats back into the params dict for upstream handling.
        updated_params = {
            **params,
            "bn1": bn1_params,
            "bn2": bn2_params,
        }
        if "proj" in params:
            updated_params["bn_proj"] = bn_proj_params

        return out, updated_params


@dataclass(frozen=True)
class GlobalAvgPool2D(layer_base.Layer):
    """Spatially averages NHWC inputs to shape (batch, channels)."""

    def init(self, rng: PRNGKey) -> Params:
        _ = rng
        return {}

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (params, rng, is_training)
        if inputs.ndim != 4:
            raise ValueError("GlobalAvgPool2D expects NHWC inputs with rank 4 (batch, h, w, c)")
        return jnp.mean(inputs, axis=(1, 2))


def build_model(num_classes: int, *, use_lax_conv: bool = False) -> layers.Sequential:
    """Construct a balanced classifier for 4x5x3 inputs.

    Uses multiple conv layers with increasing channels and strategic dropout
    to balance capacity vs overfitting on the small, imbalanced dataset.
    """

    return layers.Sequential(
        (
            # Conv block 1
            layers.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3), padding="SAME", use_lax=use_lax_conv),
            layers.Activation("leaky_relu"),
            layers.Dropout(rate=0.2),

            # Conv block 2
            layers.Conv2D(in_channels=32, out_channels=64, kernel_size=(3, 3), padding="SAME", use_lax=use_lax_conv),
            layers.Activation("leaky_relu"),
            layers.Dropout(rate=0.2),

            # Conv block 3 with downsampling
            layers.Conv2D(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                use_lax=use_lax_conv,
            ),
            layers.Activation("leaky_relu"),
            layers.Dropout(rate=0.3),

            # Flatten and classify
            layers.Flatten(),
            layers.Dense(in_features=128 * 2 * 3, out_features=128, activation="leaky_relu"),
            layers.Dropout(rate=0.5),
            layers.Dense(in_features=128, out_features=64, activation="leaky_relu"),
            layers.Dropout(rate=0.5),
            layers.Dense(in_features=64, out_features=num_classes, activation=None),
        ),
        split_rngs=True,  # Enable RNG splitting for dropout
    )


def build_resnet_model(
    num_classes: int,
    *,
    use_lax_conv: bool = False,
    width_mult: float = 1.0,
    dropout_rate: float = 0.1,
) -> layers.Sequential:
    """Tiny ResNet-style classifier tailored to the 4x5 landmark grid.

    Layout (similar to CIFAR-style ResNet-20):
      stem: 3x3 conv -> LReLU
      stage0: 2 residual blocks @ C
      stage1: downsampling residual block (stride 2) + 1 block @ 2C
      stage2: downsampling residual block (stride 2) + 1 block @ 4C
      head: global average pool -> MLP
    """

    if width_mult <= 0:
        raise ValueError("width_mult must be > 0")

    base_channels = int(32 * width_mult)
    channel_schedule = (base_channels, base_channels * 2, base_channels * 4)
    stage_strides = [(1, 1), (2, 2), (2, 2)]

    layers_list: list[layer_base.Layer] = [
        layers.Conv2D(
            in_channels=3,
            out_channels=channel_schedule[0],
            kernel_size=(3, 3),
            padding="SAME",
            use_lax=use_lax_conv,
        ),
        layers.Activation("leaky_relu"),
        layers.Dropout(rate=0.1),
    ]

    prev_channels = channel_schedule[0]
    for stage_idx, (out_channels, stride) in enumerate(zip(channel_schedule, stage_strides)):
        for block_idx in range(2):
            block_stride = stride if block_idx == 0 else (1, 1)
            layers_list.append(
                ResidualBlock(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    strides=block_stride,
                    use_lax_conv=use_lax_conv,
                )
            )
            prev_channels = out_channels
        if dropout_rate > 0:
            layers_list.append(layers.Dropout(rate=dropout_rate if stage_idx < 2 else dropout_rate / 2))

    layers_list.extend(
        [
            GlobalAvgPool2D(),
            layers.Dense(in_features=channel_schedule[-1], out_features=128, activation="leaky_relu"),
            layers.Dropout(rate=0.4),
            layers.Dense(in_features=128, out_features=num_classes, activation=None),
        ]
    )

    return layers.Sequential(tuple(layers_list), split_rngs=True)


# ----------------------------- Training loop ------------------------------ #


def weighted_cross_entropy(logits: Array, labels: Array, weights: Array) -> Array:
    num_classes = logits.shape[-1]
    smooth_eps = 0.03
    one_hot = jax.nn.one_hot(labels, num_classes)
    smoothed = (1.0 - smooth_eps) * one_hot + smooth_eps / num_classes
    log_probs = jax.nn.log_softmax(logits)
    per_example = -jnp.sum(smoothed * log_probs, axis=-1)
    return jnp.mean(per_example * weights)


def l2_regularization(params: Params, weight_decay: float = 1e-4, *, mask: Params | None = None) -> Array:
    """Compute L2 regularization penalty on parameters.

    If ``mask`` is provided, it must mirror ``params`` and contain boolean arrays
    where ``True`` means "exclude this leaf from L2". This is used to skip
    BatchNorm running statistics.
    """

    if mask is None:
        leaves = jax.tree_util.tree_leaves(params)
        return weight_decay * sum(jnp.sum(jnp.square(p)) for p in leaves)

    def _masked_l2(p, m):
        return jnp.sum(jnp.square(p) * jnp.where(m, 0.0, 1.0))

    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(_masked_l2, params, mask))
    return weight_decay * sum(leaves)


def _running_stat_mask(params: Params) -> Params:
    """Return a boolean pytree marking BatchNorm running stats to exclude from L2/optimizer.

    Leaves corresponding to ``running_mean`` and ``running_var`` are ``True``;
    everything else is ``False`` with matching shape.
    """

    def _mark(node):
        if isinstance(node, dict):
            marked = {}
            for k, v in node.items():
                if k in {"running_mean", "running_var"}:
                    marked[k] = jnp.ones_like(v, dtype=bool)
                else:
                    marked[k] = _mark(v)
            return marked
        if isinstance(node, (list, tuple)):
            return type(node)(_mark(v) for v in node)
        return jnp.zeros_like(node, dtype=bool)

    return _mark(params)


def make_train_step(network: layers.Sequential, optimizer: Adam, weight_decay: float = 1e-4, *, stat_mask: Params | None = None):
    @jax.jit
    def train_step(state: optim_base.TrainState, batch: dict[str, Array], rng: PRNGKey) -> tuple[optim_base.TrainState, Array]:
        def loss_fn(params: Params):
            out = network.apply(params, batch["x"], rng=rng, is_training=True)
            if isinstance(out, tuple):
                logits, updated_params = out
            else:
                logits, updated_params = out, params

            ce_loss = weighted_cross_entropy(logits, batch["y"], batch["w"])
            l2_loss = l2_regularization(params, weight_decay, mask=stat_mask)
            return ce_loss + l2_loss, updated_params

        (loss_value, updated_params), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optim_base.apply_updates(state.params, updates)

        if stat_mask is not None:
            merged_params = jax.tree_util.tree_map(
                lambda new_p, upd_p, mask: jnp.where(mask, upd_p, new_p),
                new_params,
                updated_params,
                stat_mask,
            )
        else:
            merged_params = new_params

        new_state = optim_base.TrainState(merged_params, new_opt_state)
        return new_state, loss_value

    return train_step


def make_eval_step(network: layers.Sequential):
    @jax.jit
    def eval_step(params: Params, batch: dict[str, Array]) -> dict[str, Array]:
        logits = network.apply(params, batch["x"], rng=None, is_training=False)
        loss_value = weighted_cross_entropy(logits, batch["y"], batch["w"])
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == batch["y"])
        return {"loss": loss_value, "acc": acc}

    return eval_step


def run_training(
    data_root: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    model_type: str = "baseline",
    resnet_width_mult: float = 1.0,
    gat_hidden_dim: int = 96,
    gat_heads: int = 4,
    gat_layers: int = 3,
    gat_readout: str = "mean",
    lr_schedule: str = "constant",
    min_learning_rate: float = 1e-5,
    lr_decay_steps: int | None = None,
    lr_warmup_steps: int = 0,
    use_swa: bool = False,
    swa_start_epoch: int = 10,
    seed: int = 0,
    steps_per_epoch: int | None = None,
    *,
    oversample: bool = True,
    class_weighting: str = "effective",
    class_weight_beta: float = 0.99,
    restrict_to_valid_classes: bool = False,
    augment: bool = True,
    fast_training: bool = False,
    log_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    checkpoint_every: int = 0,
) -> None:
    if swa_start_epoch < 0:
        raise ValueError("swa_start_epoch must be non-negative")
    if min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative")
    if lr_decay_steps is not None and lr_decay_steps <= 0:
        raise ValueError("lr_decay_steps must be positive when provided")
    if lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")

    if checkpoint_dir is None:
        if log_dir is None:
            checkpoint_dir = Path("runs") / "checkpoints"
        else:
            checkpoint_dir = Path(log_dir) / "checkpoints"
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / "best.pkl"
    last_ckpt_path = checkpoint_dir / "last.pkl"

    train_dir = data_root / "train"
    valid_dir = data_root / "valid"

    train_classes = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    valid_classes = sorted([p.name for p in valid_dir.iterdir() if p.is_dir()])

    if restrict_to_valid_classes:
        class_names = [cls for cls in train_classes if cls in valid_classes]
        dropped = sorted(set(train_classes) - set(class_names))
        if dropped:
            print(f"Dropping train-only classes to match validation set: {dropped}")
    else:
        class_names = train_classes
        missing_in_valid = sorted(set(train_classes) - set(valid_classes))
        if missing_in_valid:
            print(
                "WARNING: Validation split is missing classes seen in training; "
                "metrics will penalize predictions for them. "
                f"Missing in valid/test: {missing_in_valid}"
            )

    train_samples, class_names = _collect_split(train_dir, class_names=class_names)
    valid_samples, _ = _collect_split(valid_dir, class_names=class_names)

    representation = "graph" if model_type == "gat" else "grid"
    include_wrist = representation == "graph"

    # Compute input normalization stats on the training set (channel-wise x/y/z).
    channel_mean, channel_std = _compute_channel_mean_std(
        train_samples, include_wrist=include_wrist
    )
    print("Input normalization (mean/std per channel):")
    print(f"  mean: {np.round(channel_mean, 4)}")
    print(f"  std : {np.round(channel_std, 4)}")

    # Track per-class availability and guard against missing labels.
    class_to_files: dict[int, list[Path]] = {idx: [] for idx in range(len(class_names))}
    for path, label in train_samples:
        class_to_files[label].append(path)

    missing = [class_names[idx] for idx, files in class_to_files.items() if not files]
    if missing:
        raise ValueError(f"No training examples found for classes: {missing}")

    train_counts = Counter({idx: len(files) for idx, files in class_to_files.items()})
    if class_weighting == "none":
        class_weights = np.ones(len(class_names), dtype=np.float32)
    elif class_weighting == "inverse":
        class_weights = _compute_class_weights(train_counts)
    elif class_weighting == "sqrt":
        class_weights = np.sqrt(_compute_class_weights(train_counts))
        class_weights = class_weights / np.mean(class_weights)
    elif class_weighting == "effective":
        class_weights = _effective_num_class_weights(train_counts, beta=class_weight_beta)
    else:
        raise ValueError(f"Unknown class_weighting mode: {class_weighting}")

    print(f"Loaded {len(train_samples)} train and {len(valid_samples)} valid examples across {len(class_names)} classes.")
    print("Training per-class counts:")
    for cls_idx, cls_name in enumerate(class_names):
        print(f"  {cls_name}: {train_counts.get(cls_idx, 0)}")

    # Check validation distribution
    valid_counts = Counter([label for _, label in valid_samples])
    print("\nValidation per-class counts:")
    for cls_idx, cls_name in enumerate(class_names):
        print(f"  {cls_name}: {valid_counts.get(cls_idx, 0)}")

    print(f"\nClass weighting mode: {class_weighting} (beta={class_weight_beta if class_weighting == 'effective' else 'n/a'})")
    print(f"Weights: {np.round(class_weights, 3).tolist()}")
    print(f"LR schedule: {lr_schedule} (min_lr={min_learning_rate})")
    if use_swa:
        print(f"SWA: enabled (start_epoch={swa_start_epoch}, waits for best-val epoch).")
    else:
        print("SWA: disabled.")

    max_train_count = max(train_counts.values())
    epoch_size = max_train_count * len(class_names) if oversample else len(train_samples)
    steps = steps_per_epoch or max(1, math.ceil(epoch_size / batch_size))

    if model_type == "baseline":
        network = build_model(num_classes=len(class_names), use_lax_conv=fast_training)
        model_desc = f"baseline (use_lax_conv={fast_training})"
    elif model_type == "resnet":
        network = build_resnet_model(
            num_classes=len(class_names),
            use_lax_conv=fast_training,
            width_mult=resnet_width_mult,
        )
        model_desc = f"resnet (width_mult={resnet_width_mult}, use_lax_conv={fast_training})"
    elif model_type == "gat":
        network = HandGraphAttentionNetwork(
            num_classes=len(class_names),
            hidden_dim=gat_hidden_dim,
            num_layers=gat_layers,
            num_heads=gat_heads,
            readout=gat_readout,
            include_wrist=include_wrist,
        )
        model_desc = (
            "gat ("
            f"hidden_dim={gat_hidden_dim}, heads={gat_heads}, layers={gat_layers}, readout={gat_readout})"
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Model: {model_desc}")

    rng = jax.random.PRNGKey(seed)
    params = network.init(rng)

    decay_steps = lr_decay_steps or epochs * steps
    warmup_steps = max(0, lr_warmup_steps)
    decay_span = max(decay_steps - warmup_steps, 1)

    if lr_schedule == "constant":
        lr_sched_fn = None
    elif lr_schedule == "cosine":
        def lr_sched_fn(step):
            """JIT-safe cosine decay with optional linear warmup."""
            step_f = jnp.asarray(step, dtype=jnp.float32)
            warm = jnp.asarray(warmup_steps, dtype=jnp.float32)
            base_lr = jnp.asarray(learning_rate, dtype=jnp.float32)
            final_lr = jnp.asarray(min_learning_rate, dtype=jnp.float32)
            decay_total = jnp.asarray(decay_span, dtype=jnp.float32)

            # Linear warmup from 0 -> base_lr over warmup steps (inclusive of step 0)
            warm_denom = jnp.maximum(warm, 1.0)
            warm_lr = base_lr * (step_f + 1.0) / warm_denom

            # Cosine decay after warmup
            t = jnp.maximum(step_f - warm + 1.0, 0.0)
            progress = jnp.minimum(t / jnp.maximum(decay_total, 1.0), 1.0)
            cosine_lr = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + jnp.cos(jnp.pi * progress))

            return jnp.where(step_f < warm, warm_lr, cosine_lr)
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    # Helper to log LR even when optimizer schedule is None (constant).
    def _current_lr(step: int) -> float:
        if lr_sched_fn is None:
            return float(learning_rate)
        return float(lr_sched_fn(step))

    optimizer = Adam(lr=learning_rate, lr_schedule=lr_sched_fn)
    opt_state = optimizer.init(params)
    state = optim_base.TrainState(params, opt_state)

    data_rng = np.random.default_rng(seed)

    global_step = 0
    best_val_acc = -np.inf
    best_epoch = 0
    swa_params: Params | None = None
    swa_count = 0
    start_epoch = 1

    if resume_from is not None:
        ckpt = _load_checkpoint(resume_from)
        state = optim_base.TrainState(
            params=jax.device_put(ckpt["params"]),
            opt_state=jax.device_put(ckpt["opt_state"]),
        )
        best_val_acc = ckpt.get("best_val_acc", best_val_acc)
        best_epoch = ckpt.get("best_epoch", best_epoch)
        global_step = ckpt.get("global_step", global_step)
        start_epoch = ckpt.get("epoch", 0) + 1
        data_rng_state = ckpt.get("data_rng_state")
        if data_rng_state is not None:
            data_rng.bit_generator.state = data_rng_state
        swa_params = ckpt.get("swa_params")
        if swa_params is not None:
            swa_params = jax.device_put(swa_params)
        swa_count = ckpt.get("swa_count", swa_count)
        ckpt_classes = ckpt.get("class_names")
        if ckpt_classes is not None and list(ckpt_classes) != list(class_names):
            raise ValueError(
                "Checkpoint classes do not match dataset classes; refusing to resume to avoid label mismatch."
            )
        ckpt_mean = ckpt.get("channel_mean")
        ckpt_std = ckpt.get("channel_std")
        if ckpt_mean is not None and not np.allclose(ckpt_mean, channel_mean, atol=1e-5):
            print("WARNING: Checkpoint channel mean differs from current data stats.")
        if ckpt_std is not None and not np.allclose(ckpt_std, channel_std, atol=1e-5):
            print("WARNING: Checkpoint channel std differs from current data stats.")
        print(f"Resumed from checkpoint {resume_from} at epoch {start_epoch - 1} (global_step={global_step}).")

    stat_mask = _running_stat_mask(state.params)
    train_step = make_train_step(network, optimizer, weight_decay=1e-4, stat_mask=stat_mask)
    eval_step = make_eval_step(network)

    summary_writer = _create_summary_writer(log_dir)

    train_iter = batch_iterator(
        train_samples,
        class_weights,
        batch_size,
        rng=data_rng,
        oversample=oversample,
        augment=augment,
        norm_stats=(channel_mean, channel_std),
        representation=representation,
        include_wrist=include_wrist,
    )

    if start_epoch > epochs:
        print(f"Checkpoint epoch {start_epoch - 1} >= requested epochs={epochs}; skipping training.")
        if summary_writer is not None:
            summary_writer.close()
        return

    epoch_iter = (
        tqdm(
            range(start_epoch, epochs + 1),
            desc="Epochs",
            initial=start_epoch - 1,
            total=epochs,
            dynamic_ncols=True,
        )
        if tqdm is not None
        else range(start_epoch, epochs + 1)
    )

    for epoch in epoch_iter:
        epoch_losses = []
        epoch_accs = []
        step_iter = (
            tqdm(range(steps), desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
            if tqdm is not None
            else range(steps)
        )
        for step_idx in step_iter:
            batch_np = next(train_iter)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}
            # Generate fresh RNG key for each batch for dropout
            step_rng = jax.random.fold_in(rng, epoch * steps + step_idx)
            state, loss_value = train_step(state, batch, step_rng)
            epoch_losses.append(float(loss_value))
            # Compute training accuracy on this batch (post-update, eval mode).
            logits = network.apply(state.params, batch["x"], rng=None, is_training=False)
            preds = jnp.argmax(logits, axis=-1)
            batch_acc = float(jnp.mean(preds == batch["y"]))
            epoch_accs.append(batch_acc)
            current_lr = _current_lr(state.opt_state.step)
            if summary_writer is not None:
                summary_writer.add_scalar("loss/train_batch", float(loss_value), global_step)
                summary_writer.add_scalar("accuracy/train_batch", batch_acc, global_step)
                summary_writer.add_scalar("lr", current_lr, global_step)
            if tqdm is not None:
                step_iter.set_postfix(loss=float(loss_value), acc=batch_acc, lr=current_lr)
            global_step += 1

            if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                step_ckpt_path = checkpoint_dir / f"step_{global_step:09d}.pkl"
                _save_checkpoint(
                    step_ckpt_path,
                    state,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_acc=best_val_acc,
                    best_epoch=best_epoch,
                    class_names=class_names,
                    channel_mean=channel_mean,
                    channel_std=channel_std,
                    model_type=model_type,
                    representation=representation,
                    include_wrist=include_wrist,
                    data_rng_state=data_rng.bit_generator.state if data_rng is not None else None,
                    swa_params=swa_params if use_swa else None,
                    swa_count=swa_count if use_swa else 0,
                )
        if tqdm is not None:
            step_iter.close()

        # Validation on the entire validation set (reset each epoch for consistency)
        if valid_samples:
            all_preds = []
            all_labels = []
            loss_sum = 0.0
            total_val_samples = 0

            # Process all validation samples in batches
            for start_idx in range(0, len(valid_samples), batch_size):
                batch_samples = valid_samples[start_idx:start_idx + batch_size]

                # Load batch data
                xs = []
                ys = []
                ws = []
                for path, label in batch_samples:
                    arr = np.load(path)
                    xs.append(
                        _prepare_landmark_sample(
                            arr,
                            representation=representation,
                            augment=False,
                            rng=None,
                            norm_stats=(channel_mean, channel_std),
                            include_wrist=include_wrist,
                        )
                    )
                    ys.append(label)
                    ws.append(class_weights[label])

                batch_x = jnp.array(np.stack(xs, axis=0))
                batch_y = jnp.array(np.array(ys, dtype=np.int32))
                batch_w = jnp.array(np.array(ws, dtype=np.float32))

                # Evaluate
                val_batch = {"x": batch_x, "y": batch_y, "w": batch_w}
                metrics = eval_step(state.params, val_batch)

                # Collect predictions for analysis
                logits = network.apply(state.params, batch_x, rng=None, is_training=False)
                preds = jnp.argmax(logits, axis=-1)
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.tolist())
                batch_size_eff = len(batch_samples)
                loss_sum += float(metrics['loss']) * batch_size_eff
                total_val_samples += batch_size_eff

            # Calculate overall metrics
            avg_val_loss = loss_sum / max(1, total_val_samples)
            avg_val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

            # Debug: show prediction distribution every 10 epochs
            if epoch % 10 == 0:
                pred_dist = Counter(all_preds)
                label_dist = Counter(all_labels)
                print(f"\n  [DEBUG] Epoch {epoch} - Predicted class distribution: {dict(pred_dist)}")
                print(f"  [DEBUG] Epoch {epoch} - Actual class distribution: {dict(label_dist)}")
                print(f"  [DEBUG] Total validation samples evaluated: {len(all_preds)}")
        else:
            avg_val_loss = 0.0
            avg_val_acc = 0.0

        train_epoch_loss = np.mean(epoch_losses)
        train_epoch_acc = np.mean(epoch_accs) if epoch_accs else 0.0
        epoch_lr = _current_lr(state.opt_state.step)

        improved = False
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch
            improved = True

        # Stochastic Weight Averaging: start averaging from max(best_epoch, swa_start_epoch)
        if use_swa and epoch >= max(best_epoch, swa_start_epoch):
            if swa_params is None:
                swa_params = state.params
                swa_count = 1
            else:
                swa_params = jax.tree_util.tree_map(
                    lambda a, b: (a * swa_count + b) / (swa_count + 1),
                    swa_params,
                    state.params,
                )
                swa_count += 1

        print(
            f"Epoch {epoch:02d} | train_loss={train_epoch_loss:.4f} train_acc={train_epoch_acc:.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f} lr={epoch_lr:.6g}"
        )

        if summary_writer is not None:
            summary_writer.add_scalar("loss/train_epoch", float(train_epoch_loss), epoch)
            summary_writer.add_scalar("accuracy/train_epoch", float(train_epoch_acc), epoch)
            summary_writer.add_scalar("loss/val", float(avg_val_loss), epoch)
            summary_writer.add_scalar("accuracy/val", float(avg_val_acc), epoch)
            summary_writer.add_scalar("lr_epoch", float(epoch_lr), epoch)
            summary_writer.flush()

        _save_checkpoint(
            last_ckpt_path,
            state,
            epoch=epoch,
            global_step=global_step,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            class_names=class_names,
            channel_mean=channel_mean,
            channel_std=channel_std,
            model_type=model_type,
            representation=representation,
            include_wrist=include_wrist,
            data_rng_state=data_rng.bit_generator.state if data_rng is not None else None,
            swa_params=swa_params if use_swa else None,
            swa_count=swa_count if use_swa else 0,
        )

        if improved:
            _save_checkpoint(
                best_ckpt_path,
                state,
                epoch=epoch,
                global_step=global_step,
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                class_names=class_names,
                channel_mean=channel_mean,
                channel_std=channel_std,
                model_type=model_type,
                representation=representation,
                include_wrist=include_wrist,
                data_rng_state=data_rng.bit_generator.state if data_rng is not None else None,
                swa_params=swa_params if use_swa else None,
                swa_count=swa_count if use_swa else 0,
            )

    # Optionally swap to SWA-averaged weights for final model use.
    if use_swa and swa_params is not None:
        state = optim_base.TrainState(swa_params, state.opt_state)
        print(f"Using SWA parameters averaged over {swa_count} epochs starting at epoch {max(best_epoch, swa_start_epoch)}.")
    else:
        print("SWA disabled or not triggered; final params come from the last epoch.")

    # Save a final checkpoint reflecting any SWA swap.
    _save_checkpoint(
        last_ckpt_path,
        state,
        epoch=epochs,
        global_step=global_step,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        class_names=class_names,
        channel_mean=channel_mean,
        channel_std=channel_std,
        model_type=model_type,
        representation=representation,
        include_wrist=include_wrist,
        data_rng_state=data_rng.bit_generator.state if data_rng is not None else None,
        swa_params=swa_params if use_swa else None,
        swa_count=swa_count if use_swa else 0,
    )

    print("Training complete. You can now use `network.apply` with the learned params.")
    print(f"Checkpoints written to {checkpoint_dir} (last.pkl, best.pkl).")
    if tqdm is not None and hasattr(epoch_iter, "close"):
        epoch_iter.close()
    if summary_writer is not None:
        summary_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small hand-pose CNN using jaxnn.")
    parser.add_argument("--data-root", type=Path, default=Path("data/guitar-chords_landmarks"), help="Root directory with train/valid/test chord folders.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help=(
            "Optional override for the dataset root. Use this to point at the secondary-only "
            "export (e.g. data/guitar_chords_landmarks_secondary). If omitted, falls back to --data-root."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine",
                        help="Learning rate schedule (default constant).")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5,
                        help="Final learning rate for cosine decay.")
    parser.add_argument("--lr-decay-steps", type=int,
                        help="Override number of steps used for LR decay (defaults to epochs * steps_per_epoch).")
    parser.add_argument("--lr-warmup-steps", type=int, default=0,
                        help="Linear warmup steps before applying the decay schedule.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "resnet", "gat"],
        default="baseline",
        help="Select the baseline CNN, the residual CNN, or the new graph attention network ('gat').",
    )
    parser.add_argument("--resnet-width-mult", type=float, default=1.0,
                        help="Width multiplier for the ResNet channels (only used when --model-type=resnet).")
    parser.add_argument(
        "--gat-hidden-dim",
        type=int,
        default=96,
        help="Hidden feature size per node for the graph attention model (only used with --model-type=gat).",
    )
    parser.add_argument(
        "--gat-heads",
        type=int,
        default=4,
        help="Number of attention heads per GAT layer (only used with --model-type=gat).",
    )
    parser.add_argument(
        "--gat-layers",
        type=int,
        default=3,
        help="Number of stacked GraphAttentionTransformer layers (only used with --model-type=gat).",
    )
    parser.add_argument(
        "--gat-readout",
        choices=["mean", "max"],
        default="mean",
        help="Graph readout aggregation to convert node embeddings into logits (only for --model-type=gat).",
    )
    parser.add_argument("--use-swa", action="store_true",
                        help="Enable Stochastic Weight Averaging starting after swa-start-epoch and the best val epoch.")
    parser.add_argument("--swa-start-epoch", type=int, default=10,
                        help="Earliest epoch to begin SWA averaging (also waits for best-val epoch).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, help="Optional number of batches per epoch (defaults to dataset size / batch).")
    parser.add_argument("--class-weighting", choices=["effective", "inverse", "sqrt", "none"], default="effective",
                        help="Class weighting strategy to counter imbalance.")
    parser.add_argument("--class-weight-beta", type=float, default=0.99,
                        help="Smoothing factor for 'effective' class weighting (closer to 1.0 => smoother).")
    parser.add_argument("--no-oversample", action="store_true", help="Disable per-epoch class-balanced oversampling.")
    parser.add_argument("--restrict-to-valid-classes", action="store_true",
                        help="Drop train-only classes so metrics reflect the validation label space.")
    parser.add_argument("--no-augment", action="store_true", help="Disable on-the-fly geometric noise during training.")
    parser.add_argument(
        "--fast-training",
        action="store_true",
        help="Use XLA-backed convolutions (lax.conv_general_dilated) for faster GPU/TPU execution.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help=(
            "Directory for TensorBoard event files. Defaults to TENSORBOARD_LOGDIR or runs/<timestamp> "
            "when not provided."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory to store checkpoints (defaults to <log-dir>/checkpoints).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Path to a checkpoint .pkl file to resume training from.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="If >0, also save a checkpoint every N training steps (in addition to per-epoch/best).",
    )

    args = parser.parse_args()
    resolved_log_dir = _resolve_log_dir(args.log_dir)

    data_root = args.dataset_path or args.data_root

    run_training(
        data_root=data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        resnet_width_mult=args.resnet_width_mult,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        gat_readout=args.gat_readout,
        lr_schedule=args.lr_schedule,
        min_learning_rate=args.min_learning_rate,
        lr_decay_steps=args.lr_decay_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        use_swa=args.use_swa,
        swa_start_epoch=args.swa_start_epoch,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        oversample=not args.no_oversample,
        class_weighting=args.class_weighting,
        class_weight_beta=args.class_weight_beta,
        restrict_to_valid_classes=args.restrict_to_valid_classes,
        augment=not args.no_augment,
        fast_training=args.fast_training,
        log_dir=resolved_log_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
