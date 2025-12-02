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
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxnn import types
from jaxnn.nn import activations
from jaxnn.nn import layers
from jaxnn.nn.layers import base as layer_base
from jaxnn.optim import base as optim_base
from jaxnn.optim.adam import Adam
from jaxnn.optim.schedule import cosine_decay_schedule

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


# ----------------------------- Data utilities ----------------------------- #


def _finger_grid(landmarks: np.ndarray, augment: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
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

    # Apply data augmentation if requested
    if augment and rng is not None:
        # Small random rotation around z-axis (mimics slight hand rotation)
        angle = rng.uniform(-0.15, 0.15)  # ±~9 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        landmarks = landmarks @ rot_matrix.T

        # Small random scaling (mimics distance variation)
        scale = rng.uniform(0.95, 1.05)
        landmarks = landmarks * scale

        # Small random translation
        translation = rng.uniform(-0.02, 0.02, size=3)
        landmarks = landmarks + translation

        # Add small Gaussian noise
        noise = rng.normal(0, 0.005, size=landmarks.shape)
        landmarks = landmarks + noise

    joints_per_finger = []
    for finger_idx in range(5):
        start = finger_idx * 4
        joints_per_finger.append(landmarks[start : start + 4])

    # Stack fingers along the second axis: (4 joints, 5 fingers, 3 channels)
    grid = np.stack(joints_per_finger, axis=1)
    return grid.astype(np.float32)


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
) -> Iterator[dict[str, np.ndarray]]:
    """Yield batches with inputs, labels, and per-example weights."""

    rng = np.random.default_rng(seed)

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
                xs.append(_finger_grid(arr, augment=augment, rng=rng))
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
        conv2 = layers.Conv2D(self.out_channels, self.out_channels, (3, 3), strides=(1, 1), use_lax=self.use_lax_conv)

        params = {
            "conv1": conv1.init(k1),
            "conv2": conv2.init(k2),
        }

        if self.in_channels != self.out_channels or self.strides != (1, 1):
            proj = layers.Conv2D(self.in_channels, self.out_channels, (1, 1), strides=self.strides, use_lax=self.use_lax_conv)
            params["proj"] = proj.init(kproj)

        return params

    def apply(
        self,
        params: Params,
        inputs: Array,
        *,
        rng: PRNGKey | None = None,
        is_training: bool = True,
    ) -> Array:
        _ = (rng, is_training)  # no stochastic layers inside

        conv1 = layers.Conv2D(
            self.in_channels,
            self.out_channels,
            (3, 3),
            strides=self.strides,
            use_lax=self.use_lax_conv,
            padding="SAME",
        )
        conv2 = layers.Conv2D(
            self.out_channels,
            self.out_channels,
            (3, 3),
            strides=(1, 1),
            use_lax=self.use_lax_conv,
            padding="SAME",
        )

        out = conv1.apply(params["conv1"], inputs)
        out = activations.leaky_relu(out)

        out = conv2.apply(params["conv2"], out)

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
            shortcut = proj.apply(params["proj"], inputs)

        return activations.leaky_relu(out + shortcut)


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
    dropout_rate: float = 0.3,
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
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    per_example = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(per_example * weights)


def l2_regularization(params: Params, weight_decay: float = 1e-4) -> Array:
    """Compute L2 regularization penalty on parameters."""
    leaves = jax.tree_util.tree_leaves(params)
    return weight_decay * sum(jnp.sum(jnp.square(p)) for p in leaves)


def make_train_step(network: layers.Sequential, optimizer: Adam, weight_decay: float = 1e-4):
    @jax.jit
    def train_step(state: optim_base.TrainState, batch: dict[str, Array], rng: PRNGKey) -> tuple[optim_base.TrainState, Array]:
        def loss_fn(params: Params) -> Array:
            logits = network.apply(params, batch["x"], rng=rng, is_training=True)
            ce_loss = weighted_cross_entropy(logits, batch["y"], batch["w"])
            l2_loss = l2_regularization(params, weight_decay)
            return ce_loss + l2_loss

        loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optim_base.apply_updates(state.params, updates)
        new_state = optim_base.TrainState(new_params, new_opt_state)
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
    lr_schedule: str = "constant",
    min_learning_rate: float = 1e-5,
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
) -> None:
    if swa_start_epoch < 0:
        raise ValueError("swa_start_epoch must be non-negative")
    if min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative")

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
    elif model_type == "resnet":
        network = build_resnet_model(
            num_classes=len(class_names),
            use_lax_conv=fast_training,
            width_mult=resnet_width_mult,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"Model: {model_type} (width_mult={resnet_width_mult if model_type == 'resnet' else 'n/a'}, "
          f"use_lax_conv={fast_training})")

    rng = jax.random.PRNGKey(seed)
    params = network.init(rng)

    if lr_schedule == "constant":
        lr_sched_fn = None
    elif lr_schedule == "cosine":
        total_steps = epochs * steps
        lr_sched_fn = cosine_decay_schedule(initial_lr=learning_rate, final_lr=min_learning_rate, total_steps=total_steps)
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

    train_step = make_train_step(network, optimizer, weight_decay=1e-4)  # Moderate L2 regularization
    eval_step = make_eval_step(network)

    summary_writer = _create_summary_writer(log_dir)
    global_step = 0
    best_val_acc = -np.inf
    best_epoch = 0
    swa_params: Params | None = None
    swa_count = 0

    train_iter = batch_iterator(
        train_samples,
        class_weights,
        batch_size,
        seed=seed,
        oversample=oversample,
        augment=augment,
    )

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for step_idx in range(steps):
            batch_np = next(train_iter)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}
            # Generate fresh RNG key for each batch for dropout
            step_rng = jax.random.fold_in(rng, epoch * steps + step_idx)
            state, loss_value = train_step(state, batch, step_rng)
            epoch_losses.append(float(loss_value))
            if summary_writer is not None:
                summary_writer.add_scalar("loss/train_batch", float(loss_value), global_step)
                summary_writer.add_scalar("lr", _current_lr(state.opt_state.step), global_step)
            global_step += 1

        # Validation on the entire validation set (reset each epoch for consistency)
        if valid_samples:
            all_preds = []
            all_labels = []
            all_losses = []

            # Process all validation samples in batches
            for start_idx in range(0, len(valid_samples), batch_size):
                batch_samples = valid_samples[start_idx:start_idx + batch_size]

                # Load batch data
                xs = []
                ys = []
                ws = []
                for path, label in batch_samples:
                    arr = np.load(path)
                    xs.append(_finger_grid(arr))
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
                all_losses.append(float(metrics['loss']))

            # Calculate overall metrics
            avg_val_loss = np.mean(all_losses)
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

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch

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
            f"Epoch {epoch:02d} | train_loss={train_epoch_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f}"
        )

        if summary_writer is not None:
            summary_writer.add_scalar("loss/train_epoch", float(train_epoch_loss), epoch)
            summary_writer.add_scalar("loss/val", float(avg_val_loss), epoch)
            summary_writer.add_scalar("accuracy/val", float(avg_val_acc), epoch)
            summary_writer.flush()

    # Optionally swap to SWA-averaged weights for final model use.
    if use_swa and swa_params is not None:
        state = optim_base.TrainState(swa_params, state.opt_state)
        print(f"Using SWA parameters averaged over {swa_count} epochs starting at epoch {max(best_epoch, swa_start_epoch)}.")
    else:
        print("SWA disabled or not triggered; final params come from the last epoch.")

    print("Training complete. You can now use `network.apply` with the learned params.")
    if summary_writer is not None:
        summary_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small hand-pose CNN using jaxnn.")
    parser.add_argument("--data-root", type=Path, default=Path("data/guitar-chords_landmarks"), help="Root directory with train/valid/test chord folders.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine",
                        help="Learning rate schedule (default constant).")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5,
                        help="Final learning rate for cosine decay.")
    parser.add_argument("--model-type", choices=["baseline", "resnet"], default="baseline",
                        help="Choose between the original CNN ('baseline') and a deeper residual model ('resnet').")
    parser.add_argument("--resnet-width-mult", type=float, default=1.0,
                        help="Width multiplier for the ResNet channels (only used when --model-type=resnet).")
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

    args = parser.parse_args()
    resolved_log_dir = _resolve_log_dir(args.log_dir)

    run_training(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        resnet_width_mult=args.resnet_width_mult,
        lr_schedule=args.lr_schedule,
        min_learning_rate=args.min_learning_rate,
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
    )


if __name__ == "__main__":
    main()
