"""Minimal hand-pose classifier training script using jaxnn.

The model consumes wrist-normalized landmark .npy files produced by
``hand_pose.py``. Each file contains 20 (x, y, z) points (wrist removed).
We reshape them into a 4x5 grid (joints x fingers) with 3 channels to feed a
small ResNet-style CNN that uses leaky ReLU activations and Adam.

The script also balances uneven chord data by (a) oversampling each class to
the same per-epoch count and (b) applying class-weighted cross-entropy.
"""

from __future__ import annotations

import os
# Force JAX to use CPU only to avoid CUDA plugin conflicts
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import math
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


Array = types.Array
Params = types.Params
PRNGKey = types.PRNGKey


# ----------------------------- Data utilities ----------------------------- #


def _finger_grid(landmarks: np.ndarray) -> np.ndarray:
    """Reshape 20x3 landmarks into (4, 5, 3): 4 joints per finger x 5 fingers.

    The MediaPipe landmark order (after dropping the wrist) is grouped by finger
    in contiguous blocks of four points: thumb (1-4), index (5-8), middle
    (9-12), ring (13-16), pinky (17-20).
    """

    if landmarks.shape != (20, 3):
        raise ValueError(f"Expected landmarks with shape (20, 3), got {landmarks.shape}")

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
    else:
        # Keep the provided ordering but only include known directories
        class_names = [c for c in class_names if (split_dir / c).is_dir()]

    label_map = {name: idx for idx, name in enumerate(class_names)}
    samples: list[tuple[Path, int]] = []

    for cls in class_names:
        for npy_file in sorted((split_dir / cls).glob("*.npy")):
            samples.append((npy_file, label_map[cls]))

    return samples, list(class_names)


def _compute_class_weights(counts: Counter) -> np.ndarray:
    """Inverse frequency weights normalized to mean 1.0."""

    if not counts:
        return np.array([], dtype=np.float32)

    num_classes = len(counts)
    total = sum(counts.values())
    weights = {cls: total / (num_classes * count) for cls, count in counts.items()}
    return np.array([weights[i] for i in range(num_classes)], dtype=np.float32)


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
                xs.append(_finger_grid(arr))
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

    def __post_init__(self) -> None:
        if len(self.strides) != 2:
            raise ValueError("strides must be a 2-tuple")

    def init(self, rng: PRNGKey) -> Params:
        k1, k2, kproj = jax.random.split(rng, 3)

        conv1 = layers.Conv2D(self.in_channels, self.out_channels, (3, 3), strides=self.strides)
        conv2 = layers.Conv2D(self.out_channels, self.out_channels, (3, 3), strides=(1, 1))

        params = {
            "conv1": conv1.init(k1),
            "conv2": conv2.init(k2),
        }

        if self.in_channels != self.out_channels or self.strides != (1, 1):
            proj = layers.Conv2D(self.in_channels, self.out_channels, (1, 1), strides=self.strides)
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

        conv1 = layers.Conv2D(self.in_channels, self.out_channels, (3, 3), strides=self.strides)
        conv2 = layers.Conv2D(self.out_channels, self.out_channels, (3, 3), strides=(1, 1))

        out = conv1.apply(params["conv1"], inputs)
        out = activations.leaky_relu(out)

        out = conv2.apply(params["conv2"], out)

        shortcut = inputs
        if "proj" in params:
            proj = layers.Conv2D(self.in_channels, self.out_channels, (1, 1), strides=self.strides)
            shortcut = proj.apply(params["proj"], inputs)

        return activations.leaky_relu(out + shortcut)


def build_model(num_classes: int) -> layers.Sequential:
    """Construct a compact ResNet-ish classifier for 4x5x3 inputs."""

    return layers.Sequential(
        (
            layers.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3), padding="SAME"),
            layers.Activation("leaky_relu"),

            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),

            ResidualBlock(in_channels=32, out_channels=64, strides=(2, 2)),
            ResidualBlock(in_channels=64, out_channels=64),

            layers.Flatten(),
            layers.Dense(in_features=64 * 2 * 3, out_features=128, activation="leaky_relu"),
            layers.Dense(in_features=128, out_features=num_classes, activation=None),
        )
    )


# ----------------------------- Training loop ------------------------------ #


def weighted_cross_entropy(logits: Array, labels: Array, weights: Array) -> Array:
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits)
    per_example = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(per_example * weights)


def make_train_step(network: layers.Sequential, optimizer: Adam):
    @jax.jit
    def train_step(state: optim_base.TrainState, batch: dict[str, Array]) -> tuple[optim_base.TrainState, Array]:
        def loss_fn(params: Params) -> Array:
            logits = network.apply(params, batch["x"], is_training=True)
            return weighted_cross_entropy(logits, batch["y"], batch["w"])

        loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optim_base.apply_updates(state.params, updates)
        new_state = optim_base.TrainState(new_params, new_opt_state)
        return new_state, loss_value

    return train_step


def make_eval_step(network: layers.Sequential):
    @jax.jit
    def eval_step(params: Params, batch: dict[str, Array]) -> dict[str, Array]:
        logits = network.apply(params, batch["x"], is_training=False)
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
    seed: int = 0,
    steps_per_epoch: int | None = None,
) -> None:
    train_dir = data_root / "train"
    valid_dir = data_root / "valid"

    train_samples, class_names = _collect_split(train_dir)
    valid_samples, _ = _collect_split(valid_dir, class_names=class_names)

    # Track per-class availability and guard against missing labels.
    class_to_files: dict[int, list[Path]] = {idx: [] for idx in range(len(class_names))}
    for path, label in train_samples:
        class_to_files[label].append(path)

    missing = [class_names[idx] for idx, files in class_to_files.items() if not files]
    if missing:
        raise ValueError(f"No training examples found for classes: {missing}")

    train_counts = Counter({idx: len(files) for idx, files in class_to_files.items()})
    class_weights = _compute_class_weights(train_counts)

    print(f"Loaded {len(train_samples)} train and {len(valid_samples)} valid examples across {len(class_names)} classes.")
    print("Training per-class counts:")
    for cls_idx, cls_name in enumerate(class_names):
        print(f"  {cls_name}: {train_counts.get(cls_idx, 0)}")

    # Check validation distribution
    valid_counts = Counter([label for _, label in valid_samples])
    print("\nValidation per-class counts:")
    for cls_idx, cls_name in enumerate(class_names):
        print(f"  {cls_name}: {valid_counts.get(cls_idx, 0)}")

    print(f"\nClass weights: {class_weights}")

    network = build_model(num_classes=len(class_names))
    rng = jax.random.PRNGKey(seed)
    params = network.init(rng)

    optimizer = Adam(lr=learning_rate)
    opt_state = optimizer.init(params)
    state = optim_base.TrainState(params, opt_state)

    train_step = make_train_step(network, optimizer)
    eval_step = make_eval_step(network)

    train_iter = batch_iterator(train_samples, class_weights, batch_size, seed=seed)
    valid_iter = batch_iterator(valid_samples, class_weights, batch_size, shuffle=False, oversample=False, seed=seed + 1)

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        steps = steps_per_epoch or max(1, len(train_samples) // batch_size)
        for _ in range(steps):
            batch_np = next(train_iter)
            batch = {k: jnp.array(v) for k, v in batch_np.items()}
            state, loss_value = train_step(state, batch)
            epoch_losses.append(float(loss_value))

        # Validation on the entire validation set
        if valid_samples:
            val_losses = []
            val_accs = []
            all_preds = []
            all_labels = []
            val_steps = max(1, len(valid_samples) // batch_size)
            for _ in range(val_steps):
                val_batch_np = next(valid_iter)
                val_batch = {k: jnp.array(v) for k, v in val_batch_np.items()}
                metrics = eval_step(state.params, val_batch)
                val_losses.append(float(metrics['loss']))
                val_accs.append(float(metrics['acc']))

                # Collect predictions for analysis
                logits = network.apply(state.params, val_batch["x"], is_training=False)
                preds = jnp.argmax(logits, axis=-1)
                all_preds.extend(preds.tolist())
                all_labels.extend(val_batch["y"].tolist())

            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accs)

            # Debug: show prediction distribution every 10 epochs
            if epoch % 10 == 0:
                pred_dist = Counter(all_preds)
                label_dist = Counter(all_labels)
                print(f"\n  [DEBUG] Epoch {epoch} - Predicted class distribution: {dict(pred_dist)}")
                print(f"  [DEBUG] Epoch {epoch} - Actual class distribution: {dict(label_dist)}")
        else:
            avg_val_loss = 0.0
            avg_val_acc = 0.0

        print(
            f"Epoch {epoch:02d} | train_loss={np.mean(epoch_losses):.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f}"
        )

    print("Training complete. You can now use `network.apply` with the learned params.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small hand-pose CNN using jaxnn.")
    parser.add_argument("--data-root", type=Path, default=Path("data/guitar-chords_landmarks"), help="Root directory with train/valid/test chord folders.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, help="Optional number of batches per epoch (defaults to dataset size / batch).")

    args = parser.parse_args()

    run_training(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
    )


if __name__ == "__main__":
    main()
