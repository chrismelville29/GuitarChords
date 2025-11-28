from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Iterable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

# Allow running as a script without installing the package.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from jaxnn import types
from jaxnn.nn import losses, model
from jaxnn.optim import base
from jaxnn.optim.sgd import SGD
from jaxnn.train import loop

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    epochs: int = 1
    learning_rate: float = 1e-2
    momentum: float = 0.9
    max_train_batches: int | None = 100
    max_eval_batches: int | None = 20
    seed: int = 0
    device: str = "auto"  # cpu, gpu, or auto


def _prepare_mnist_dataset(split: str, batch_size: int) -> Iterable[Mapping[str, np.ndarray]]:
    """Load MNIST from TFDS and return a numpy iterator."""
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover - requires extra deps
        raise ImportError(
            "Install tensorflow and tensorflow-datasets to run the MNIST CNN example."
        ) from exc

    def _preprocess(image: tf.Tensor, label: tf.Tensor) -> Mapping[str, tf.Tensor]:
        image = tf.cast(image, tf.float32) / 255.0
        return {"x": image, "y": tf.cast(label, tf.int32)}

    ds = tfds.load("mnist", split=split, shuffle_files=split == "train", as_supervised=True)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if split == "train":
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)


def _run_train_epoch(
    train_step,
    state: base.TrainState,
    dataset: Iterable[Mapping[str, np.ndarray]],
    max_batches: int | None,
) -> tuple[base.TrainState, float]:
    total_loss = 0.0
    count = 0
    for step, batch in enumerate(dataset):
        if max_batches is not None and step >= max_batches:
            break
        state, loss_value = train_step(state, batch)
        total_loss += float(loss_value)
        count += 1
    mean_loss = total_loss / max(1, count)
    return state, mean_loss


def _run_eval(
    eval_step,
    params,
    dataset: Iterable[Mapping[str, np.ndarray]],
    max_batches: int | None,
) -> Mapping[str, float]:
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for step, batch in enumerate(dataset):
        if max_batches is not None and step >= max_batches:
            break
        metrics = eval_step(params, batch)
        total_loss += float(metrics["loss"])
        total_acc += float(metrics["accuracy"])
        count += 1
    denom = max(1, count)
    return {"loss": total_loss / denom, "accuracy": total_acc / denom}


def preprocess_opencv_digit(image: np.ndarray) -> jnp.ndarray:
    """Convert a BGR/gray image into a normalized MNIST tensor."""
    if cv2 is None:  # pragma: no cover - optional dependency
        raise ImportError("opencv-python is required for OpenCV helpers.")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = 1.0 - (resized.astype(np.float32) / 255.0)
    normalized = normalized[np.newaxis, ..., np.newaxis]
    return jnp.array(normalized)


def predict_digit_from_image(
    image_path: str,
    params: types.Params,
    apply_fn: Callable[[types.Params, jnp.ndarray], jnp.ndarray],
) -> tuple[int, float]:
    """Run a trained model on a single image path using OpenCV preprocessing."""
    if cv2 is None:  # pragma: no cover - optional dependency
        raise ImportError("opencv-python is required for OpenCV helpers.")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")
    batch = preprocess_opencv_digit(image)
    logits = apply_fn(params, batch)
    probs = jax.nn.softmax(logits, axis=-1)
    pred = int(jnp.argmax(probs, axis=-1)[0])
    confidence = float(jnp.max(probs))
    return pred, confidence


def annotate_prediction(frame: np.ndarray, prediction: int, confidence: float) -> np.ndarray:
    """Overlay the predicted digit on a frame for quick visualization."""
    if cv2 is None:  # pragma: no cover - optional dependency
        raise ImportError("opencv-python is required for OpenCV helpers.")
    annotated = frame.copy()
    cv2.putText(
        annotated,
        f"Pred: {prediction} ({confidence:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _select_device(device: str):
    """Return a context manager that places new arrays on the requested device."""
    if device == "auto":
        print(f"JAX devices: {jax.devices()}")
        return nullcontext()
    try:
        devices = jax.devices(device)
    except RuntimeError as exc:  # pragma: no cover - device query failure
        raise RuntimeError(
            f"Could not select device '{device}'. Available devices: {jax.devices()}"
        ) from exc

    if not devices:
        raise RuntimeError(
            f"Requested device '{device}' is not available. "
            "Install a matching jax/jaxlib build (e.g. CUDA wheel) and ensure drivers are loaded."
        )

    print(f"Using {device} device: {devices[0]}")
    return jax.default_device(devices[0])


def train_mnist_cnn(config: TrainConfig) -> tuple[base.TrainState, Callable[[types.Params, jnp.ndarray], jnp.ndarray]]:
    """End-to-end training loop for the MNIST convnet."""
    with _select_device(config.device):
        rng = jax.random.PRNGKey(config.seed)
        init_fn, base_apply_fn = model.build_mnist_cnn()
        params = init_fn(rng)

        optimizer = SGD(learning_rate=config.learning_rate, momentum=config.momentum)
        opt_state = optimizer.init(params)
        state = base.TrainState(params, opt_state)

        train_apply_fn = lambda p, x: base_apply_fn(p, x, is_training=True)
        eval_apply_fn = lambda p, x: base_apply_fn(p, x, is_training=False)

        train_step = loop.make_train_step(train_apply_fn, losses.cross_entropy_logits, optimizer)
        eval_step = loop.make_eval_step(eval_apply_fn, losses.cross_entropy_logits)

        for epoch in range(config.epochs):
            train_ds = _prepare_mnist_dataset("train", config.batch_size)
            state, train_loss = _run_train_epoch(train_step, state, train_ds, config.max_train_batches)

            eval_ds = _prepare_mnist_dataset("test", config.batch_size)
            metrics = _run_eval(eval_step, state.params, eval_ds, config.max_eval_batches)

            print(
                f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
                f"val_loss={metrics['loss']:.4f} val_acc={metrics['accuracy']:.4f}"
            )

        inference_fn = lambda p, x: base_apply_fn(p, x, is_training=False)
        return state, inference_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple ConvNet on MNIST.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Force a backend. Use --device gpu to ensure a CUDA build of jax/jaxlib is used.",
    )
    args = parser.parse_args()

    config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        seed=args.seed,
        device=args.device,
    )

    train_mnist_cnn(config)


if __name__ == "__main__":
    main()
