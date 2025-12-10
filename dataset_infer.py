"""Batch inference over the *secondary* guitar-chord validation set.

This script:
  - Loads a trained baseline / resnet / GAT checkpoint from train_hand_pose.py
  - Runs inference on the post-processed secondary dataset split (default: valid)
  - Copies each .npy sample into `correctly_classified/<chord>/` or
    `incorrectly_classified/<chord>/` under an output directory.
  - Saves annotated JPEGs (via OpenCV) with true/pred labels drawn on the original
    secondary images when available; otherwise uses a blank canvas.

Usage example:
  python dataset_infer.py \\
      --checkpoint checkpoints/20251208-183500/gat/best.pkl \\
      --model-type gat \\
      --split valid

The script only touches the secondary processed dataset by default:
  data/guitar_chords_landmarks_secondary/<split>/<chord>/*.npy
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from train_hand_pose import (  # type: ignore
    _load_checkpoint,
    _prepare_landmark_sample,
    build_model,
    build_resnet_model,
)
from jaxnn.models.hand_graph_attention import HandGraphAttentionNetwork  # type: ignore


# --------------------------- Model reconstruction --------------------------- #


def _default_model_kwargs(model_type: str) -> dict[str, Any]:
    if model_type == "baseline":
        return {"use_lax_conv": False}
    if model_type == "resnet":
        return {"use_lax_conv": False, "width_mult": 1.0}
    if model_type == "gat":
        return {
            "hidden_dim": 96,
            "num_heads": 4,
            "num_layers": 3,
            "readout": "mean",
            "include_wrist": True,
            "activation": "gelu",
        }
    raise ValueError(f"Unknown model_type: {model_type}")


def _merge_model_kwargs(model_type: str, ckpt_kwargs: dict[str, Any] | None, overrides: dict[str, Any]) -> dict[str, Any]:
    merged = _default_model_kwargs(model_type)
    if ckpt_kwargs:
        merged.update({k: v for k, v in ckpt_kwargs.items() if v is not None})
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v
    return merged


def _build_network(model_type: str, num_classes: int, model_kwargs: dict[str, Any]):
    if model_type == "baseline":
        return build_model(num_classes=num_classes, use_lax_conv=model_kwargs.get("use_lax_conv", False))
    if model_type == "resnet":
        return build_resnet_model(
            num_classes=num_classes,
            use_lax_conv=model_kwargs.get("use_lax_conv", False),
            width_mult=model_kwargs.get("width_mult", 1.0),
        )
    if model_type == "gat":
        return HandGraphAttentionNetwork(
            num_classes=num_classes,
            hidden_dim=model_kwargs.get("hidden_dim", 96),
            num_heads=model_kwargs.get("num_heads", 4),
            num_layers=model_kwargs.get("num_layers", 3),
            readout=model_kwargs.get("readout", "mean"),
            include_wrist=model_kwargs.get("include_wrist", True),
            activation=model_kwargs.get("activation", "gelu"),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


# ------------------------------- Utilities --------------------------------- #


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _collect_split(data_root: Path, split: str, class_names: Sequence[str]) -> list[tuple[Path, int]]:
    """Return (path, label_idx) for every .npy file in the requested split."""
    split_dir = (data_root / split).expanduser().resolve()
    samples: list[tuple[Path, int]] = []
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for chord_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        chord = chord_dir.name
        if chord not in label_map:
            print(f"[WARN] Skipping unknown chord '{chord}' not present in checkpoint classes.")
            continue
        for npy_path in sorted(chord_dir.glob("*.npy")):
            samples.append((npy_path, label_map[chord]))
    return samples


def _build_raw_image_index(raw_root: Path) -> dict[tuple[str, str], Path]:
    """
    Build a lookup: (chord, filename) -> raw image path for the secondary dataset.
    The raw secondary data keeps all images under raw_root/train/<chord>/filename.jpg
    (or raw_root/<chord>/filename.jpg when no train/ split exists).
    """
    raw_root = raw_root.expanduser().resolve()
    if not raw_root.exists():
        return {}

    index: dict[tuple[str, str], Path] = {}
    for img_path in raw_root.rglob("*.jpg"):
        chord = img_path.parent.name
        index[(chord, img_path.name)] = img_path
    return index


def _guess_raw_image_path(npy_path: Path, chord: str, raw_index: dict[tuple[str, str], Path]) -> Path | None:
    """
    Attempt to map a processed secondary npy file back to its original image.
    The processed files are named 'secondary_<orig>.npy'; we strip the prefix
    and swap the extension to '.jpg'.
    """
    orig_name = npy_path.name.replace("secondary_", "", 1).replace(".npy", ".jpg")
    return raw_index.get((chord, orig_name))


def _render_overlay(img: np.ndarray, text_lines: list[str], is_correct: bool) -> np.ndarray:
    """Overlay status + text on an image using cv2."""
    if img is None or img.size == 0:
        # Fallback canvas when the original image can't be loaded.
        img = np.zeros((512, 512, 3), dtype=np.uint8)

    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 24
    margin = 12

    status_color = (0, 200, 0) if is_correct else (0, 0, 255)

    # Draw status box
    status_text = "CORRECT" if is_correct else "INCORRECT"
    cv2.rectangle(overlay, (margin // 2, margin // 2), (180, margin + line_height), status_color, -1)
    cv2.putText(
        overlay,
        status_text,
        (margin, margin + int(line_height * 0.8)),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Draw text lines
    y = margin + line_height * 2
    for line in text_lines:
        cv2.putText(
            overlay,
            line,
            (margin, y),
            font,
            font_scale,
            status_color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height

    return overlay


def _default_output_dir(split: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("evaluation_outputs") / f"secondary_{split}_{timestamp}"


# ---------------------------------- CLI ------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset-wide inference on the secondary guitar-chord set.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pkl saved by train_hand_pose.py")
    parser.add_argument("--data-root", type=Path, default=Path("data/guitar_chords_landmarks_secondary"),
                        help="Root of the processed secondary dataset (expects split/chord/*.npy).")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="valid",
                        help="Which split of the secondary dataset to evaluate.")
    parser.add_argument("--model-type", choices=["baseline", "resnet", "gat"],
                        help="Override model type (defaults to checkpoint).")
    parser.add_argument("--resnet-width-mult", type=float, help="Width multiplier for resnet (override checkpoint).")
    parser.add_argument("--gat-hidden-dim", type=int, help="Hidden dimension for GAT.")
    parser.add_argument("--gat-heads", type=int, help="Attention heads for GAT.")
    parser.add_argument("--gat-layers", type=int, help="Number of GAT layers.")
    parser.add_argument("--gat-readout", choices=["mean", "max"], help="Readout type for GAT.")
    parser.add_argument("--include-wrist", action=argparse.BooleanOptionalAction,
                        help="Force including wrist node for graph representation.")
    parser.add_argument("--representation", choices=["grid", "graph"],
                        help="Override representation (grid vs graph).")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--secondary-raw-root", type=Path, default=Path("data/secondary_data"),
                        help="Root of the raw secondary images (used for saving annotated jpgs).")
    parser.add_argument("--save-npy-copies", action=argparse.BooleanOptionalAction, default=True,
                        help="Also copy the npy files into the output dirs alongside annotated images.")
    parser.add_argument("--output-dir", type=Path, help="Directory to write classified copies (defaults under evaluation_outputs/).")
    return parser.parse_args()


# --------------------------------- Driver ---------------------------------- #


def main() -> None:
    args = parse_args()

    ckpt = _load_checkpoint(args.checkpoint)
    ckpt_model_type = ckpt.get("model_type")
    model_type = args.model_type or ckpt_model_type
    if model_type is None:
        raise ValueError("Model type not found in checkpoint; please provide --model-type.")

    ckpt_kwargs = ckpt.get("model_kwargs") or {}
    override_kwargs = {
        "width_mult": args.resnet_width_mult,
        "hidden_dim": args.gat_hidden_dim,
        "num_heads": args.gat_heads,
        "num_layers": args.gat_layers,
        "readout": args.gat_readout,
        "include_wrist": args.include_wrist,
    }
    model_kwargs = _merge_model_kwargs(model_type, ckpt_kwargs, override_kwargs)

    representation = args.representation or ckpt.get("representation") or ("graph" if model_type == "gat" else "grid")
    include_wrist = model_kwargs.get("include_wrist", ckpt.get("include_wrist", False))

    class_names = ckpt.get("class_names")
    if not class_names:
        raise ValueError("Checkpoint missing class_names; cannot map predictions.")
    class_names = list(class_names)

    channel_mean = np.asarray(ckpt.get("channel_mean"), dtype=np.float32)
    channel_std = np.asarray(ckpt.get("channel_std"), dtype=np.float32)
    norm_stats = (channel_mean, channel_std)

    # Prepare data
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    samples = _collect_split(data_root, args.split, class_names)
    if not samples:
        raise RuntimeError(f"No samples found in {data_root}/{args.split}")

    raw_index = _build_raw_image_index(args.secondary_raw_root)

    # Build network and load params
    network = _build_network(model_type, num_classes=len(class_names), model_kwargs=model_kwargs)
    params = jax.device_put(ckpt["params"])

    # Prepare output structure
    out_root = args.output_dir.expanduser().resolve() if args.output_dir else _default_output_dir(args.split)
    correct_root = out_root / "correctly_classified"
    incorrect_root = out_root / "incorrectly_classified"
    for chord in class_names:
        (correct_root / chord).mkdir(parents=True, exist_ok=True)
        (incorrect_root / chord).mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0

    def run_batch(batch_paths: list[Path], batch_labels: list[int]) -> None:
        nonlocal total, correct
        xs = []
        for path in batch_paths:
            arr = np.load(path)
            xs.append(
                _prepare_landmark_sample(
                    arr,
                    representation=representation,
                    augment=False,
                    rng=None,
                    norm_stats=norm_stats,
                    include_wrist=include_wrist,
                )
            )
        batch_x = jnp.array(np.stack(xs, axis=0))
        logits = network.apply(params, batch_x, rng=None, is_training=False)
        scores = _softmax(np.asarray(logits))
        preds = np.argmax(scores, axis=-1)

        for idx, (path, true_label, pred_label) in enumerate(zip(batch_paths, batch_labels, preds.tolist())):
            total += 1
            chord_name = class_names[true_label]
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
                dest_dir = correct_root / chord_name
            else:
                dest_dir = incorrect_root / chord_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            if args.save_npy_copies:
                shutil.copy2(path, dest_dir / path.name)

            # Save annotated image (if we can find the original).
            raw_img_path = _guess_raw_image_path(path, chord_name, raw_index)
            img = cv2.imread(str(raw_img_path)) if raw_img_path is not None else None
            prob = float(scores[idx, pred_label]) if scores.ndim == 2 else 0.0
            text_lines = [
                f"True: {chord_name}",
                f"Pred: {class_names[pred_label]} ({prob:.3f})",
            ]
            annotated = _render_overlay(img, text_lines, is_correct)
            out_img_path = dest_dir / f"{path.stem}.jpg"
            cv2.imwrite(str(out_img_path), annotated)

    batch_size = max(1, args.batch_size)
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        batch_paths, batch_labels = zip(*batch)
        run_batch(list(batch_paths), list(batch_labels))

    accuracy = correct / total if total else 0.0
    print(f"[DONE] Evaluated {total} samples from {data_root}/{args.split}")
    print(f"[RESULT] Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"[OUTPUT] Classified copies written to: {out_root}")


if __name__ == "__main__":
    main()
