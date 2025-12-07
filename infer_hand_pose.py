"""Inference helper for guitar-chord hand pose classifiers.

Features:
  - Loads training checkpoints (.pkl) produced by train_hand_pose.py
  - Rebuilds baseline / resnet / GAT models (uses checkpoint metadata, CLI overrides allowed)
  - Accepts inputs as:
      * Image files (runs MediaPipe HandLandmarker to extract 3D world landmarks)
      * .npy files containing 20- or 21-point landmarks
      * .pkl files containing a numpy array, list, or dict with 'landmarks'/'coords'
  - Handles 20 vs 21 landmarks automatically and applies the same normalization as training
  - Optionally saves MediaPipe overlay images for visual verification
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import mediapipe as mp
import cv2

from hand_pose import (
    normalize_to_wrist,
    create_landmarker,
    _detect_world_landmarks,  # type: ignore
    draw_landmarks_on_image,
    DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
    DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE,
)
from train_hand_pose import (
    _load_checkpoint,  # type: ignore
    _prepare_landmark_sample,  # type: ignore
    build_model,
    build_resnet_model,
)
from jaxnn.models.hand_graph_attention import HandGraphAttentionNetwork


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


def _save_model_details(path: Path, details: dict[str, Any]) -> None:
    """Persist model metadata to a pickle file."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(details, f)


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


# ------------------------------- I/O helpers -------------------------------- #


def _load_pickle_input(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for key in ("landmarks", "coords", "points"):
            if key in obj:
                obj = obj[key]
                break
    arr = np.asarray(obj, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _load_landmarks(path: Path) -> Tuple[np.ndarray, dict]:
    """Load landmarks from .npy or .pkl file.

    Returns (landmarks, meta). Meta currently unused but reserved for future use.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32), {}
    if suffix == ".pkl":
        return _load_pickle_input(path), {}
    raise ValueError(f"Unsupported landmark file type: {path}")


def _detect_from_image(
    image_path: Path,
    min_det_conf: float,
    min_presence_conf: float,
    num_hands: int = 2,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Run MediaPipe HandLandmarker and return (world_landmarks, overlay, rgb_image)."""
    mp_image = mp.Image.create_from_file(str(image_path))
    with create_landmarker(
        min_detection_confidence=min_det_conf,
        min_presence_confidence=min_presence_conf,
        num_hands=num_hands,
    ) as landmarker:
        world, detection_result = _detect_world_landmarks(mp_image, landmarker)

    if world is None:
        raise RuntimeError(f"No hands detected in {image_path}")

    overlay = draw_landmarks_on_image(mp_image.numpy_view(), detection_result) if detection_result else None
    return world.astype(np.float32), overlay, mp_image.numpy_view()


def _prepare_input_tensor(
    coords: np.ndarray,
    *,
    representation: str,
    include_wrist: bool,
    norm_stats: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Convert raw landmarks into model-ready tensor."""
    coords = np.asarray(coords, dtype=np.float32)

    if representation == "grid":
        if coords.shape == (21, 3):
            coords = normalize_to_wrist(coords)
        elif coords.shape != (20, 3):
            raise ValueError(f"Grid representation expects 20 or 21 points, got {coords.shape}")
    elif representation == "graph":
        # hand_graph_features (inside _prepare_landmark_sample) handles 20/21
        pass
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return _prepare_landmark_sample(
        coords,
        representation=representation,
        augment=False,
        rng=None,
        norm_stats=norm_stats,
        include_wrist=include_wrist,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


# ----------------------------- Inference driver ----------------------------- #


def run_single(
    params: Any,
    network,
    landmarks: np.ndarray,
    *,
    representation: str,
    include_wrist: bool,
    norm_stats: tuple[np.ndarray, np.ndarray],
    class_names: Iterable[str],
    top_k: int = 3,
    as_prob: bool = True,
) -> dict[str, Any]:
    tensor = _prepare_input_tensor(
        landmarks,
        representation=representation,
        include_wrist=include_wrist,
        norm_stats=norm_stats,
    )
    batch = jnp.expand_dims(jnp.array(tensor), axis=0)
    logits = network.apply(params, batch, rng=None, is_training=False)
    logits_np = np.asarray(logits)[0]
    scores = _softmax(logits_np) if as_prob else logits_np

    top_indices = np.argsort(-scores)[:top_k]
    return {
        "top_indices": top_indices.tolist(),
        "top_labels": [class_names[i] for i in top_indices],
        "top_scores": [float(scores[i]) for i in top_indices],
        "logits": logits_np.tolist(),
    }


# ---------------------------------- CLI ------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference utility for guitar-chord hand pose models.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pkl saved by train_hand_pose.py")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to images, .npy, or .pkl landmark files.")
    parser.add_argument("--input-type", choices=["auto", "image", "npy", "pkl"], default="auto", help="Force input interpretation.")
    parser.add_argument("--model-type", choices=["baseline", "resnet", "gat"], help="Override model type (defaults to checkpoint).")
    parser.add_argument("--resnet-width-mult", type=float, help="Width multiplier for resnet (override checkpoint).")
    parser.add_argument("--gat-hidden-dim", type=int, help="Hidden dimension for GAT.")
    parser.add_argument("--gat-heads", type=int, help="Attention heads for GAT.")
    parser.add_argument("--gat-layers", type=int, help="Number of GAT layers.")
    parser.add_argument("--gat-readout", choices=["mean", "max"], help="Readout type for GAT.")
    parser.add_argument("--include-wrist", action=argparse.BooleanOptionalAction, help="Force including wrist node for graph repr.")
    parser.add_argument("--representation", choices=["grid", "graph"], help="Override representation (grid vs graph).")
    parser.add_argument("--top-k", type=int, default=3, help="How many predictions to print.")
    parser.add_argument("--raw-logits", action="store_true", help="Print raw logits instead of softmax probabilities.")
    parser.add_argument("--save-overlays", type=Path, help="Optional directory to save MediaPipe overlays for image inputs.")
    parser.add_argument("--min-detection-confidence", type=float, default=DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
                        help="MediaPipe hand detection confidence (image inputs).")
    parser.add_argument("--min-presence-confidence", type=float, default=DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE,
                        help="MediaPipe hand presence confidence (image inputs).")
    parser.add_argument(
        "--export-model-details",
        type=Path,
        help="Optional path to save model metadata (.pkl). Defaults to <checkpoint>_model_details.pkl.",
    )
    return parser.parse_args()


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

    channel_mean = np.asarray(ckpt.get("channel_mean"), dtype=np.float32)
    channel_std = np.asarray(ckpt.get("channel_std"), dtype=np.float32)
    norm_stats = (channel_mean, channel_std)

    model_details = ckpt.get("model_details") or {}
    model_details.update(
        {
            "model_type": model_type,
            "model_kwargs": model_kwargs,
            "representation": representation,
            "include_wrist": include_wrist,
            "class_names": list(class_names),
            "channel_mean": channel_mean,
            "channel_std": channel_std,
        }
    )

    # Build network and load params
    network = _build_network(model_type, num_classes=len(class_names), model_kwargs=model_kwargs)
    params = jax.device_put(ckpt["params"])

    export_path = args.export_model_details or args.checkpoint.with_name(f"{args.checkpoint.stem}_model_details.pkl")
    try:
        _save_model_details(export_path, model_details)
        print(f"[INFO] Model details saved to {export_path}")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[WARN] Failed to export model details to {export_path}: {exc}")

    if args.save_overlays:
        args.save_overlays.mkdir(parents=True, exist_ok=True)

    for input_path_str in args.inputs:
        path = Path(input_path_str)
        if not path.exists():
            print(f"[WARN] Skipping missing input: {path}")
            continue

        input_kind = args.input_type
        if input_kind == "auto":
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                input_kind = "image"
            elif path.suffix.lower() == ".npy":
                input_kind = "npy"
            elif path.suffix.lower() == ".pkl":
                input_kind = "pkl"
            else:
                print(f"[WARN] Unknown file type for {path}, skipping.")
                continue

        overlay = None
        if input_kind == "image":
            landmarks, overlay, _ = _detect_from_image(
                path,
                args.min_detection_confidence,
                args.min_presence_confidence,
            )
        else:
            landmarks, _ = _load_landmarks(path)

        result = run_single(
            params,
            network,
            landmarks,
            representation=representation,
            include_wrist=include_wrist,
            norm_stats=norm_stats,
            class_names=class_names,
            top_k=args.top_k,
            as_prob=not args.raw_logits,
        )

        print(f"\nInput: {path}")
        for idx, (label, score) in enumerate(zip(result["top_labels"], result["top_scores"]), start=1):
            print(f"  {idx}. {label}: {score:.4f}")
        if args.raw_logits:
            print(f"  logits: {np.round(result['logits'], 4).tolist()}")

        if overlay is not None and args.save_overlays:
            out_path = args.save_overlays / f"{path.stem}_overlay.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"  saved overlay -> {out_path}")


if __name__ == "__main__":
    main()
