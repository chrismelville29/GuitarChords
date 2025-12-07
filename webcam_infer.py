"""Live webcam inference for guitar-chord classifiers (baseline / ResNet / GAT).

Loads a training checkpoint saved by ``train_hand_pose.py``, rebuilds the correct
model variant (respecting checkpoint metadata and optional CLI overrides), and
runs real-time predictions from a webcam feed using MediaPipe Hands + OpenCV.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import jax
import mediapipe as mp
import numpy as np

from hand_pose import (
    DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
    DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE,
    _detect_world_landmarks,  # type: ignore
    create_landmarker,
    draw_landmarks_on_image,
)
from infer_hand_pose import _build_network, _merge_model_kwargs, run_single
from train_hand_pose import _load_checkpoint  # type: ignore


def _load_model_and_metadata(args: argparse.Namespace):
    """Restore checkpoint and rebuild the requested network."""
    ckpt = _load_checkpoint(args.checkpoint)

    ckpt_model_type = ckpt.get("model_type")
    model_type = args.model_type or ckpt_model_type
    if model_type is None:
        raise ValueError("Checkpoint is missing model_type; supply --model-type.")

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

    class_names: Iterable[str] | None = ckpt.get("class_names")
    if not class_names:
        raise ValueError("Checkpoint missing class_names; cannot decode predictions.")
    class_names = list(class_names)

    channel_mean = np.asarray(ckpt.get("channel_mean"), dtype=np.float32)
    channel_std = np.asarray(ckpt.get("channel_std"), dtype=np.float32)
    norm_stats = (channel_mean, channel_std)

    network = _build_network(model_type, num_classes=len(class_names), model_kwargs=model_kwargs)
    params = jax.device_put(ckpt["params"])

    return {
        "network": network,
        "params": params,
        "class_names": class_names,
        "representation": representation,
        "include_wrist": include_wrist,
        "norm_stats": norm_stats,
        "model_type": model_type,
        "model_kwargs": model_kwargs,
    }


def _put_text(frame: np.ndarray, text: str, y: int, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam demo for guitar-chord hand pose classifiers.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pkl from train_hand_pose.py")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--model-type", choices=["baseline", "resnet", "gat"], help="Override model type from checkpoint")
    parser.add_argument("--representation", choices=["grid", "graph"], help="Override representation (grid vs graph)")
    parser.add_argument("--resnet-width-mult", type=float, help="Width multiplier for ResNet (override checkpoint)")
    parser.add_argument("--gat-hidden-dim", type=int, help="Hidden dimension for GAT (override checkpoint)")
    parser.add_argument("--gat-heads", type=int, help="Attention heads for GAT (override checkpoint)")
    parser.add_argument("--gat-layers", type=int, help="Number of GAT layers (override checkpoint)")
    parser.add_argument("--gat-readout", choices=["mean", "max"], help="Readout type for GAT")
    parser.add_argument("--include-wrist", action=argparse.BooleanOptionalAction, help="Include wrist node for graph inputs")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions (default: 3)")
    parser.add_argument("--raw-logits", action="store_true", help="Display raw logits instead of probabilities")
    parser.add_argument("--mirror", action="store_true", help="Mirror the camera feed horizontally")
    parser.add_argument("--draw-landmarks", action="store_true", help="Overlay MediaPipe hand landmarks")
    parser.add_argument("--exit-key", type=str, default="q", help="Key to quit the window (default: q)")
    parser.add_argument("--min-detection-confidence", type=float, default=DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
                        help="MediaPipe hand detection confidence")
    parser.add_argument("--min-presence-confidence", type=float, default=DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE,
                        help="MediaPipe hand presence confidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_bundle = _load_model_and_metadata(args)
    network = model_bundle["network"]
    params = model_bundle["params"]
    class_names = model_bundle["class_names"]
    representation = model_bundle["representation"]
    include_wrist = model_bundle["include_wrist"]
    norm_stats = model_bundle["norm_stats"]

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    landmarker = create_landmarker(
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        num_hands=1,
    )

    prev_time = time.time()
    window_name = "Guitar Chord Prediction"

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame from camera; exiting.")
                break

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            world, detection_result = _detect_world_landmarks(mp_image, landmarker)

            display_frame = frame_bgr
            prediction_text = "No hand detected"
            top_scores: list[float] | None = None

            if world is not None:
                result = run_single(
                    params,
                    network,
                    world,
                    representation=representation,
                    include_wrist=include_wrist,
                    norm_stats=norm_stats,
                    class_names=class_names,
                    top_k=args.top_k,
                    as_prob=not args.raw_logits,
                )
                prediction_text = " / ".join(
                    f"{lbl}: {score:.2f}" for lbl, score in zip(result["top_labels"], result["top_scores"])
                )
                top_scores = result["top_scores"]

            # Optionally redraw landmarks on a copy
            if args.draw_landmarks and detection_result is not None:
                annotated = draw_landmarks_on_image(frame_rgb, detection_result)
                display_frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

            # FPS calculation
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            _put_text(display_frame, prediction_text, 32, color=(0, 255, 0) if top_scores else (0, 0, 255))
            _put_text(display_frame, f"FPS: {fps:.1f}", 64, color=(200, 200, 200))

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(args.exit_key):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
