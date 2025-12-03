#!/usr/bin/env bash
# Quick launcher for training all architectures with distinct checkpoint dirs.
# Usage: ./training_commands.sh [data_root]
#   data_root defaults to the secondary export: data/guitar_chords_landmarks_secondary
# Environment overrides:
#   EPOCHS (default 500), BATCH (default 64), LR (default 1e-3),
#   WANDB_MODE (online|offline|disabled, default online to view logs live),
#   PATIENCE (default 8), CLASS_WEIGHTINGS (space-separated, default "effective inverse"),
#   OVERSAMPLE_OPTIONS (default "on" to keep it quick),
#   RESNET_WIDTHS, RESNET_LRS.

set -euo pipefail

DATA_ROOT=${1:-data/guitar_chords_landmarks_secondary}
EPOCHS=${EPOCHS:-500}
BATCH=${BATCH:-64}
LR=${LR:-1e-3}
WANDB_MODE=${WANDB_MODE:-online}
PATIENCE=${PATIENCE:-8}

# ResNet sweep settings (can override via env if desired).
RESNET_WIDTHS=(${RESNET_WIDTHS:-0.75 1.0 1.5 2.0})
RESNET_LRS=(${RESNET_LRS:-1e-3 5e-4})
CLASS_WEIGHTINGS=(${CLASS_WEIGHTINGS:-effective inverse})
OVERSAMPLE_OPTIONS=(${OVERSAMPLE_OPTIONS:-on})

# Single timestamp so runs are grouped, while checkpoints stay per-architecture.
STAMP=$(date +"%Y%m%d-%H%M%S")

run_cmd() {
  local arch="$1"; shift
  local extra=("$@")
  local LOG_DIR="runs/${STAMP}/${arch}"
  local CKPT_DIR="checkpoints/${STAMP}/${arch}"
  mkdir -p "$LOG_DIR" "$CKPT_DIR"
  echo "------------------------------------------------------------"
  echo "Training ${arch} | log_dir=${LOG_DIR} | ckpts=${CKPT_DIR}"
  echo "------------------------------------------------------------"
  python train_hand_pose.py \
    --data-root "$DATA_ROOT" \
    --batch-size "$BATCH" \
    --epochs "$EPOCHS" \
    --model-type "$arch" \
    --log-dir "$LOG_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --wandb-mode "$WANDB_MODE" \
    --early-stop-patience "$PATIENCE" \
    "${extra[@]}"
}

# Baseline CNN across class-weighting and oversample variants
for cw in "${CLASS_WEIGHTINGS[@]}"; do
  for osmp in "${OVERSAMPLE_OPTIONS[@]}"; do
    extra_osmp=()
    tag_osmp="oversample"
    if [[ "$osmp" == "off" ]]; then
      extra_osmp=(--no-oversample)
      tag_osmp="no_oversample"
    fi
    run_cmd baseline \
      --learning-rate "$LR" \
      --class-weighting "$cw" \
      "${extra_osmp[@]}" \
      --wandb-tags "baseline" "cw-${cw}" "${tag_osmp}"
  done
done

# ResNet sweep over width multiplier and LR grid
for width in "${RESNET_WIDTHS[@]}"; do
  for lr in "${RESNET_LRS[@]}"; do
    for cw in "${CLASS_WEIGHTINGS[@]}"; do
      for osmp in "${OVERSAMPLE_OPTIONS[@]}"; do
        extra_osmp=()
        tag_osmp="oversample"
        if [[ "$osmp" == "off" ]]; then
          extra_osmp=(--no-oversample)
          tag_osmp="no_oversample"
        fi
        run_cmd resnet \
          --resnet-width-mult "$width" \
          --learning-rate "$lr" \
          --class-weighting "$cw" \
          "${extra_osmp[@]}" \
          --wandb-tags "resnet" "w${width}" "lr${lr}" "cw-${cw}" "${tag_osmp}"
      done
    done
  done
done

# Graph Attention Network with class-weighting / oversample variants
for cw in "${CLASS_WEIGHTINGS[@]}"; do
  for osmp in "${OVERSAMPLE_OPTIONS[@]}"; do
    extra_osmp=()
    tag_osmp="oversample"
    if [[ "$osmp" == "off" ]]; then
      extra_osmp=(--no-oversample)
      tag_osmp="no_oversample"
    fi
    run_cmd gat \
      --learning-rate "$LR" \
      --class-weighting "$cw" \
      "${extra_osmp[@]}" \
      --wandb-tags "gat" "cw-${cw}" "${tag_osmp}"
  done
done

echo "All trainings queued with timestamp ${STAMP}."
