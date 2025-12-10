# GuitarChords

Hand pose to guitar chord pipeline built with MediaPipe for landmark extraction and lightweight JAX models for classification. This repo includes:
- `hand_pose.py` for exporting 3D hand landmarks from images (downloads the public `dduka/guitar-chords` dataset on first run).
- `train_hand_pose.py` for training baseline, ResNet, or GAT classifiers on the exported landmarks.
- `infer_hand_pose.py` and `webcam_infer.py` for offline image or webcam inference.
- `jaxnn/` - a small neural network library (installed in editable mode during setup).

## Installation (conda)
Prereqs: Anaconda/Miniconda, Git, and a C++ build toolchain. Run everything from the repo root.

1. Create and activate the env (Python 3.12)
   ```bash
   conda create -n guitar-chords python=3.12 -y
   conda activate guitar-chords
   python -m pip install --upgrade pip setuptools wheel
   ```
2. Install the core library (pulls JAX/JAXLIB automatically)
   ```bash
   pip install -e ./jaxnn
   ```
   The editable install brings in the default CPU wheels; GPU builds have a JIT error.
3. Install hand-pose pipeline extras
   ```bash
   pip install mediapipe opencv-python huggingface_hub tqdm wandb
   ```
   - `wandb` is optional; skip it if you will not log to Weights & Biases.



## Prepare data
- Recommended dataset (higher quality): download from https://universe.roboflow.com/hlcv-jtcas/guitar-chords-eu59l, export locally, and point `hand_pose.py` at it via `--dataset-dir`.
- Export landmarks (downloads the Hugging Face dataset if missing):
  ```bash
  python hand_pose.py --process-dataset
  ```
  Landmarks are saved under `data/guitar-chords_landmarks/`; debug overlays go to `data/guitar-chords_debug_viz/`.

## Train
- Train a baseline model on the exported landmarks:
  ```bash
  python train_hand_pose.py --data-root data/guitar-chords_landmarks --epochs 50 --model-type baseline
  ```
- Run the full sweep used in experiments:
  ```bash
  ./training_commands.sh
  ```

## Inference
- Classify a single image with an existing checkpoint:
  ```bash
  python infer_hand_pose.py --checkpoint checkpoints/<run>/baseline/best.pkl --image path/to/sample.jpg
  ```
- Live webcam demo (requires a camera):
  ```bash
  python webcam_infer.py --checkpoint checkpoints/<run>/baseline/best.pkl
  ```
- Batch evaluate the **secondary** dataset and save annotated results:
  ```bash
  python dataset_infer.py \
    --checkpoint checkpoints/<run>/gat/best.pkl \
    --model-type gat \
    --split valid
  ```
  What it does:
  - Reads the processed secondary landmarks from `data/guitar_chords_landmarks_secondary/<split>`.
  - Saves a JPEG per sample with green/red labels showing the true chord and the model’s prediction; the `.npy` files are copied alongside by default. Use `--no-save-npy-copies` if you only want the JPEGs.

## Maintainers
Chris Melville, Chris Hardwick, Harsh Chandirasekar - melvi083@umn.edu, hardw050@umn.edu, chand863@umn.edu
