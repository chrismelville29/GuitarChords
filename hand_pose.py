import math
import random
from pathlib import Path
from typing import Iterable, Optional

import cv2
import mediapipe as mp
from huggingface_hub import snapshot_download
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

DATASET_REPO = "dduka/guitar-chords"
DEFAULT_DATASET_DIR = Path("data/guitar-chords")
DEFAULT_SECONDARY_DATASET_DIR = Path("data/secondary_data")
SECONDARY_DEFAULT_SPLITS = (0.8, 0.1, 0.1)  # train, valid, test
SECONDARY_SPLIT_SEED = 1337
DEFAULT_OUTPUT_DIR = Path("data/guitar-chords_landmarks")
DEFAULT_ORIGINAL_OUTPUT_DIR = Path("data/guitar-chords_landmarks_original")
DEFAULT_DEBUG_VIZ_DIR = Path("data/guitar-chords_debug_viz")
MODEL_PATH = Path("./models/hand_landmarker.task")
MIN_CONFIDENCE_FLOOR = 0.4
DEFAULT_MIN_HAND_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE = 0.5
FALLBACK_MIN_HAND_DETECTION_CONFIDENCE = MIN_CONFIDENCE_FLOOR
FALLBACK_MIN_HAND_PRESENCE_CONFIDENCE = MIN_CONFIDENCE_FLOOR
FINAL_FALLBACK_DETECTION_CONFIDENCE = MIN_CONFIDENCE_FLOOR
FINAL_FALLBACK_PRESENCE_CONFIDENCE = MIN_CONFIDENCE_FLOOR

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  if len(hand_landmarks_list) == 0:
    cv2.putText(annotated_image, "no hand detected", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, (0, 0, 255), FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def _clamp_confidence(value: float) -> float:
  """
  Enforces the minimum confidence floor to avoid overly-permissive detections.
  """
  return max(value, MIN_CONFIDENCE_FLOOR)


def create_landmarker(min_detection_confidence: float = DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
                      min_presence_confidence: float = DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE,
                      num_hands: int = 2):
  """
  Builds a reusable MediaPipe hand landmarker.
  """
  BaseOptions = mp.tasks.BaseOptions
  HandLandmarker = mp.tasks.vision.HandLandmarker
  HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  options = HandLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
      running_mode=VisionRunningMode.IMAGE,
      num_hands=num_hands,
      min_hand_detection_confidence=_clamp_confidence(min_detection_confidence),
      min_hand_presence_confidence=_clamp_confidence(min_presence_confidence))
  return HandLandmarker.create_from_options(options)


def _select_primary_hand_index(hand_landmarks_list):
  """
  Picks the hand closest to the center of the frame.
  """
  if len(hand_landmarks_list) == 0:
    raise Exception("no hands found")
  if len(hand_landmarks_list) == 1:
    return 0

  hand_one_wrist = hand_landmarks_list[0][0].x
  hand_two_wrist = hand_landmarks_list[1][0].x
  return 0 if abs(hand_one_wrist - .5) < abs(hand_two_wrist - .5) else 1


def _extract_world_landmarks(hand_landmarker_result):
  hand_landmarks_list = hand_landmarker_result.hand_landmarks
  hand_world_landmarks = hand_landmarker_result.hand_world_landmarks

  if len(hand_landmarks_list) == 0:
    return None

  idx = _select_primary_hand_index(hand_landmarks_list)
  return np.array([(lm.x, lm.y, lm.z) for lm in hand_world_landmarks[idx]], dtype=np.float32)


def _preprocess_image(image: np.ndarray) -> np.ndarray:
  """
  Applies image enhancements to improve hand detection in challenging conditions.
  Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast.
  """
  # Convert to LAB color space for better processing
  lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
  l, a, b = cv2.split(lab)

  # Apply CLAHE to L channel for contrast enhancement
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  l_enhanced = clahe.apply(l)

  # Merge channels and convert back to RGB
  enhanced_lab = cv2.merge([l_enhanced, a, b])
  enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

  return enhanced_rgb


def _detect_world_landmarks(mp_image: mp.Image,
                            landmarker,
                            fallback_confidence: Optional[float] = FALLBACK_MIN_HAND_DETECTION_CONFIDENCE,
                            use_preprocessing: bool = True):
  """
  Runs detection with multi-stage fallback strategy:
  1. Try with original settings
  2. Retry with clamped fallback confidence (never below 0.4)
  3. Apply image preprocessing and retry with the same clamped confidence
  Returns (world_landmarks or None, detection_result_used).
  """
  # First attempt with original settings
  result = landmarker.detect(mp_image)
  world = _extract_world_landmarks(result)
  if world is not None:
    return world, result

  if fallback_confidence is None:
    return world, result

  fallback_confidence = _clamp_confidence(fallback_confidence)
  fallback_presence_confidence = _clamp_confidence(FALLBACK_MIN_HAND_PRESENCE_CONFIDENCE)

  # Second attempt with fallback confidence
  with create_landmarker(
      min_detection_confidence=fallback_confidence,
      min_presence_confidence=fallback_presence_confidence
  ) as fallback_landmarker:
    fallback_result = fallback_landmarker.detect(mp_image)
    world_fallback = _extract_world_landmarks(fallback_result)
    if world_fallback is not None:
      return world_fallback, fallback_result

  # Third attempt with image preprocessing and the clamped confidence
  if use_preprocessing:
    enhanced_image = _preprocess_image(mp_image.numpy_view())
    mp_enhanced = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_image)

    with create_landmarker(
        min_detection_confidence=_clamp_confidence(FINAL_FALLBACK_DETECTION_CONFIDENCE),
        min_presence_confidence=_clamp_confidence(FINAL_FALLBACK_PRESENCE_CONFIDENCE)
    ) as final_landmarker:
      final_result = final_landmarker.detect(mp_enhanced)
      world_final = _extract_world_landmarks(final_result)
      return world_final, final_result

  return None, result


def get_landmarks(im_path,
                  min_detection_confidence=DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
                  min_presence_confidence=DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE,
                  show_marks=False):
    """
    Takes in a image path and returns the world coordinates of the center most hand
    im_path: Path to image (string)
    min_detection_confidence: minimum detection threshold for a hand to be considered a hand
    min_presence_confidence: minimum confidence for hand presence in the image
    show_marks: bool whether to display a window with the detected points (displays both hands)
    """

    def _detect(detector):
      mp_image = mp.Image.create_from_file(str(im_path))
      world, detection_result = _detect_world_landmarks(mp_image, detector)

      if show_marks and detection_result is not None:
        overlay = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("hand detection", overlay)
        cv2.waitKey(0)

      if world is None:
        raise Exception(f"no hands found: {im_path}")
      return world

    with create_landmarker(
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence
    ) as landmarker:
      return _detect(landmarker)
    
def normalize_hand(hand_pose):
  """
  Normalizes hand pose so that point 0 is 0, 0, 0, and point 1 is 0, 0, 1
  hand_pose: a N(21) by 3 numpy array of hand points
  returns: a N(21) by 3 numpy array of normalized hand points
  """
  A = hand_pose[0] # wrist
  B = hand_pose[5] # Index finger mcp (base of index finger)

  v = B - A
  v /= np.linalg.norm(v)

  rotation_axis = np.cross(v, (0, 0, 1))
  rotation_axis /= np.linalg.norm(rotation_axis)
  angle = np.arccos(np.dot(v, (0, 0, 1)))  

  cross_product_matrix = np.array([
     [0, -rotation_axis[2], rotation_axis[1]],
     [rotation_axis[2], 0, -rotation_axis[0]],
     [-rotation_axis[1], rotation_axis[0], 0]
  ])

  rodrigues = np.eye(3) * math.cos(angle) + np.outer(rotation_axis, rotation_axis) * (1 - math.cos(angle)) + cross_product_matrix * math.sin(angle)

  scale_matrix = (1 / np.linalg.norm(B - A)) * np.eye(3)

  scale_rotation = scale_matrix @ rodrigues

  scale_rotation_translation = -scale_matrix @ rodrigues @ hand_pose[0]

  transformation_matrix = np.array(
     [[scale_rotation[0, 0], scale_rotation[0, 1], scale_rotation[0, 2], scale_rotation_translation[0]],
     [scale_rotation[1, 0], scale_rotation[1, 1], scale_rotation[1, 2], scale_rotation_translation[1]],
     [scale_rotation[2, 0], scale_rotation[2, 1], scale_rotation[2, 2], scale_rotation_translation[2]],
     [0, 0, 0, 1]])

  homogenous_coords = np.hstack((hand_pose, np.ones((hand_pose.shape[0], 1))))

  transformed = homogenous_coords @ transformation_matrix.T

  return transformed[:, :3]


def normalize_to_wrist(hand_pose: np.ndarray):
  """
  Produces a 20-point pose expressed relative to the wrist (landmark 0).
  Translation: shift so wrist is the origin, then drop the wrist point.
  Scale: divide by palm size (mean distance wrist→MCP joints 5,9,13,17) so magnitudes
  are comparable across hands.
  """
  if hand_pose.shape[0] != 21:
    raise ValueError(f"Expected 21 points, got shape {hand_pose.shape}")

  wrist = hand_pose[0]
  centered = hand_pose[1:] - wrist

  ref_ids = [5, 9, 13, 17]
  dists = [np.linalg.norm(hand_pose[i] - wrist) for i in ref_ids]
  scale = float(np.mean(dists))

  if scale <= 1e-8:
    return centered

  return centered / scale


def download_guitar_chords_dataset(target_dir: Path = DEFAULT_DATASET_DIR,
                                   repo_id: str = DATASET_REPO,
                                   revision: Optional[str] = None) -> Path:
  """
  Downloads the guitar-chords dataset from Hugging Face if it's not already present.
  """
  target_dir = Path(target_dir)
  target_dir.mkdir(parents=True, exist_ok=True)

  # If we already see jpgs, assume the snapshot is present.
  if any(target_dir.glob("**/*.jpg")):
    return target_dir

  snapshot_download(
      repo_id=repo_id,
      repo_type="dataset",
      local_dir=target_dir,
      local_dir_use_symlinks=False,
      resume_download=True,
      revision=revision,
  )
  return target_dir


def _iter_image_files(root: Path, splits: Optional[Iterable[str]] = None) -> list[Path]:
  """
  Collects image files under the root (optionally restricted to a set of splits).
  """
  search_dirs = [root]
  if splits:
    search_dirs = [root / split for split in splits]

  image_files: list[Path] = []
  for search_dir in search_dirs:
    if search_dir.exists():
      image_files.extend(sorted(search_dir.rglob("*.jpg")))
  return sorted(image_files)


def _prepare_secondary_images(root: Path,
                              valid_frac: float = SECONDARY_DEFAULT_SPLITS[1],
                              test_frac: float = SECONDARY_DEFAULT_SPLITS[2],
                              seed: int = SECONDARY_SPLIT_SEED) -> tuple[list[tuple[Path, Path]], dict[str, int]]:
  """
  Splits an unsplit secondary dataset into train/valid/test and returns output-relative paths.
  Images are renamed with a 'secondary_' prefix to avoid collisions with the primary dataset.
  """
  root = Path(root)
  source_root = root / "train" if (root / "train").exists() else root

  if valid_frac < 0 or test_frac < 0:
    raise ValueError("Secondary split fractions must be non-negative.")
  if valid_frac + test_frac >= 1.0:
    raise ValueError("valid_frac + test_frac must be < 1.0 for the secondary dataset.")

  train_frac = 1.0 - valid_frac - test_frac
  rng = random.Random(seed)

  prepared: list[tuple[Path, Path]] = []
  split_counts: dict[str, int] = {"train": 0, "valid": 0, "test": 0}

  for chord_dir in sorted([p for p in source_root.iterdir() if p.is_dir()]):
    image_paths = sorted(chord_dir.glob("*.jpg"))
    if not image_paths:
      continue

    rng.shuffle(image_paths)
    total = len(image_paths)

    train_count = int(total * train_frac)
    valid_count = int(total * valid_frac)

    # Guarantee at least one training example when images are present.
    if total > 0 and train_count == 0:
      train_count = 1

    # Avoid over-allocation from rounding.
    if train_count + valid_count > total:
      valid_count = max(0, total - train_count)

    test_count = total - train_count - valid_count

    split_slices = {
        "train": image_paths[:train_count],
        "valid": image_paths[train_count:train_count + valid_count],
        "test": image_paths[train_count + valid_count:],
    }

    for split_name, paths in split_slices.items():
      for img_path in paths:
        # Prefix filename to avoid clobbering primary dataset samples.
        rel_path = Path(split_name) / chord_dir.name / f"secondary_{img_path.name}"
        prepared.append((img_path, rel_path))
        split_counts[split_name] += 1

  return prepared, split_counts

def _count_images_in_dir(directory: Path) -> int:
  return sum(1 for _ in directory.glob("*.jpg"))


def dataset_split_stats(dataset_dir: Path = DEFAULT_DATASET_DIR,
                        splits: Optional[Iterable[str]] = None) -> dict[str, dict[str, int]]:
  """
  Returns nested counts: {split: {chord: num_images}}.
  """
  dataset_dir = Path(dataset_dir)
  split_names = list(splits) if splits else sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])

  stats: dict[str, dict[str, int]] = {}
  for split in split_names:
    split_dir = dataset_dir / split
    if not split_dir.exists():
      continue
    chord_counts: dict[str, int] = {}
    for chord_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
      chord_counts[chord_dir.name] = _count_images_in_dir(chord_dir)
    stats[split] = chord_counts
  return stats


def print_dataset_stats(dataset_dir: Path = DEFAULT_DATASET_DIR,
                        splits: Optional[Iterable[str]] = None):
  dataset_dir = download_guitar_chords_dataset(dataset_dir)
  stats = dataset_split_stats(dataset_dir, splits=splits)
  if not stats:
    print(f"No splits found under {dataset_dir}")
    return

  total_images = 0
  chord_totals: dict[str, int] = {}

  for split, chord_counts in stats.items():
    split_total = sum(chord_counts.values())
    total_images += split_total
    print(f"{split}: {split_total} images")
    for chord, count in sorted(chord_counts.items()):
      print(f"  {chord}: {count}")
      chord_totals[chord] = chord_totals.get(chord, 0) + count

  if chord_totals:
    print("Overall by chord:")
    for chord, count in sorted(chord_totals.items()):
      print(f"  {chord}: {count}")

  print(f"Grand total: {total_images} images")


def process_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR,
                    output_dir: Path = DEFAULT_OUTPUT_DIR,
                    min_detection_confidence: float = DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
                    debug_viz_dir: Path = DEFAULT_DEBUG_VIZ_DIR,
                    limit: Optional[int] = None,
                    splits: Optional[Iterable[str]] = None,
                    include_secondary: bool = False,
                    secondary_dir: Path = DEFAULT_SECONDARY_DATASET_DIR,
                    secondary_valid_frac: float = SECONDARY_DEFAULT_SPLITS[1],
                    secondary_test_frac: float = SECONDARY_DEFAULT_SPLITS[2],
                    secondary_seed: int = SECONDARY_SPLIT_SEED,
                    keep_original: bool = False):
  """
  Downloads (if necessary) and processes the dataset into 20-point wrist-normalized npy files by default,
  or preserves the original 21-point coordinates when keep_original=True.
  Optionally also folds in the local ./data/secondary_data set by auto-splitting it into
  train/valid/test before processing.
  """
  dataset_dir = download_guitar_chords_dataset(dataset_dir)
  output_dir = Path(output_dir)
  if keep_original and output_dir == DEFAULT_OUTPUT_DIR:
    output_dir = DEFAULT_ORIGINAL_OUTPUT_DIR
  debug_viz_dir = Path(debug_viz_dir)

  export_desc = "21-point original landmarks" if keep_original else "20-point wrist-normalized landmarks"
  print(f"Exporting {export_desc} to {output_dir}")

  images: list[tuple[Path, Path]] = []

  primary_images = _iter_image_files(dataset_dir, splits=splits)
  for image_path in primary_images:
    rel_path = image_path.relative_to(dataset_dir)
    images.append((image_path, rel_path))

  secondary_counts = None
  if include_secondary:
    secondary_dir = Path(secondary_dir)
    if not secondary_dir.exists():
      print(f"Secondary dataset directory {secondary_dir} not found; skipping.")
    else:
      secondary_images, secondary_counts = _prepare_secondary_images(
          secondary_dir,
          valid_frac=secondary_valid_frac,
          test_frac=secondary_test_frac,
          seed=secondary_seed,
      )
      images.extend(secondary_images)
      print(
          f"Added secondary dataset with splits: "
          f"{secondary_counts['train']} train / {secondary_counts['valid']} valid / {secondary_counts['test']} test images"
      )

  total_images = len(images)
  if limit is not None:
    images = images[:limit]
    if limit < total_images:
      print(f"Limit set to {limit}; processing first {len(images)} of {total_images} combined images.")

  failures = []
  skipped_missing = 0
  saved_count = 0
  with create_landmarker(
      min_detection_confidence=min_detection_confidence,
      min_presence_confidence=DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE
  ) as landmarker:
    for idx, (image_path, rel_path) in enumerate(images, start=1):
      out_path = (output_dir / rel_path).with_suffix(".npy")
      debug_path = (debug_viz_dir / rel_path)
      try:
        mp_image = mp.Image.create_from_file(str(image_path))
        world_landmarks, detection_result = _detect_world_landmarks(mp_image, landmarker)
        if world_landmarks is None:
          if keep_original:
            skipped_missing += 1
            for stale_path in (out_path, debug_path):
              if stale_path.exists():
                stale_path.unlink()
            continue
          raise Exception("no hands found")

        exported_landmarks = world_landmarks if keep_original else normalize_to_wrist(world_landmarks)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, exported_landmarks.astype(np.float32))
        saved_count += 1

        # Save debug visualization with landmarks drawn
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if idx % 200 == 0:
          print(f"Processed {idx} / {len(images)} images")
      except Exception as exc:
        failures.append((str(image_path), str(exc)))
        for stale_path in (out_path, debug_path):
          if stale_path.exists():
            stale_path.unlink()

  print(f"Finished processing {len(images)} images with {len(failures)} failures; saved {saved_count} samples.")
  if skipped_missing:
    print(f"Skipped {skipped_missing} images without detections (original-export mode).")
  if failures:
    print("Failed files:")
    for path, msg in failures[:10]:
      print(f"  {path}: {msg}")
    if len(failures) > 10:
      print(f"  ... {len(failures) - 10} more")

  result = {"processed": saved_count, "failed": failures}
  if skipped_missing:
    result["skipped"] = skipped_missing
  return result


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Hand pose extraction utilities.")
  parser.add_argument("--image", help="Path to a single image to convert to wrist-normalized npy.")
  parser.add_argument("--output", help="Output path for the npy file when using --image.")
  parser.add_argument("--show", action="store_true", help="Show detection overlay for single image processing.")
  parser.add_argument("--original", action="store_true",
                      help="Keep the original 21-point landmark output instead of wrist-normalized 20-point data.")
  parser.add_argument("--process-dataset", action="store_true",
                      help="Automatically download dduka/guitar-chords and export wrist-normalized npy files.")
  parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                      help="Where to download/read the dataset.")
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                      help="Where to write processed npy files (mirrors dataset structure).")
  parser.add_argument("--debug-viz-dir", type=Path, default=DEFAULT_DEBUG_VIZ_DIR,
                      help="Where to write debug landmark overlays (mirrors dataset structure).")
  parser.add_argument("--min-detection-confidence", type=float,
                      default=DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
                      help="Minimum confidence threshold for hand detection.")
  parser.add_argument("--limit", type=int, help="Limit number of images when processing the dataset.")
  parser.add_argument("--splits", nargs="+", help="Optional list of dataset splits to process (e.g. train valid test).")
  parser.add_argument("--dataset-stats", action="store_true",
                      help="Print image counts per split and chord for the dataset directory.")
  parser.add_argument("--include-secondary", action="store_true",
                      help="Also process ./data/secondary_data by auto-splitting it into train/valid/test.")
  parser.add_argument("--secondary-dir", type=Path, default=DEFAULT_SECONDARY_DATASET_DIR,
                      help="Root directory for the raw secondary dataset.")
  parser.add_argument("--secondary-valid-frac", type=float, default=SECONDARY_DEFAULT_SPLITS[1],
                      help="Validation fraction for the auto-split of the secondary dataset (default 0.1).")
  parser.add_argument("--secondary-test-frac", type=float, default=SECONDARY_DEFAULT_SPLITS[2],
                      help="Test fraction for the auto-split of the secondary dataset (default 0.1).")
  parser.add_argument("--secondary-seed", type=int, default=SECONDARY_SPLIT_SEED,
                      help="Random seed used when shuffling secondary data before splitting.")
  args = parser.parse_args()

  ran = False
  dataset_output_dir = args.output_dir
  if args.original and dataset_output_dir == DEFAULT_OUTPUT_DIR:
    dataset_output_dir = DEFAULT_ORIGINAL_OUTPUT_DIR

  if args.image:
    world_landmarks = get_landmarks(args.image,
                                    min_detection_confidence=args.min_detection_confidence,
                                    show_marks=args.show)
    exported_landmarks = world_landmarks if args.original else normalize_to_wrist(world_landmarks)
    output_path = Path(args.output) if args.output else Path(args.image).with_suffix(".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, exported_landmarks.astype(np.float32))
    descriptor = "21-point original landmarks" if args.original else "20-point wrist-normalized landmarks"
    print(f"Saved {descriptor} to {output_path}")
    ran = True

  if args.dataset_stats:
    print_dataset_stats(dataset_dir=args.dataset_dir, splits=args.splits)
    ran = True

  if args.process_dataset:
    process_dataset(dataset_dir=args.dataset_dir,
                    output_dir=dataset_output_dir,
                    min_detection_confidence=args.min_detection_confidence,
                    debug_viz_dir=args.debug_viz_dir,
                    limit=args.limit,
                    splits=args.splits,
                    include_secondary=args.include_secondary,
                    secondary_dir=args.secondary_dir,
                    secondary_valid_frac=args.secondary_valid_frac,
                    secondary_test_frac=args.secondary_test_frac,
                    secondary_seed=args.secondary_seed,
                    keep_original=args.original)
    ran = True

  if not ran:
    parser.print_help()
