import math
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
DEFAULT_OUTPUT_DIR = Path("data/guitar-chords_landmarks")
DEFAULT_DEBUG_VIZ_DIR = Path("data/guitar-chords_debug_viz")
MODEL_PATH = Path("./models/hand_landmarker.task")
DEFAULT_MIN_HAND_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE = 0.5
FALLBACK_MIN_HAND_DETECTION_CONFIDENCE = 0.3
FALLBACK_MIN_HAND_PRESENCE_CONFIDENCE = 0.3
FINAL_FALLBACK_DETECTION_CONFIDENCE = 0.1
FINAL_FALLBACK_PRESENCE_CONFIDENCE = 0.1

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
      min_hand_detection_confidence=min_detection_confidence,
      min_hand_presence_confidence=min_presence_confidence)
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
  2. Retry with lower confidence thresholds
  3. Apply image preprocessing and retry with lowest confidence
  Returns (world_landmarks or None, detection_result_used).
  """
  # First attempt with original settings
  result = landmarker.detect(mp_image)
  world = _extract_world_landmarks(result)
  if world is not None:
    return world, result

  if fallback_confidence is None:
    return world, result

  # Second attempt with fallback confidence
  with create_landmarker(
      min_detection_confidence=fallback_confidence,
      min_presence_confidence=FALLBACK_MIN_HAND_PRESENCE_CONFIDENCE
  ) as fallback_landmarker:
    fallback_result = fallback_landmarker.detect(mp_image)
    world_fallback = _extract_world_landmarks(fallback_result)
    if world_fallback is not None:
      return world_fallback, fallback_result

  # Third attempt with image preprocessing and very low confidence
  if use_preprocessing:
    enhanced_image = _preprocess_image(mp_image.numpy_view())
    mp_enhanced = mp.Image(image_format=mp.ImageFormat.SRGB, data=enhanced_image)

    with create_landmarker(
        min_detection_confidence=FINAL_FALLBACK_DETECTION_CONFIDENCE,
        min_presence_confidence=FINAL_FALLBACK_PRESENCE_CONFIDENCE
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


def process_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR,
                    output_dir: Path = DEFAULT_OUTPUT_DIR,
                    min_detection_confidence: float = DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
                    debug_viz_dir: Path = DEFAULT_DEBUG_VIZ_DIR,
                    limit: Optional[int] = None,
                    splits: Optional[Iterable[str]] = None):
  """
  Downloads (if necessary) and processes the dataset into 20-point wrist-normalized npy files.
  """
  dataset_dir = download_guitar_chords_dataset(dataset_dir)
  output_dir = Path(output_dir)
  images = _iter_image_files(dataset_dir, splits=splits)

  if limit is not None:
    images = images[:limit]

  failures = []
  debug_viz_dir = Path(debug_viz_dir)
  with create_landmarker(
      min_detection_confidence=min_detection_confidence,
      min_presence_confidence=DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE
  ) as landmarker:
    for idx, image_path in enumerate(images, start=1):
      try:
        mp_image = mp.Image.create_from_file(str(image_path))
        world_landmarks, detection_result = _detect_world_landmarks(mp_image, landmarker)
        if world_landmarks is None:
          raise Exception("no hands found")

        normalized_landmarks = normalize_to_wrist(world_landmarks)

        rel_path = image_path.relative_to(dataset_dir)
        out_path = (output_dir / rel_path).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, normalized_landmarks.astype(np.float32))

        # Save debug visualization with landmarks drawn
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        debug_path = (debug_viz_dir / rel_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if idx % 200 == 0:
          print(f"Processed {idx} / {len(images)} images")
      except Exception as exc:
        failures.append((str(image_path), str(exc)))

  print(f"Finished processing {len(images)} images with {len(failures)} failures.")
  if failures:
    print("Failed files:")
    for path, msg in failures[:10]:
      print(f"  {path}: {msg}")
    if len(failures) > 10:
      print(f"  ... {len(failures) - 10} more")

  return {"processed": len(images) - len(failures), "failed": failures}


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Hand pose extraction utilities.")
  parser.add_argument("--image", help="Path to a single image to convert to wrist-normalized npy.")
  parser.add_argument("--output", help="Output path for the npy file when using --image.")
  parser.add_argument("--show", action="store_true", help="Show detection overlay for single image processing.")
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
  args = parser.parse_args()

  ran = False

  if args.image:
    world_landmarks = get_landmarks(args.image,
                                    min_detection_confidence=args.min_detection_confidence,
                                    show_marks=args.show)
    normalized_landmarks = normalize_to_wrist(world_landmarks)
    output_path = Path(args.output) if args.output else Path(args.image).with_suffix(".npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, normalized_landmarks.astype(np.float32))
    print(f"Saved 20-point wrist-normalized landmarks to {output_path}")
    ran = True

  if args.process_dataset:
    process_dataset(dataset_dir=args.dataset_dir,
                    output_dir=args.output_dir,
                    min_detection_confidence=args.min_detection_confidence,
                    debug_viz_dir=args.debug_viz_dir,
                    limit=args.limit,
                    splits=args.splits)
    ran = True

  if not ran:
    parser.print_help()
