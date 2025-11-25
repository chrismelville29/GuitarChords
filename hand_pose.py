import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
import cv2
import sys

from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math

import os

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  if len(hand_landmarks_list) == 0:
      raise("BOTH HANDS NOT FOUND")
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


def get_landmarks(im_path, min_detection_confidence=.12, show_marks=False):
    """
    Takes in a image path and returns the world coordinates of the center most hand
    im_path: Path to image (string)
    min_detection_confidence: minimum detection threshold for a hand to be considered a hand, after minimal testing .12 was found to detect all the hands in a portion of our dataset
    show_marks: bool whether to display a window with the detected points (displays both hands)
    """

    model_path = './models/hand_landmarker.task'

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the image mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence)

    with HandLandmarker.create_from_options(options) as landmarker:
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(im_path)

        hand_landmarker_result = landmarker.detect(mp_image)

        # display hand points over image
        if show_marks:
            cv2.imshow("hand detection", draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result))
            cv2.waitKey(0)

        # print(vars(hand_landmarker_result))
        hand_landmarks_list = hand_landmarker_result.hand_landmarks
        handedness_list = hand_landmarker_result.handedness
        hand_world_landmarks = hand_landmarker_result.hand_world_landmarks

        if len(handedness_list) == 0:
          raise Exception("no hands found", im_path)
        elif len(handedness_list) == 1:
          handness = handedness_list[0][0].display_name
        elif len(handedness_list) == 2:
          hand_one_wrist = hand_landmarks_list[0][0].x
          hand_two_wrist = hand_landmarks_list[1][0].x

          # check which hand is closest to the center of the camera
          if abs(hand_one_wrist - .5) < abs(hand_two_wrist - .5):
            handness = handedness_list[0][0].display_name
          else:
            handness = handedness_list[1][0].display_name
          
        world_landmarks = []

        for i in range(len(hand_landmarks_list)):
            if handedness_list[i][0].display_name == handness:
                for j in range(len(hand_landmarks_list[i])):
                    landmark = hand_world_landmarks[i][j]
                    world_landmarks.append((landmark.x, landmark.y, landmark.z))
            
        return np.array(world_landmarks)
    
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


if __name__ == "__main__":

  # example get_landmarks call:
  dir = "./data/train/"
  key = sys.argv[1] + "/"
  img_no = sys.argv[2]
  img = "image_" + img_no + ".jpg"

  landmarks = get_landmarks(dir + key + img, show_marks=False)
  normalized_landmarks = normalize_hand(landmarks)
  print(landmarks)
  print(normalized_landmarks)