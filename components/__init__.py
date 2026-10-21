import os
import threading

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ),
    "models",
    "hand_landmarker.task"
)