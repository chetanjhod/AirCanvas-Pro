import os
import time
import mediapipe as mp
import cv2
import numpy as np


class HandTracker:
    def __init__(self, model_path="models/hand_landmarker.task"):
        self.model_path = model_path
        self.landmarker = None
        self.last_timestamp = 0

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Hand tracking model not found: {self.model_path}"
            )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6
        )

        self.landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            int(time.time() * 1000),
            self.last_timestamp + 1
        )
        self.last_timestamp = timestamp

        result = self.landmarker.detect_for_video(image, timestamp)

        hands = []

        if result.hand_landmarks:
            height, width = frame.shape[:2]

            for landmarks in result.hand_landmarks:
                points = []

                for landmark in landmarks:
                    points.append((
                        int(landmark.x * width),
                        int(landmark.y * height)
                    ))

                hands.append(points)

        return hands

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None