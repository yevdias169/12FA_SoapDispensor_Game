import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core import base_options as base_options_module

from . import config


class ModelWrapper:
    """
    Loads the trained RandomForest + MediaPipe pipeline exactly once.
    Call predict(frame_bgr) per frame; call close() on teardown.
    """

    def __init__(self):
        # Load sklearn RandomForestClassifier from pickle (matches Train Classifier.py)
        model_dict = pickle.load(open(config.MODEL_PATH, 'rb'))
        self.model = model_dict['model']

        # HandLandmarker in IMAGE mode — no running_mode set, matching test_classifier.py
        # (omitting running_mode defaults to IMAGE in mediapipe 0.10.x)
        BaseOptions = base_options_module.BaseOptions
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.HAND_LANDMARKER_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = HandLandmarker.create_from_options(options)

    def predict(self, frame_bgr: np.ndarray) -> int | None:
        """
        Return the predicted finger count (1–5), or None if no hand is detected.

        Preprocessing replicates test_classifier.py exactly:
          1. BGR → RGB                      (test_classifier.py line 38)
          2. mp.Image(SRGB)                 (test_classifier.py line 41)
          3. Wrist-relative normalisation   (test_classifier.py lines 48-52)
             — wrist is landmark 0; 21 landmarks × 2 coords = 42 features
          4. model.predict() → class 0-4 → finger count = class + 1
        """
        # Step 1-2: colour conversion + MediaPipe wrap
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.detector.detect(mp_image)
        if not results.hand_landmarks:
            return None

        # Step 3: wrist-relative normalisation (mirrors test_classifier.py lines 48-52)
        hand = results.hand_landmarks[0]
        wrist_x, wrist_y = hand[0].x, hand[0].y
        data_aux = []
        for lm in hand:
            data_aux.append(lm.x - wrist_x)
            data_aux.append(lm.y - wrist_y)
        # 21 landmarks × 2 = 42 features; matches model.n_features_in_ == 42

        # Step 4: class 0-4 → finger count 1-5 (matches test_classifier.py line 56)
        prediction = self.model.predict([np.asarray(data_aux)])
        return int(prediction[0]) + 1

    def close(self):
        self.detector.close()
