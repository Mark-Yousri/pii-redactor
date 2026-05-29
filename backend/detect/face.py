import os
import urllib.request
from dataclasses import dataclass

import mediapipe as mp
import numpy as np

import backend.config as config

_detector = None

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        print(f"[face] Downloading MediaPipe face model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[face] Model downloaded.")


def load_face_detector():
    global _detector
    if _detector is None:
        _ensure_model()
        base_opts = mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_opts,
            min_detection_confidence=config.FACE_DET_THRESHOLD,
        )
        _detector = mp.tasks.vision.FaceDetector.create_from_options(opts)
    return _detector


@dataclass
class FaceDetection:
    bbox: tuple  # x0, y0, x1, y1 in pixel space
    confidence: float
    page_num: int


def detect_faces(image: np.ndarray, page_num: int) -> list:
    detector = load_face_detector()
    h, w = image.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = detector.detect(mp_image)
    detections = []
    for det in result.detections:
        score = det.categories[0].score
        box = det.bounding_box
        x0 = max(0, box.origin_x)
        y0 = max(0, box.origin_y)
        x1 = min(w, box.origin_x + box.width)
        y1 = min(h, box.origin_y + box.height)
        detections.append(FaceDetection(
            bbox=(float(x0), float(y0), float(x1), float(y1)),
            confidence=float(score),
            page_num=page_num,
        ))
    return detections
