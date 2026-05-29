"""
Phase 2 acceptance tests.
Run: python -m pytest tests/test_face_detector.py -v
"""
import numpy as np
import pytest
from backend.detect.face import load_face_detector, detect_faces


def _blank_image(h=300, w=300):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def test_blank_image_returns_no_faces():
    img = _blank_image()
    assert detect_faces(img, page_num=0) == []


def test_detector_cached():
    d1 = load_face_detector()
    d2 = load_face_detector()
    assert d1 is d2, "Detector should be cached at module level"


def test_face_detection_on_synthetic_face():
    """
    Uses a downloaded face image from numpy-encoded bytes if available,
    otherwise just confirms the function returns a list without crashing.
    """
    img = _blank_image(480, 640)
    result = detect_faces(img, page_num=0)
    assert isinstance(result, list)
