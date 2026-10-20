import cv2
import numpy as np

from swisher.camera_tracking import CameraRegionTracker


def test_anchor_region_follows_camera_translation_and_recovers_after_occlusion():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, -55], [0, 1, 20]]), (640, 480))
    tracker = CameraRegionTracker(image, (200, 150, 350, 400))
    assert tracker.update(shifted).matrix is not None
    region = tracker.region((200, 150, 350, 400))
    assert np.allclose(region, (145, 170, 295, 420), atol=2)
    assert tracker.update(np.zeros_like(image)).matrix is None
    assert tracker.region((200, 150, 350, 400)) is None
    assert tracker.update(shifted).matrix is not None


def test_featureless_camera_abstains():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker = CameraRegionTracker(image)
    assert tracker.update(image).matrix is None
