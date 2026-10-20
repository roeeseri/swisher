from types import SimpleNamespace

import cv2
import numpy as np

from swisher import analysis
from swisher.detection import Detection


def test_selection_region_excludes_poster_without_cropping_raised_hands(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source fixture")
    calls = []

    class Capture:
        used = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 1,
                cv2.CAP_PROP_POS_MSEC: 0,
            }.get(prop, 0)

        def read(self):
            if self.used:
                return False, None
            self.used = True
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            pass

    class Pose:
        path = source

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def process(self, crop):
            calls.append(crop.shape)
            return SimpleNamespace(pose_landmarks=None)

    class Detector:
        name = "fixture"

        def detect(self, *args, **kwargs):
            return [Detection("person", .9, (250, 200, 350, 400)),
                    Detection("person", .9, (250, 60, 350, 150))]

    monkeypatch.setattr(analysis.cv2, "VideoCapture", lambda _: Capture())
    monkeypatch.setattr(analysis.pose_backend, "Pose", Pose)
    monkeypatch.setattr(analysis, "RfDetrDetector", Detector)
    monkeypatch.setattr(analysis, "_video_writer", lambda *args: SimpleNamespace(
        write=lambda frame: None, release=lambda: None))
    result = analysis.analyze_video(
        source, tmp_path / "overlay.mp4", player_region=(100, 50, 500, 450),
        player_selection_region=(200, 200, 400, 400),
    )
    assert calls == [(400, 400, 3)]
    assert result.detections.player_detected_frames == 1
    assert result.provenance["player_selection_region"] == (200, 200, 400, 400)
