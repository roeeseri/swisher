"""MediaPipe Tasks adapter; no dependency on the retired Solutions API."""

from __future__ import annotations

import platform
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import mediapipe as mp
import numpy as np

POSE_CONNECTIONS = {
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
}


def model_path(variant: str) -> Path:
    if variant not in {"lite", "full", "heavy"}:
        raise ValueError("unsupported pose model")
    target = Path(__file__).resolve().parents[2] / "models" / f"pose_landmarker_{variant}.task"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 100000:
        return target
    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"
    )
    temporary = target.with_suffix(f".{uuid4().hex}.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if temporary.stat().st_size < 100000:
            raise ValueError("pose model download was incomplete")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class Pose:
    def __init__(self, *, model_complexity=2, **_legacy_options):
        variant = {0: "lite", 1: "full", 2: "heavy"}[model_complexity]
        self.path = model_path(variant)
        self.metal = platform.system() == "Darwin"
        delegate = (
            mp.tasks.BaseOptions.Delegate.GPU if self.metal else mp.tasks.BaseOptions.Delegate.CPU
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.path), delegate=delegate),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self.model = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process(self, rgb):
        # Apple Metal requires RGBA pixel buffers; RGB can abort the native process.
        pixels = prepare_pixels(rgb, self.metal)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGBA if self.metal else mp.ImageFormat.SRGB, data=pixels
        )
        prediction = self.model.detect(image)
        landmarks = (
            SimpleNamespace(landmark=prediction.pose_landmarks[0])
            if prediction.pose_landmarks
            else None
        )
        return SimpleNamespace(
            pose_landmarks=landmarks,
            segmentation_mask=(
                prediction.segmentation_masks[0].numpy_view().copy()
                if prediction.segmentation_masks
                else None
            ),
        )

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.model.close()


def prepare_pixels(rgb: np.ndarray, metal: bool) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("pose input must be an RGB uint8 image")
    if metal:
        alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
        return np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
    return np.ascontiguousarray(rgb)
