from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

PERSON_CLASS_ID = 1
SPORTS_BALL_CLASS_ID = 37


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@lru_cache(maxsize=1)
def _model() -> object:
    from rfdetr import RFDETRNano

    return RFDETRNano()


class RfDetrDetector:
    """RF-DETR COCO adapter restricted to people and sports balls."""

    name = "rf-detr-nano-coco"

    def detect(self, rgb_frame: np.ndarray, threshold: float = 0.2) -> list[Detection]:
        result = _model().predict(rgb_frame, threshold=threshold)  # type: ignore[attr-defined]
        detections: list[Detection] = []
        for box, class_id, confidence in zip(result.xyxy, result.class_id, result.confidence):
            class_value = int(class_id)
            if class_value not in {PERSON_CLASS_ID, SPORTS_BALL_CLASS_ID}:
                continue
            detections.append(
                Detection(
                    class_name="person" if class_value == PERSON_CLASS_ID else "sports_ball",
                    confidence=float(confidence),
                    xyxy=tuple(float(value) for value in box),
                )
            )
        return detections


def select_primary_player(
    detections: list[Detection],
    frame_shape: tuple[int, int, int],
    previous_box: tuple[float, float, float, float] | None = None,
) -> Detection | None:
    """Select the likely shooter; phone footage normally has one dominant person."""
    people = [item for item in detections if item.class_name == "person"]
    if not people:
        return None
    height, width = frame_shape[:2]
    frame_center = np.array([width / 2, height / 2])
    diagonal = max(float(np.hypot(width, height)), 1.0)
    balls = [item for item in detections if item.class_name == "sports_ball"]
    previous_center = None
    if previous_box:
        previous_center = np.array(
            [(previous_box[0] + previous_box[2]) / 2, (previous_box[1] + previous_box[3]) / 2]
        )

    def score(item: Detection) -> float:
        item_center = np.array(item.center)
        center_distance = float(np.linalg.norm(item_center - frame_center)) / diagonal
        area_ratio = item.area / max(width * height, 1)
        continuity = 0.0
        if previous_center is not None:
            continuity = 1.0 - min(
                float(np.linalg.norm(item_center - previous_center)) / diagonal, 1.0
            )
        ball_affinity = 0.0
        if balls:
            nearest_ball = min(
                float(np.linalg.norm(item_center - np.array(ball.center))) for ball in balls
            )
            ball_affinity = 1.0 - min(nearest_ball / (diagonal * 0.45), 1.0)
        return (
            2.0 * area_ratio
            + 0.7 * item.confidence
            + 0.55 * continuity
            + 0.8 * ball_affinity
            - 0.1 * center_distance
        )

    return max(people, key=score)


def padded_crop(
    box: tuple[float, float, float, float], width: int, height: int, padding: float = 0.45
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    x_pad, y_pad = box_width * padding, box_height * padding
    return (
        max(0, int(x1 - x_pad)),
        max(0, int(y1 - y_pad)),
        min(width, int(x2 + x_pad)),
        min(height, int(y2 + y_pad)),
    )


def smooth_box(
    previous: tuple[float, float, float, float] | None,
    current: tuple[float, float, float, float],
    alpha: float = 0.35,
) -> tuple[float, float, float, float]:
    if previous is None:
        return current
    return tuple((1 - alpha) * old + alpha * new for old, new in zip(previous, current))
