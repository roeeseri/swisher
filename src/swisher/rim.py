from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))


@dataclass(frozen=True)
class RimDetection:
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]


@lru_cache(maxsize=1)
def _pipeline() -> object:
    from transformers import pipeline

    return pipeline(
        "zero-shot-object-detection",
        model="IDEA-Research/grounding-dino-tiny",
        device="cpu",
    )


class GroundingDinoRimDetector:
    name = "grounding-dino-tiny-zero-shot"

    def detect(self, rgb_frame: np.ndarray) -> RimDetection | None:
        height, width = rgb_frame.shape[:2]
        results = _pipeline()(
            Image.fromarray(rgb_frame),
            candidate_labels=["basketball rim", "basketball hoop"],
            threshold=0.22,
        )
        candidates: list[RimDetection] = []
        for item in results:
            if float(item["score"]) < 0.30:
                continue
            box = item["box"]
            xyxy = (
                float(box["xmin"]),
                float(box["ymin"]),
                float(box["xmax"]),
                float(box["ymax"]),
            )
            if _plausible_rim(xyxy, width, height):
                candidates.append(RimDetection(confidence=float(item["score"]), xyxy=xyxy))
        return max(candidates, key=lambda item: item.confidence, default=None)


def _plausible_rim(
    xyxy: tuple[float, float, float, float], frame_width: int, frame_height: int
) -> bool:
    x1, y1, x2, y2 = xyxy
    width, height = x2 - x1, y2 - y1
    if width <= 3 or height <= 2:
        return False
    area_ratio = (width * height) / max(frame_width * frame_height, 1)
    aspect_ratio = width / max(height, 1)
    touches_most_frame = width > frame_width * 0.7 or height > frame_height * 0.7
    return 0.00015 <= area_ratio <= 0.08 and 0.6 <= aspect_ratio <= 6.0 and not touches_most_frame
