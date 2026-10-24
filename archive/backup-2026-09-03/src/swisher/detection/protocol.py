from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from swisher.video.frames import VideoFrame


class Detection(BaseModel):
    label: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = Field(ge=0, le=1)
    frame_index: int
    timestamp_seconds: float


class ObjectDetector(Protocol):
    name: str
    version: str

    def detect(self, frame: VideoFrame) -> list[Detection]:
        ...
