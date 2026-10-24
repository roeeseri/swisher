from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ConsentStatus(StrEnum):
    APPROVED = "approved"
    PENDING = "pending"
    REVOKED = "revoked"


class QualityDecision(StrEnum):
    PASS = "pass"
    WARN = "warn"
    REJECT = "reject"


class Outcome(StrEnum):
    MAKE = "make"
    MISS = "miss"
    UNKNOWN = "unknown"


class Phase(StrEnum):
    BASELINE = "baseline"
    INTERVENTION = "intervention"
    POST = "post_intervention"


class CameraView(StrEnum):
    SIDE_OBLIQUE = "side_oblique"
    FRONT_OBLIQUE = "front_oblique"
    SIDE = "side"
    FRONT = "front"
    OTHER = "other"


class VideoMetadata(StrictModel):
    path: str
    container: str | None = None
    codec: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    frame_count: int | None = Field(default=None, ge=0)
    rotation_degrees: int = 0
    time_base: str | None = None
    captured_at: datetime | None = None


class QualityIssue(StrictModel):
    code: str
    message: str
    decision: QualityDecision
    observed: str | float | int | None = None
    required: str | float | int | None = None


class QualityReport(StrictModel):
    decision: QualityDecision
    issues: list[QualityIssue] = Field(default_factory=list)
    inspected_at: datetime = Field(default_factory=utc_now)
    inspector_version: str = "video-quality-v0.1"


class SessionManifest(StrictModel):
    schema_version: str = "1.0.0"
    session_id: str = Field(min_length=3)
    player_id: str = Field(min_length=3)
    video_id: str = Field(min_length=3)
    coach_id: str | None = None
    camera_view: CameraView
    drill: str
    phase: Phase
    consent_status: ConsentStatus
    storage_uri: str
    annotation_uri: str | None = None
    age_band: str | None = None
    height_cm: float | None = Field(default=None, gt=0)
    dominant_hand: str | None = None
    experience_band: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def approved_before_analysis(self) -> SessionManifest:
        if self.consent_status != ConsentStatus.APPROVED:
            raise ValueError("analysis requires approved consent")
        return self


class FrameRange(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> FrameRange:
        if self.end < self.start:
            raise ValueError("end frame must not precede start frame")
        return self


class Measurement(StrictModel):
    name: str
    value: float | int | str | bool | None
    unit: str
    confidence: float = Field(ge=0, le=1)
    method: str
    method_version: str
    source_frames: FrameRange
    quality_flags: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)


class ShotEvents(StrictModel):
    shot_start: int = Field(ge=0)
    load: int | None = Field(default=None, ge=0)
    release: int | None = Field(default=None, ge=0)
    rim_interaction: int | None = Field(default=None, ge=0)
    shot_end: int = Field(ge=0)

    @model_validator(mode="after")
    def chronological(self) -> ShotEvents:
        values = [
            self.shot_start,
            self.load,
            self.release,
            self.rim_interaction,
            self.shot_end,
        ]
        known = [value for value in values if value is not None]
        if known != sorted(known):
            raise ValueError("shot events must be chronological")
        return self


class ShotAnnotation(StrictModel):
    schema_version: str = "1.0.0"
    annotation_id: str
    session_id: str
    video_id: str
    shot_id: str
    annotator_id: str
    annotation_version: int = Field(default=1, ge=1)
    events: ShotEvents
    outcome: Outcome
    diagnosis_code: str | None = None
    evidence: str | None = None
    coach_cue: str | None = None
    drill_name: str | None = None
    phase: Phase
    confidence: int = Field(ge=1, le=5)
    notes: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Keypoint(StrictModel):
    name: str
    x: float
    y: float
    z: float | None = None
    confidence: float = Field(ge=0, le=1)
    visible: bool


class BoundingBox(StrictModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = Field(ge=0, le=1)


class PoseFrame(StrictModel):
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    keypoints: list[Keypoint]
    bounding_box: BoundingBox | None = None
    estimator: str
    estimator_version: str
    source_video: str


def dump_json(model: BaseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def jsonable(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
