from __future__ import annotations

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class EventWindow(BaseModel):
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    confidence: float = Field(ge=0, le=1)
    method: str = "wrist_motion_v0.1"


class ShotMetrics(BaseModel):
    min_elbow_angle_deg: float | None = None
    max_elbow_angle_deg: float | None = None
    min_knee_angle_deg: float | None = None
    max_knee_angle_deg: float | None = None
    trunk_lean_range_deg: float | None = None
    pose_consistency: float | None = Field(default=None, ge=0, le=1)
    release_elbow_angle_deg: float | None = None
    release_knee_angle_deg: float | None = None
    release_wrist_angle_deg: float | None = None
    release_height_body_ratio: float | None = None
    jump_height_body_ratio: float | None = None
    landing_drift_shoulder_widths: float | None = None
    follow_through_hold_seconds: float | None = None
    source: str = "mediapipe_pose_v0.1"


class DetectionSummary(BaseModel):
    detector: str
    sampled_frames: int
    player_detected_frames: int
    ball_detected_frames: int
    player_detection_rate: float = Field(ge=0, le=1)
    ball_detection_rate: float = Field(ge=0, le=1)
    rim_detected: bool
    rim_confidence: float = Field(ge=0, le=1)
    rim_box: tuple[float, float, float, float] | None = None


class QualitySummary(BaseModel):
    mean_brightness: float
    mean_blur_score: float
    warnings: list[str]


class ReleaseEvent(BaseModel):
    frame: int
    seconds: float
    confidence: float = Field(ge=0, le=1)
    shooting_side: str
    wrist_ball_distance_norm: float
    method: str = "ball_wrist_separation_v0.1"


class ShotOutcome(BaseModel):
    status: str
    confidence: float = Field(ge=0, le=1)
    reason: str


class CoachingFeedback(BaseModel):
    available: bool
    focus: str
    observation: str
    cue: str
    drill: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]


class AnalysisResult(BaseModel):
    status: str
    message: str
    video: VideoMetadata
    pose_confidence: float = Field(ge=0, le=1)
    valid_pose_frames: int
    event_window: EventWindow | None
    release: ReleaseEvent | None
    outcome: ShotOutcome
    metrics: ShotMetrics
    detections: DetectionSummary
    quality: QualitySummary
    coaching: CoachingFeedback
    limitations: list[str]
