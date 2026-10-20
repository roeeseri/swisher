from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Literal
from uuid import uuid4

import av
import cv2
from pydantic import BaseModel, Field, model_validator

IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,64}$")


class SessionMetadata(BaseModel):
    schema_version: str = "1.0"
    session_id: str
    player_id: str
    coach_id: str
    phase: Literal["baseline", "intervention", "post_intervention"]
    camera_view: str = "side_oblique"
    consent_confirmed: bool
    video_filename: str
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="after")
    def validate_identifiers_and_consent(self) -> SessionMetadata:
        for name, value in {
            "session_id": self.session_id,
            "player_id": self.player_id,
            "coach_id": self.coach_id,
        }.items():
            if not IDENTIFIER_PATTERN.fullmatch(value):
                raise ValueError(
                    f"{name} must contain 3-64 letters, numbers, underscores or dashes"
                )
        if not self.consent_confirmed:
            raise ValueError("recording consent must be confirmed before storage")
        return self


class ShotAnnotation(BaseModel):
    schema_version: str = "1.0"
    annotation_id: str = Field(default_factory=lambda: str(uuid4()))
    shot_id: str
    shot_start: int = Field(ge=0)
    release_frame: int | None = Field(default=None, ge=0)
    rim_interaction: int | None = Field(default=None, ge=0)
    shot_end: int = Field(ge=0)
    outcome: Literal["made", "missed", "unknown"] = "unknown"
    diagnosis: str = "none"
    coach_evidence: str = ""
    coach_cue: str = ""
    drill: str = ""
    confidence: int = Field(default=3, ge=1, le=5)
    annotation_version: int = Field(default=1, ge=1)
    origin: Literal["manual", "auto_reviewed"] = "manual"
    coach_approved: bool = False
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="after")
    def validate_timeline(self) -> ShotAnnotation:
        events = [self.shot_start]
        if self.release_frame is not None:
            events.append(self.release_frame)
        if self.rim_interaction is not None:
            events.append(self.rim_interaction)
        events.append(self.shot_end)
        if events != sorted(events):
            raise ValueError("shot events must be chronological")
        if self.shot_end == self.shot_start:
            raise ValueError("shot must contain more than one frame")
        if not IDENTIFIER_PATTERN.fullmatch(self.shot_id):
            raise ValueError("shot_id contains unsupported characters")
        return self


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, session_id: str) -> Path:
        if not IDENTIFIER_PATTERN.fullmatch(session_id):
            raise ValueError("invalid session identifier")
        path = self.root / session_id
        if path.is_symlink():
            raise ValueError("symlink sessions are not supported")
        return path

    def list_sessions(self) -> list[str]:
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / "session.json").is_file()
        )

    def create(
        self,
        *,
        session_id: str,
        player_id: str,
        coach_id: str,
        phase: str,
        camera_view: str,
        consent_confirmed: bool,
        video_name: str,
        video_bytes: bytes,
    ) -> SessionMetadata:
        for identifier in (session_id, player_id, coach_id):
            if not IDENTIFIER_PATTERN.fullmatch(identifier):
                raise ValueError("מזהים: 3–64 אותיות באנגלית, מספרים, קו תחתון או מקף")
        if not consent_confirmed:
            raise ValueError("נדרשת הסכמה לפני שמירת הסרטון")
        if not video_bytes or len(video_bytes) > 200 * 1024 * 1024:
            raise ValueError("יש להעלות קובץ בגודל עד 200MB")
        directory = self.directory(session_id)
        if directory.exists():
            raise FileExistsError(f"session already exists: {session_id}")
        staging = Path(tempfile.mkdtemp(prefix=".upload_", dir=self.root))
        safe_suffix = Path(video_name).suffix.lower()
        if safe_suffix not in {".mp4", ".mov", ".m4v", ".avi", ".webm"}:
            safe_suffix = ".mp4"
        video_path = staging / f"source{safe_suffix}"
        try:
            video_path.write_bytes(video_bytes)
            if safe_suffix == ".webm":
                converted = staging / "indexed.mp4"
                normalize_browser_video(video_path, converted)
                video_path = converted
            facts = inspect_video(video_path)
            metadata = SessionMetadata(
                session_id=session_id,
                player_id=player_id,
                coach_id=coach_id,
                phase=phase,  # type: ignore[arg-type]
                camera_view=camera_view,
                consent_confirmed=consent_confirmed,
                video_filename=video_path.name,
                **facts,
            )
            self._atomic_json(staging / "session.json", metadata.model_dump(mode="json"))
            (staging / "artifacts").mkdir()
            if directory.exists():
                raise FileExistsError("session already exists")
            staging.rename(directory)
            return metadata
        except Exception:
            shutil.rmtree(staging)
            raise

    def load(self, session_id: str) -> SessionMetadata:
        return SessionMetadata.model_validate_json(
            (self.directory(session_id) / "session.json").read_text(encoding="utf-8")
        )

    def video_path(self, session_id: str) -> Path:
        metadata = self.load(session_id)
        if Path(metadata.video_filename).name != metadata.video_filename:
            raise ValueError("invalid video filename")
        return self.directory(session_id) / metadata.video_filename

    def annotations(self, session_id: str) -> list[ShotAnnotation]:
        path = self.directory(session_id) / "annotations.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [ShotAnnotation.model_validate(item) for item in payload]

    def upsert_annotation(self, session_id: str, annotation: ShotAnnotation) -> None:
        metadata = self.load(session_id)
        if annotation.shot_end >= metadata.frame_count:
            raise ValueError("shot_end exceeds the video frame count")
        annotations = self.annotations(session_id)
        for index, existing in enumerate(annotations):
            if existing.shot_id == annotation.shot_id:
                self._atomic_json(
                    self.directory(session_id)
                    / "history"
                    / f"{existing.shot_id}_v{existing.annotation_version}_{uuid4().hex}.json",
                    existing.model_dump(mode="json"),
                )
                annotation = annotation.model_copy(
                    update={
                        "annotation_id": existing.annotation_id,
                        "annotation_version": existing.annotation_version + 1,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                annotations[index] = annotation
                break
        else:
            annotations.append(annotation)
        annotations.sort(key=lambda item: item.shot_start)
        self._atomic_json(
            self.directory(session_id) / "annotations.json",
            [item.model_dump(mode="json") for item in annotations],
        )

    def artifact_directory(self, session_id: str, shot_id: str) -> Path:
        if not IDENTIFIER_PATTERN.fullmatch(shot_id):
            raise ValueError("invalid shot identifier")
        path = self.directory(session_id) / "artifacts" / shot_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _atomic_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def inspect_video(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("video could not be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError("video metadata is invalid")
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
        }
    finally:
        capture.release()


def normalize_browser_video(
    source: Path, destination: Path, *, max_duration_seconds: float = 125
) -> None:
    """Finalize MediaRecorder WebM into indexed H.264; retain the original WebM.

    Source presentation times are retained, not reconstructed as frame/fps.
    """
    if not 0 < max_duration_seconds <= 600:
        raise ValueError("invalid normalization duration limit")
    with av.open(str(source)) as incoming, av.open(str(destination), "w") as outgoing:
        stream = incoming.streams.video[0]
        output = outgoing.add_stream("libx264", rate=stream.average_rate or 30)
        output.width, output.height = stream.width, stream.height
        output.pix_fmt = "yuv420p"
        output.time_base = Fraction(1, 90000)
        output.options = {"preset": "fast", "crf": "20"}
        first = None
        count = 0
        for frame in incoming.decode(stream):
            if frame.time is None:
                raise ValueError("recording timestamps are missing")
            if first is None:
                first = frame.time
            if frame.time - first > max_duration_seconds:
                raise ValueError("recording exceeds the allowed duration")
            frame.pts = round((frame.time - first) * 90000)
            frame.time_base = output.time_base
            for packet in output.encode(frame):
                outgoing.mux(packet)
            count += 1
        if count < 2:
            raise ValueError("recording contains fewer than two frames")
        for packet in output.encode():
            outgoing.mux(packet)


def read_frame(path: Path, frame_number: int) -> object:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("video could not be opened")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"frame {frame_number} could not be read")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def extract_clip(source: Path, destination: Path, start_frame: int, end_frame: int) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("source video must not be overwritten")
    facts = inspect_video(source)
    if start_frame < 0 or end_frame >= facts["frame_count"]:
        raise ValueError("clip range is outside source video")
    if end_frame <= start_frame:
        raise ValueError("clip end must follow clip start")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("source video could not be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("output clip could not be created")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for _ in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError("source video ended before the requested frame")
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
