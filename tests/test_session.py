from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from swisher.session import SessionStore, ShotAnnotation, extract_clip, inspect_video


def synthetic_video(path: Path, frames: int = 12) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((48, 64, 3), index * 10, dtype=np.uint8)
        writer.write(image)
    writer.release()


def test_session_round_trip_and_annotation_versioning() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video = root / "pilot.mp4"
        synthetic_video(video)
        store = SessionStore(root / "sessions")
        metadata = store.create(
            session_id="session_001",
            player_id="player_001",
            coach_id="coach_001",
            phase="baseline",
            camera_view="side_oblique",
            consent_confirmed=True,
            video_name=video.name,
            video_bytes=video.read_bytes(),
        )
        assert metadata.frame_count == 12
        annotation = ShotAnnotation(shot_id="shot_001", shot_start=1, release_frame=3, shot_end=8)
        store.upsert_annotation(metadata.session_id, annotation)
        store.upsert_annotation(
            metadata.session_id, annotation.model_copy(update={"drill": "form"})
        )
        records = store.annotations(metadata.session_id)
        assert len(records) == 1
        assert records[0].annotation_version == 2
        assert records[0].drill == "form"


def test_extract_clip_preserves_expected_frame_count() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.mp4"
        destination = root / "clip.mp4"
        synthetic_video(source, frames=15)
        extract_clip(source, destination, 3, 8)
        assert inspect_video(destination)["frame_count"] == 6
