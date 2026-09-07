import numpy as np
import pytest

from swisher.analysis import _coaching_feedback, _release_event
from swisher.contracts import ShotMetrics
from swisher.proposals import propose_shots
from swisher.session import SessionStore, ShotAnnotation, extract_clip


def pose(index, raised=True, visible=True):
    return {
        "index": index,
        "valid": visible,
        "xy": {
            "ls": np.array([80.0, 100.0]),
            "rs": np.array([120.0, 100.0]),
            "le": np.array([80.0, 80.0]),
            "re": np.array([120.0, 80.0]),
            "lw": np.array([80.0, 60.0 if raised else 160.0]),
            "rw": np.array([120.0, 60.0 if raised else 160.0]),
            "lh": np.array([80.0, 180.0]),
            "rh": np.array([120.0, 180.0]),
        },
        "visibility_by_name": {k: 0.9 for k in ("ls", "rs", "le", "re", "lw", "rw", "lh", "rh")},
    }


def test_unknown_release_round_trip():
    item = ShotAnnotation(shot_id="shot_001", shot_start=0, shot_end=100)
    assert ShotAnnotation.model_validate_json(item.model_dump_json()).release_frame is None
    assert item.outcome == "unknown"
    assert not item.coach_approved


def test_proposals_keep_unknown_release_and_multiple_episodes():
    frames = [pose(i, 10 <= i <= 25 or 70 <= i <= 85) for i in range(110)]
    proposals = propose_shots(frames, [], 30, 110, _release_event)
    assert len(proposals) == 2
    assert all(p["release_frame"] is None and p["review_required"] for p in proposals)
    assert proposals[0]["shot_end"] < proposals[1]["shot_start"]


def test_occlusion_not_invented():
    assert not propose_shots(
        [pose(i, visible=False) for i in range(60)], [], 30, 60, _release_event
    )


def test_long_gap_not_release():
    frames = [pose(i) for i in (0, 100, 200)]
    balls = [
        {"frame": i, "center": np.array([120.0 + i, 60.0 - i * 0.1]), "confidence": 0.9}
        for i in (0, 100, 200)
    ]
    assert _release_event(frames, balls, 30) is None


def test_hidden_wrists_not_release():
    frames = [pose(i) for i in (0, 3, 6)]
    for frame in frames:
        frame["visibility_by_name"]["lw"] = 0.1
    balls = [
        {"frame": i, "center": np.array([120.0 + i * 10, 60.0 - i]), "confidence": 0.9}
        for i in (0, 3, 6)
    ]
    assert _release_event(frames, balls, 60) is None


def test_unknown_visibility_does_not_become_coaching():
    feedback = _coaching_feedback("ok", ShotMetrics(min_knee_angle_deg=170), None, [])
    assert not feedback.available


def test_traversal_and_consent_before_storage(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with pytest.raises(ValueError):
        store.load("../outside")
    with pytest.raises(ValueError):
        store.artifact_directory("session_001", "../outside")
    with pytest.raises(ValueError):
        store.create(
            session_id="session_001",
            player_id="player_001",
            coach_id="coach_001",
            phase="baseline",
            camera_view="side",
            consent_confirmed=False,
            video_name="source.mp4",
            video_bytes=b"not a video",
        )
    assert list(store.root.iterdir()) == []


def test_never_overwrite_original(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"original")
    with pytest.raises(ValueError):
        extract_clip(source, source, 0, 20)
    assert source.read_bytes() == b"original"


def test_existing_incomplete_directory_is_preserved(tmp_path):
    store = SessionStore(tmp_path)
    (tmp_path / "session_001").mkdir()
    marker = tmp_path / "session_001" / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        store.create(
            session_id="session_001",
            player_id="player_001",
            coach_id="coach_001",
            phase="baseline",
            camera_view="side",
            consent_confirmed=True,
            video_name="source.mp4",
            video_bytes=b"invalid",
        )
    assert marker.read_text() == "keep"
