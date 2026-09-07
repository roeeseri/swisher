from pathlib import Path

from streamlit.testing.v1 import AppTest
from test_session import synthetic_video

from swisher.contracts import AnalysisResult
from swisher.session import SessionStore

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_empty_studio_loads_without_camera_or_video(monkeypatch, tmp_path):
    store = SessionStore(tmp_path / "sessions")
    monkeypatch.setattr("swisher.session.SessionStore", lambda root: store)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 3
    assert app.button[0].disabled


def test_manual_annotation_and_reopen(monkeypatch, tmp_path):
    store = SessionStore(tmp_path / "sessions")
    source = tmp_path / "source.mp4"
    synthetic_video(source)
    store.create(
        session_id="session_test",
        player_id="player_test",
        coach_id="coach_test",
        phase="baseline",
        camera_view="side",
        consent_confirmed=True,
        video_name="source.mp4",
        video_bytes=source.read_bytes(),
    )
    monkeypatch.setattr("swisher.session.SessionStore", lambda root: store)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    button = next(b for b in app.button if b.label == "אישור ושמירת הזריקה")
    button.click().run(timeout=30)
    assert not app.exception
    annotations = store.annotations("session_test")
    assert len(annotations) == 1
    assert annotations[0].release_frame is None
    assert annotations[0].outcome == "unknown"
    selection = next(s for s in app.selectbox if s.label == "הצעה לבדיקה או זריקה לעריכה")
    selection.select("edit:manual_001").run(timeout=30)
    assert not app.exception

    def fake_analysis(source_path, output_path, on_progress=None, **kwargs):
        synthetic_video(output_path)
        return AnalysisResult(
            status="abstained",
            message="synthetic integration test; not real inference",
            video={
                "width": 64,
                "height": 48,
                "fps": 30,
                "frame_count": 12,
                "duration_seconds": 0.4,
            },
            pose_confidence=0,
            valid_pose_frames=0,
            event_window=None,
            release=None,
            outcome={"status": "unknown", "confidence": 0, "reason": "synthetic"},
            metrics={},
            detections={
                "detector": "fake_test_only",
                "sampled_frames": 4,
                "player_detected_frames": 0,
                "ball_detected_frames": 0,
                "player_detection_rate": 0,
                "ball_detection_rate": 0,
                "rim_detected": False,
                "rim_confidence": 0,
            },
            quality={"mean_brightness": 100, "mean_blur_score": 100, "warnings": []},
            coaching={
                "available": False,
                "focus": "test",
                "observation": "no evidence",
                "cue": "",
                "drill": "",
                "confidence": 0,
                "evidence": [],
            },
            limitations=[],
        )

    monkeypatch.setattr("swisher.analysis.analyze_video", fake_analysis)
    next(b for b in app.button if b.label == "זהה גוף והצע תיוגים").click().run(timeout=30)
    assert not app.exception
    assert (store.directory("session_test") / "latest_scan.json").exists()
    next(b for b in app.button if b.label == "נתח את הזריקה המאושרת").click().run(timeout=30)
    assert not app.exception
    assert list((store.directory("session_test") / "artifacts").glob("*/v1/*/report.json"))
