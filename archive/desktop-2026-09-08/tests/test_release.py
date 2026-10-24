import numpy as np

from swisher.analysis import _release_event


def _pose(frame: int) -> dict[str, object]:
    return {
        "index": frame,
        "valid": True,
        "visibility_by_name": {k: 0.95 for k in ("ls", "rs", "lw", "rw")},
        "xy": {
            "ls": np.array([80.0, 100.0]),
            "rs": np.array([120.0, 100.0]),
            "lw": np.array([100.0, 70.0]),
            "rw": np.array([120.0, 70.0]),
        },
    }


def test_release_detected_when_ball_separates_from_wrist_upward() -> None:
    poses = [_pose(0), _pose(3), _pose(6)]
    balls = [
        {"frame": 0, "center": np.array([120.0, 70.0]), "confidence": 0.9},
        {"frame": 3, "center": np.array([130.0, 60.0]), "confidence": 0.9},
        {"frame": 6, "center": np.array([180.0, 40.0]), "confidence": 0.9},
    ]
    release = _release_event(poses, balls, fps=60)
    assert release is not None
    assert release.frame == 6
    assert release.shooting_side == "right"
