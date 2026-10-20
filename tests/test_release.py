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


def test_visible_wrist_can_bound_release_without_inventing_shooting_hand():
    poses = [_pose(0), _pose(3), _pose(6)]
    for pose in poses:
        pose["visibility_by_name"]["lw"] = 0.01
    balls = [
        {"frame": 0, "center": np.array([120.0, 70.0]), "confidence": 0.9},
        {"frame": 3, "center": np.array([130.0, 60.0]), "confidence": 0.9},
        {"frame": 6, "center": np.array([180.0, 40.0]), "confidence": 0.9},
    ]
    release = _release_event(poses, balls, 60)
    assert release is not None
    assert release.shooting_side == "unknown"


def test_background_ball_does_not_hide_release_or_create_a_distant_switch():
    poses = [_pose(0), _pose(3), _pose(6)]
    balls = []
    for frame, center in [(0, [120.0, 70.0]), (3, [130.0, 60.0]), (6, [180.0, 40.0])]:
        balls += [
            {"frame": frame, "center": np.array(center), "confidence": 0.8},
            {"frame": frame, "center": np.array([900.0, 30.0]), "confidence": 0.99},
        ]
    assert _release_event(poses, balls, 60) is not None
    assert _release_event(poses, [b for b in balls if b["confidence"] == 0.99], 60) is None


def test_no_hand_specific_metrics_when_shooting_hand_is_unknown():
    from swisher.analysis import _release_metrics
    from swisher.contracts import ReleaseEvent

    release = ReleaseEvent(
        frame=3, seconds=0.05, confidence=0.8, shooting_side="unknown", wrist_ball_distance_norm=1.0
    )
    assert _release_metrics([_pose(0), _pose(3), _pose(6)], release, None, 60) == (None,) * 7


def test_ball_still_near_hand_does_not_become_release():
    poses = [_pose(0), _pose(3), _pose(6)]
    balls = [
        {"frame": i, "center": np.array([120.0 + i, 68.0 - i]), "confidence": 0.9}
        for i in [0, 3, 6]
    ]
    assert _release_event(poses, balls, 60) is None
