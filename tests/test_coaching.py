from swisher.analysis import _coaching_feedback
from swisher.contracts import ReleaseEvent, ShotMetrics


def test_abstains_from_coaching_when_analysis_is_low_quality() -> None:
    feedback = _coaching_feedback("abstained", ShotMetrics(), None, ["סרטון מטושטש"])
    assert feedback.available is False
    assert feedback.focus == "איכות הצילום"


def test_prioritizes_balance_when_consistency_is_low() -> None:
    feedback = _coaching_feedback(
        "ok",
        ShotMetrics(pose_consistency=0.3, min_knee_angle_deg=160),
        None,
        [],
    )
    assert feedback.available is False  # no release; no personalized technical claim


def test_follow_through_feedback_requires_reliable_release() -> None:
    release = ReleaseEvent(
        frame=20,
        seconds=0.5,
        confidence=0.75,
        shooting_side="right",
        wrist_ball_distance_norm=1.6,
    )
    feedback = _coaching_feedback(
        "ok",
        ShotMetrics(pose_consistency=0.9, min_knee_angle_deg=120, follow_through_hold_seconds=0.1),
        release,
        [],
    )
    assert feedback.focus == "Follow-through"
    assert feedback.confidence == 0  # no calibrated probability for coaching benefit
