from __future__ import annotations

import math

from swisher.studio.feedback import generate, measure, propose


def poses(drop_after=12):
    result = []
    for index in range(90):
        theta = math.radians(100 + 65 * abs(index - 25) / 65)
        points = {
            "ls": [50, 20],
            "rs": [70, 20],
            "lh": [50, 60],
            "rh": [70, 60],
            "lk": [50, 100],
            "la": [50 + 40 * math.sin(theta), 100 - 40 * math.cos(theta)],
            "rk": [70, 100],
            "ra": [75, 140],
            "le": [50, 0],
            "lw": [50, -20],
            "re": [80, 40],
            "rw": [90, 65],
        }
        if index > 30 + drop_after:
            points.update(le=[70, 40], lw=[70, 65])
        visibility = {n: 0.95 for n in points}
        visibility["re"] = visibility["rw"] = 0
        result.append(
            {
                "index": index,
                "valid": True,
                "camera_tracking_valid": True,
                "xy": points,
                "visibility_by_name": visibility,
            }
        )
    return result


CANDIDATE = {
    "shot_id": "shot_001",
    "shot_start": 0,
    "shot_end": 89,
    "release_frame": 30,
    "motion_frame": 28,
}


def test_follow_through_uses_actual_contiguous_post_release_frames():
    f = measure(CANDIDATE, poses(drop_after=4), 30)
    assert abs(f["hold_seconds"] - 4 / 30) < 1e-8
    assert f["hold_ended"] and f["hold_frames"] == [31, 34]
    note = propose(CANDIDATE, f)
    assert note["focus"] == "follow_through"
    assert "0.13" in note["text"] and "המשוער" in note["text"]
    assert note["requires_coach_review"]


def test_visibility_gap_never_becomes_a_hand_drop_diagnosis():
    rows = poses(drop_after=4)
    rows[33]["valid"] = False
    f = measure(CANDIDATE, rows, 30)
    assert "hold_seconds" not in f
    assert propose(CANDIDATE, f).get("focus") != "follow_through"


def test_no_release_still_allows_visible_motion_but_never_follow_through():
    c = {**CANDIDATE, "release_frame": None}
    note = generate([c], poses(), 30)["shot_001"]
    assert note["available"]
    assert note["focus"] == "knee_motion"
    assert "בקטע המסומן" in note["text"]
    assert note["features"]["release_frame"] is None


def test_low_visibility_abstains_without_inventing_a_note():
    rows = poses()
    for row in rows[:60]:
        row["valid"] = False
    note = generate([CANDIDATE], rows, 30)["shot_001"]
    assert not note["available"]
    assert note["kind"] == "abstention"


def test_large_relative_difference_uses_same_view_reference():
    f = {"coverage": 1, "frames": [0, 89], "knee_min": 145, "knee_range": 30}
    note = propose(CANDIDATE, f, {"knee_min": 110})
    assert note["focus"] == "knee_consistency"
    assert "145" in note["evidence"][1] and "110" in note["evidence"][1]
    assert propose(CANDIDATE, f).get("focus") == "knee_motion"


def test_censored_high_arm_is_reported_as_lower_bound():
    f = measure(CANDIDATE, poses(drop_after=90), 30)
    note = propose(CANDIDATE, f)
    assert not f["hold_ended"]
    assert "לפחות" in note["text"]
