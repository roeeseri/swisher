from types import SimpleNamespace

import numpy as np
import pytest

from swisher.silhouette import compare
from swisher.studio.store import StudioStore
from swisher.studio.training import dashboard, register, require_permission, respond


def test_minor_cannot_upload_without_guardian(tmp_path):
    store = StudioStore(tmp_path)
    with pytest.raises(ValueError):
        register(
            store, {"player": "p1", "minor": True, "permission": True, "consent_reference": "ref"}
        )
    with pytest.raises(ValueError):
        require_permission(store, "p1")
    register(
        store,
        {
            "player": "p1",
            "minor": True,
            "permission": True,
            "guardian_consent": True,
            "consent_reference": "ref",
        },
    )
    require_permission(store, "p1")
    assert StudioStore(tmp_path).snapshot()["players"]["p1"]["guardian_consent"]


def test_empty_profile_never_invents_cue(tmp_path):
    state = dashboard(StudioStore(tmp_path), "p1", "left")
    assert state["baseline"] == []
    assert "אין עדיין" in respond(state, "משוב")
    assert "לא אסיק" in respond(state, "למה החטאתי?")
    assert state["quality"]["accuracy_validated"] is False


def test_only_latest_approved_review_same_player_view(tmp_path):
    store = StudioStore(tmp_path)
    j = store.add_job(player="p1", view="left")
    store.data["shots"] = [
        {
            "id": "s",
            "job_id": j["id"],
            "reviews": [{"decision": "corrected", "note": "cue", "coach": "c", "outcome": "made"}],
        }
    ]
    assert dashboard(store, "p2", "left")["approved_cues"] == []
    assert dashboard(store, "p1", "right")["approved_cues"] == []
    assert respond(dashboard(store, "p1", "left"), "משוב").endswith("cue")
    store.data["shots"][0]["reviews"].append({"decision": "uncertain"})
    assert dashboard(store, "p1", "left")["approved_cues"] == []


def test_silhouette_is_measurement_not_accuracy():
    points = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9)] * 10
    result = compare(np.ones((20, 20)), points)
    assert result["visible_landmarks_inside_mask"] == 1
    assert result["calibrated_probability"] is None
    assert "not accuracy" in result["limitation"]
    assert not compare(np.zeros((20, 20)), points)["available"]
    assert not compare(np.full((20, 20), float("nan")), points)["available"]


def test_erase_stays_inside_workspace_and_refuses_active_jobs(tmp_path):
    from swisher.studio.training import erase_local_workspace

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"private")
    store = StudioStore(tmp_path / "workspace")
    j = store.add_job(player="p1", view="left", source=str(outside))
    with pytest.raises(ValueError):
        erase_local_workspace(store, "DELETE LOCAL DATA")
    j = store.data["jobs"][0]
    j["status"] = "ready"
    (store.root / "uploads").mkdir()
    (store.root / "uploads" / "clip.mp4").write_bytes(b"private")
    (store.root / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        erase_local_workspace(store, "yes")
    erase_local_workspace(store, "DELETE LOCAL DATA")
    assert outside.read_bytes() == b"private"
    assert not (store.root / "uploads").exists()
    assert StudioStore(store.root).snapshot()["jobs"] == []


def test_baseline_uses_reviewed_boundaries_and_keeps_capture_separate(tmp_path, monkeypatch):
    from swisher.studio import training

    store = StudioStore(tmp_path / "workspace")
    folder = tmp_path / "analysis"
    folder.mkdir()
    (folder / "pose.jsonl").write_text("{}\n")
    calls = []

    def measured(candidate, rows, fps):
        calls.append(candidate)
        return {"coverage": 0.9, "knee_min": 100 + candidate["shot_start"], "knee_side": "l"}

    monkeypatch.setattr(training, "measure", measured)
    for job_index in range(2):
        job = store.add_job(
            player="p1", view="left", report=str(folder / "report.json"), video={"fps": 30}
        )
        for i in range(5 if job_index == 0 else 2):
            store.data["shots"].append(
                {
                    "id": f"{job_index}-{i}",
                    "job_id": job["id"],
                    "candidate": {"shot_start": 0, "shot_end": 50, "release_frame": 20},
                    "reviews": [
                        {
                            "decision": "corrected",
                            "note": "reviewed",
                            "coach": "c",
                            "outcome": "unknown",
                            "start_frame": i + 1,
                            "end_frame": 49,
                            "release_frame": 21,
                        }
                    ],
                }
            )
    state = dashboard(store, "p1", "left")
    assert len(state["baseline"]) == 2
    assert [b["ready"] for b in state["baseline"]] == [True, False]
    assert state["baseline"][0]["median"] == 103
    assert all(c["release_frame"] == 21 and c["shot_start"] > 0 for c in calls)
    assert state["labeled_outcomes"] == 0
