"""Local, evidence-bound training. Baselines describe motion, never ideal technique."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import median

from swisher.studio.feedback import measure
from swisher.studio.store import now


def register(store, payload):
    player = str(payload.get("player", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", player):
        raise ValueError("השתמשו בכינוי שחקן באנגלית או במספרים, עד 40 תווים")
    if payload.get("permission") is not True or payload.get("minor") not in (True, False):
        raise ValueError("נדרש אישור שימוש וציון קבוצת גיל")
    if payload["minor"] and payload.get("guardian_consent") is not True:
        raise ValueError("לקטין נדרש אישור הורה מתועד מחוץ למערכת")
    reference = str(payload.get("consent_reference", "")).strip()
    if not reference or len(reference) > 100:
        raise ValueError("נדרש מזהה אישור; אין להזין שם או פרטי קשר")
    with store.lock:
        profiles = store.data.setdefault("players", {})
        profiles[player] = {
            "player": player,
            "minor": payload["minor"],
            "permission": True,
            "guardian_consent": payload.get("guardian_consent") is True,
            "consent_reference": reference,
            "updated_at": now(),
        }
        store.save()
    return profiles[player]


def require_permission(store, player):
    p = store.snapshot().get("players", {}).get(player)
    if not p or not p["permission"] or (p["minor"] and not p["guardian_consent"]):
        raise ValueError("יש ליצור פרופיל עם אישור שימוש לפני העלאת וידאו")


def dashboard(store, player, view):
    state = store.snapshot()
    jobs = {
        j["id"]: j for j in state["jobs"] if j.get("player") == player and j.get("view") == view
    }
    observations, approved = [], []
    for shot in state["shots"]:
        job = jobs.get(shot["job_id"])
        if not job or not shot.get("reviews"):
            continue
        review = shot["reviews"][-1]
        if review["decision"] not in ("approved", "corrected"):
            continue
        approved.append(
            {
                "shot_id": shot["id"],
                "text": review["note"],
                "coach": review["coach"],
                "outcome": review["outcome"],
            }
        )
        report = job.get("report")
        trace = Path(report).parent / "pose.jsonl" if report else None
        if not trace or not trace.exists():
            continue
        candidate = {
            **shot["candidate"],
            "shot_start": review["start_frame"],
            "shot_end": review["end_frame"],
            "release_frame": review["release_frame"],
        }
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        features = measure(candidate, rows, job["video"]["fps"])
        if features.get("coverage", 0) < 0.8 or review["release_frame"] is None:
            continue
        # Side and camera are part of the baseline identity, not pooled silently.
        observations.append({"shot_id": shot["id"], "job_id": job["id"], "features": features})
    baselines = []
    for key, unit in [("knee_min", "deg"), ("trunk_range", "deg"), ("hold_seconds", "s")]:
        groups = {}
        for o in observations:
            f = o["features"]
            side = (
                f.get("knee_side")
                if key == "knee_min"
                else f.get("arm_side")
                if key == "hold_seconds"
                else "both"
            )
            # Separate capture setups until repeatable calibration exists.
            group = (o["job_id"], side)
            if key in f and math.isfinite(f[key]):
                groups.setdefault(group, []).append(o)
        for (job_id, side), group in groups.items():
            values = [o["features"][key] for o in group]
            center = median(values)
            baselines.append(
                {
                    "metric": key,
                    "unit": unit,
                    "side": side,
                    "job_id": job_id,
                    "count": len(values),
                    "median": center,
                    "mad": median([abs(v - center) for v in values]),
                    "ready": len(values) >= 5,
                    "source_shots": [o["shot_id"] for o in group],
                    "confidence": {
                        "coverage": min(o["features"]["coverage"] for o in group),
                        "calibrated_probability": None,
                    },
                }
            )
    known = [a for a in approved if a["outcome"] in ("made", "missed")]
    return {
        "player": player,
        "view": view,
        "baseline": baselines,
        "approved_cues": approved,
        "reviewed_shots": len(approved),
        "labeled_outcomes": len(known),
        "made": sum(a["outcome"] == "made" for a in known),
        "limitation": "פרופיל תיאורי לפי מצלמה וצד גוף; אינו מודל שאומן ואינו טכניקה אידאלית.",
        "quality": {
            "reviewed_measurements": len(observations),
            "accuracy_validated": False,
            "confidence_semantics": "visible-frame coverage; not probability",
        },
    }


def respond(state, text):
    text = str(text).strip()[:300]
    if text in ("התחל", "start", "התחל דריל"):
        return "נתחיל סט של 5 זריקות מהמקום שסיכמת עם המאמן. אחרי כל זריקה אמור ״זריקה״. אמור ״עצור״ בכל רגע."
    if text in ("עצור", "stop"):
        return "האימון נעצר. קח הפסקה לפני הסט הבא."
    if text in ("משוב", "feedback", "מה לשפר"):
        cues = state["approved_cues"]
        return (
            ("הערת המאמן האחרונה: " + cues[-1]["text"])
            if cues
            else "אין עדיין משוב שאושר על ידי מאמן. העלה סרטון והעבר אותו לבדיקה."
        )
    if text in ("פרופיל", "profile"):
        return f"יש {state['reviewed_shots']} זריקות שנבדקו. הפרופיל מתאר עקביות בלבד; אין עדיין אימות דיוק מקצועי."
    return "אני תומך כרגע בפקודות: התחל, עצור, משוב ופרופיל. לא אסיק אבחנה חדשה משיחה."


def erase_local_workspace(store, confirmation):
    """Erase the dedicated local workspace only; never follow imported source paths."""
    import shutil

    if confirmation != "DELETE LOCAL DATA":
        raise ValueError("נדרש אישור מחיקת כל הנתונים המקומיים")
    with store.lock:
        if any(
            j["status"] in ("queued", "processing", "uploading") for j in store.data["jobs"]
        ) or any(s.get("media_status") in ("queued", "processing") for s in store.data["shots"]):
            raise ValueError("יש להמתין לסיום העיבוד לפני מחיקה")
        for child in store.root.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        store.data = {"version": 1, "jobs": [], "shots": [], "players": {}}
        store.save()
    return {"deleted": True, "scope": "local workspace only; external exports and backups excluded"}
