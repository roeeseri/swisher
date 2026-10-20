from __future__ import annotations

import json
import math
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from swisher.session import SessionStore


def now():
    return datetime.now(UTC).isoformat()


class StudioStore:
    """Single-process persistent queue; immutable suggestions and versioned coach decisions."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "workspace.json"
        self.lock = threading.RLock()
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"version": 1, "jobs": [], "shots": []}
        )
        for job in self.data["jobs"]:
            if job["status"] in ("processing", "uploading"):
                job["status"] = "queued"
        for shot in self.data["shots"]:
            if shot["media_status"] == "processing":
                shot["media_status"] = "queued"
            if shot.get("visual_version") != 2:
                shot["media_status"] = "queued"
        self.save()

    def save(self):
        with self.lock:
            SessionStore._atomic_json(self.path, self.data)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.data))

    def add_job(self, **fields):
        with self.lock:
            job = {
                "id": uuid4().hex,
                "status": "queued",
                "progress": 0,
                "created_at": now(),
                "error": None,
                **fields,
            }
            self.data["jobs"].append(job)
            self.save()
            return dict(job)

    def shot(self, identifier):
        return next((s for s in self.data["shots"] if s["id"] == identifier), None)

    def review(self, identifier, payload):
        with self.lock:
            shot = self.shot(identifier)
            if shot is None:
                raise ValueError("הזריקה לא נמצאה")
            if payload.get("version") != len(shot["reviews"]):
                raise ValueError("הפריט השתנה בחלון אחר. רעננו לפני שמירה")
            decision = payload.get("decision")
            if decision not in ("approved", "corrected", "not_shot", "uncertain"):
                raise ValueError("יש לבחור החלטה")
            coach = str(payload.get("coach", "")).strip()
            note = str(payload.get("note", "")).strip()
            if not coach or len(coach) > 80:
                raise ValueError("נדרש מזהה מאמן עד 80 תווים")
            if len(note) > 3000:
                raise ValueError("הערה ארוכה מדי")
            if decision == "approved" and not shot["suggestion"]["available"]:
                raise ValueError("אין הערה מוצעת לאישור. כתבו הערה משלכם")
            if decision == "corrected" and not note:
                raise ValueError("יש לכתוב את הערת המאמן")
            job = next(j for j in self.data["jobs"] if j["id"] == shot["job_id"])
            original = shot["candidate"]
            start = int(payload.get("start_frame", original["shot_start"]))
            end = int(payload.get("end_frame", original["shot_end"]))
            release = payload.get("release_frame", original.get("release_frame"))
            release = None if release is None else int(release)
            if not 0 <= start < end < job["video"]["frame_count"]:
                raise ValueError("טווח הזריקה מחוץ לסרטון")
            if release is not None and not start <= release <= end:
                raise ValueError("פריים השחרור צריך להיות בתוך הזריקה")
            outcome = payload.get("outcome", "unknown")
            if outcome not in ("made", "missed", "unknown"):
                raise ValueError("תוצאת זריקה לא תקינה")
            review = {
                "version": len(shot["reviews"]) + 1,
                "coach": coach,
                "decision": decision,
                "note": shot["suggestion"]["text"] if decision == "approved" else note,
                "start_frame": start,
                "end_frame": end,
                "release_frame": release,
                "outcome": outcome,
                "created_at": now(),
            }
            shot["reviews"].append(review)
            previous = (
                shot["reviews"][-2]
                if len(shot["reviews"]) > 1
                else {
                    "start_frame": original["shot_start"],
                    "end_frame": original["shot_end"],
                    "release_frame": original.get("release_frame"),
                }
            )
            if any(
                review[key] != previous[key]
                for key in ("start_frame", "end_frame", "release_frame")
            ):
                shot["media_status"] = "queued"
            self.save()
            return review


def suggestion(candidate, measurement):
    """Descriptive vision evidence only; never claim a trained coaching model."""
    base = {
        "available": False,
        "text": "אין מספיק ראיות להערה אוטומטית. מה המאמן רואה?",
        "method": "vision_measurement_rules_v1",
        "evidence": [],
        "limitations": "היטל דו־ממדי, תלוי זווית צילום. ההצעה אינה אבחון או הוכחת סיבה להחטאה.",
    }
    if not measurement or measurement.get("tracking_review_failed"):
        return base
    if measurement.get("valid_frame_coverage", 0) < 0.7 or candidate.get("release_frame") is None:
        return base
    metrics = measurement.get("metrics") or {}
    low, high = metrics.get("min_knee_angle_deg"), metrics.get("max_knee_angle_deg")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (low, high)):
        return base
    if not 0 <= low <= high <= 180:
        return base
    return {
        **base,
        "available": True,
        "text": "בדקו את רצף כיפוף ויישור הברכיים בזריקה ואת התיאום שלו עם עליית הידיים.",
        "evidence": [
            f"טווח זווית ברך נצפה: {low:.0f}°–{high:.0f}° (אחוזונים 5–95).",
            "שני צדי הגוף עשויים להשתתף במדידה; זו הצעה לבדיקה חזותית.",
        ],
    }
