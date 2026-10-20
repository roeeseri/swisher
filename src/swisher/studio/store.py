from __future__ import annotations

import copy
import hashlib
import json
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

    def import_feedback(self, payload):
        """Merge registered offline reviews into the live store, preserving every coach/version."""
        with self.lock:
            if not isinstance(payload, dict) or not isinstance(payload.get("shots"), list):
                raise TypeError("מבנה קובץ התיוגים לא תקין")
            registry_path = self.root / "coach_packages.json"
            registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
            package_id = payload.get("package_id")
            if package_id not in registry:
                raise ValueError("החבילה אינה רשומה במערכת הזו")
            registration = registry[package_id]
            current_jobs = {j["id"]: j for j in self.data["jobs"]}
            for identifier, digest in registration["job_hashes"].items():
                if (
                    identifier not in current_jobs
                    or current_jobs[identifier].get("source_sha256") != digest
                ):
                    raise ValueError("סרטוני המקור אינם תואמים לחבילה")
            before = copy.deepcopy(self.data)
            imported = skipped = 0
            try:
                for item in payload.get("shots", []):
                    if not isinstance(item, dict) or not isinstance(item.get("reviews"), list):
                        raise TypeError("רשומת תיוג לא תקינה")
                    identifier = item.get("shot_id")
                    shot = self.shot(identifier)
                    if shot is None or identifier not in registration["candidates"]:
                        raise ValueError("זריקה לא מוכרת בחבילת התיוג")
                    digest = hashlib.sha256(
                        json.dumps(shot["candidate"], sort_keys=True).encode()
                    ).hexdigest()
                    if (
                        digest != registration["candidates"][identifier]
                        or item.get("candidate") != shot["candidate"]
                    ):
                        raise ValueError("פרטי זריקת המקור השתנו")
                    base = registration["base_versions"].get(identifier, 0)
                    for row in item.get("reviews", []):
                        if not isinstance(row, dict):
                            raise TypeError("החלטת מאמן לא תקינה")
                        if row.get("version", 0) <= base:
                            continue
                        external_id = (
                            package_id
                            + ":"
                            + str(
                                row.get("review_id")
                                or hashlib.sha256(
                                    json.dumps(row, sort_keys=True).encode()
                                ).hexdigest()
                            )
                        )
                        if any(r.get("external_review_id") == external_id for r in shot["reviews"]):
                            skipped += 1
                            continue
                        approved_suggestion = None
                        if row.get("decision") == "approved":
                            supplied = item.get("suggestion", {})
                            known = [shot["suggestion"]] + [
                                h["suggestion"] for h in shot.get("suggestion_history", [])
                            ]
                            approved_suggestion = next(
                                (
                                    n
                                    for n in known
                                    if n.get("text") == supplied.get("text")
                                    and n.get("method") == supplied.get("method")
                                ),
                                None,
                            )
                            if approved_suggestion is None:
                                raise ValueError("ההערה שאושרה אינה תואמת לגרסה שנמסרה למאמן")
                        self.review(
                            identifier,
                            {**row, "version": len(shot["reviews"])},
                            approved_suggestion=approved_suggestion,
                        )
                        shot["reviews"][-1].update(
                            external_review_id=external_id,
                            original_reviewed_at=row.get("created_at"),
                            import_package=package_id,
                        )
                        imported += 1
                        self.save()
                self.save()
            except Exception:
                for original in before["shots"]:
                    current = self.shot(original["id"])
                    current.clear()
                    current.update(original)
                self.save()
                raise
            return {"imported_reviews": imported, "already_imported": skipped}

    def review(self, identifier, payload, *, approved_suggestion=None):
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
            original_suggestion = approved_suggestion or shot["suggestion"]
            if decision == "approved" and not original_suggestion["available"]:
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
                "note": original_suggestion["text"] if decision == "approved" else note,
                "suggestion_snapshot": copy.deepcopy(original_suggestion),
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
    """A summary of angle ranges alone is insufficient for a temporal coaching cue."""
    from swisher.studio.feedback import propose

    return propose(
        candidate,
        {"coverage": 0, "frames": [candidate.get("shot_start", 0), candidate.get("shot_end", 0)]},
    )
