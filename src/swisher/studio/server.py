"""Local Swisher Studio. Run: python -m swisher.studio.server --port 8770."""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import ssl
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from swisher.session import inspect_video, normalize_browser_video
from swisher.studio import media
from swisher.studio.feedback import generate
from swisher.studio.store import StudioStore, now, suggestion


class Studio:
    def __init__(self, root: Path):
        self.store = StudioStore(root)
        self.token = secrets.token_urlsafe(32)
        self.stopping = threading.Event()
        self.phone_info = None
        self.region_cache = {}
        self.visual_cache = {}
        self.pair_code = f"{secrets.randbelow(100000000):08d}"
        self.pair_expires = time.monotonic() + 600
        self.phone_sessions = {}
        self.pair_attempts = {}

    def visuals(self, job):
        key = job["id"]
        if key not in self.visual_cache:
            poses, balls = {}, {}
            report = job.get("report") or job.get("existing_report")
            if report:
                folder = Path(report).parent
                trace = folder / "pose.jsonl"
                if trace.exists():
                    for line in trace.open():
                        row = json.loads(line)
                        poses[row["index"]] = row
                ball_path = folder / "balls.json"
                if ball_path.exists():
                    for ball in json.loads(ball_path.read_text()):
                        balls.setdefault(ball["frame"], []).append(ball)
            if len(self.visual_cache) >= 3:
                self.visual_cache.pop(next(iter(self.visual_cache)))
            self.visual_cache[key] = (poses, balls)
        return self.visual_cache[key]

    def frame_region(self, job, index):
        key = job["id"]
        if key not in self.region_cache:
            regions = {}
            report = job.get("report") or job.get("existing_report")
            if report:
                trace = Path(report).parent / "pose.jsonl"
                if trace.exists():
                    for line in trace.open():
                        row = json.loads(line)
                        if row.get("camera_tracking_valid", True) and row.get("player_region"):
                            regions[row["index"]] = row["player_region"]
            self.region_cache[key] = regions
        return self.region_cache[key].get(index)

    def feedback_for(self, report_path, report):
        trace = Path(report_path).parent / "pose.jsonl"
        if not trace.exists():
            return {}
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        return generate(
            report["candidates"],
            records,
            report["video"]["fps"],
            report.get("provenance", {}).get("camera_motion_failures", 0) == 0,
        )

    def refresh_suggestions(self):
        from collections import Counter

        predictions = {}
        for job in self.store.snapshot()["jobs"]:
            report_path = job.get("report") or job.get("existing_report")
            if not report_path or not Path(report_path).exists():
                continue
            report = json.loads(Path(report_path).read_text())
            predictions[job["id"]] = self.feedback_for(report_path, report)
        changed = skipped = 0
        with self.store.lock:
            backup = self.store.root / "history" / ("feedback_before_" + uuid4().hex + ".json")
            from swisher.session import SessionStore

            SessionStore._atomic_json(backup, self.store.data)
            for shot in self.store.data["shots"]:
                if shot["reviews"]:
                    skipped += 1
                    continue
                proposed = predictions.get(shot["job_id"], {}).get(shot["candidate"]["shot_id"])
                if proposed is None or proposed == shot["suggestion"]:
                    continue
                shot.setdefault("suggestion_history", []).append(
                    {"updated_at": now(), "suggestion": shot["suggestion"]}
                )
                shot["suggestion"] = proposed
                changed += 1
            self.store.save()
            notes = [s["suggestion"] for s in self.store.data["shots"]]
            return {
                "updated": changed,
                "reviewed_preserved": skipped,
                "available": sum(n["available"] for n in notes),
                "focus": dict(Counter(n.get("focus", "abstained") for n in notes)),
            }

    def start(self):
        threading.Thread(target=self.work, daemon=True).start()

    def work(self):
        while not self.stopping.is_set():
            with self.store.lock:
                job = next((j for j in self.store.data["jobs"] if j["status"] == "queued"), None)
                if job:
                    job["status"] = "processing"
                    self.store.save()
            if job:
                try:
                    self.process(job)
                except Exception:
                    traceback.print_exc()
                    with self.store.lock:
                        job["status"] = "error"
                        job["error"] = "העיבוד נכשל. המקור נשמר; אפשר לנסות שוב."
                        self.store.save()
                continue
            with self.store.lock:
                shot = next(
                    (s for s in self.store.data["shots"] if s["media_status"] == "queued"), None
                )
                if shot:
                    shot["media_status"] = "processing"
                    job = next(j for j in self.store.data["jobs"] if j["id"] == shot["job_id"])
                    version = len(shot["reviews"])
                    review = dict(shot["reviews"][-1]) if shot["reviews"] else None
            if shot:
                try:
                    c = shot["candidate"]
                    folder = self.store.root / "shots" / shot["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    start = review["start_frame"] if review else c["shot_start"]
                    end = review["end_frame"] if review else c["shot_end"]
                    media.clip(
                        Path(job["source"]),
                        folder / "clip.mp4",
                        start,
                        end,
                        job["video"]["fps"],
                        self.visuals(job),
                    )
                    index = review["release_frame"] if review else c.get("release_frame")
                    if index is None:
                        index = max(start, min(end, c.get("motion_frame", start)))
                    media.frame(Path(job["source"]), folder / "frame.jpg", index)
                    with self.store.lock:
                        shot["visual_version"] = 2
                        shot["media_status"] = (
                            "ready" if len(shot["reviews"]) == version else "queued"
                        )
                        self.store.save()
                except Exception:
                    traceback.print_exc()
                    with self.store.lock:
                        shot["media_status"] = "error"
                        self.store.save()
                continue
            self.stopping.wait(0.5)

    def process(self, job):
        folder = self.store.root / "jobs" / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        source = Path(job["source"])
        if source.suffix.lower() == ".webm":
            normalized = folder / "source.mp4"
            normalize_browser_video(source, normalized, max_duration_seconds=600)
            source = normalized
            job["source"] = str(source.resolve())
        report_path = (
            Path(job["existing_report"]) if job.get("existing_report") else folder / "report.json"
        )
        if report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            from swisher.analysis import analyze_video

            facts = inspect_video(source)
            if facts["frame_count"] / facts["fps"] > 600:
                raise ValueError("video exceeds ten minutes")
            region = job.get("region")
            if region:
                w, h = facts["width"], facts["height"]
                region = (
                    int(region[0] * w),
                    int(region[1] * h),
                    int(region[2] * w),
                    int(region[3] * h),
                )

            def progress(value):
                with self.store.lock:
                    job["progress"] = value
                    self.store.save()

            result = analyze_video(
                source,
                folder / "overlay.mp4",
                on_progress=progress,
                player_region=region,
                trace_path=folder / "pose.jsonl",
                max_duration_seconds=600,
            )
            report = result.model_dump(mode="json")
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        metrics_path = report_path.parent / "proposal_metrics.json"
        if not metrics_path.exists() and (report_path.parent / "pose.jsonl").exists():
            # Use the same measurement exporter as the batch pipeline.
            import importlib.util

            script = Path(__file__).resolve().parents[3] / "scripts" / "proposal_metrics.py"
            if script.exists():
                spec = importlib.util.spec_from_file_location("studio_metrics", script)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.export_metrics(report_path.parent)
        measurements = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
        lookup = {m["proposal_id"]: m for m in measurements}
        predicted_feedback = self.feedback_for(report_path, report)
        with self.store.lock:
            job["video"] = report["video"]
            job["source_sha256"] = report.get("provenance", {}).get("source_sha256")
            job["report"] = str(report_path.resolve())
            existing = {
                s["candidate"]["shot_id"]
                for s in self.store.data["shots"]
                if s["job_id"] == job["id"]
            }
            for candidate in report["candidates"]:
                if candidate["shot_id"] in existing:
                    continue
                self.store.data["shots"].append(
                    {
                        "id": uuid4().hex,
                        "job_id": job["id"],
                        "candidate": candidate,
                        "suggestion": predicted_feedback.get(candidate["shot_id"])
                        or suggestion(candidate, lookup.get(candidate["shot_id"])),
                        "measurement": lookup.get(candidate["shot_id"]),
                        "reviews": [],
                        "media_status": "queued",
                        "boundary_warning": bool(job.get("capture_id"))
                        and (
                            candidate["shot_start"] < report["video"]["fps"]
                            or candidate["shot_end"]
                            > report["video"]["frame_count"] - report["video"]["fps"]
                        ),
                    }
                )
            job["status"] = "ready"
            job["progress"] = 1
            self.store.save()

    def import_pilot(self, root: Path):
        manifest = json.loads((root / "all_manifest.json").read_text())
        with self.store.lock:
            imported = {j.get("existing_report") for j in self.store.data["jobs"]}
            for item in sorted(manifest["videos"], key=lambda i: (i["id"] != "IMG_1083", i["id"])):
                report = (root / item["id"] / "report.json").resolve()
                if (
                    str(report) in imported
                    or not report.exists()
                    or not Path(item["path"]).exists()
                ):
                    continue
                self.store.add_job(
                    name=item["id"],
                    source=str(Path(item["path"]).resolve()),
                    existing_report=str(report),
                    player="לא שויך",
                    view="צילום פיילוט",
                    consent=True,
                    capture_id=None,
                )


class Handler(BaseHTTPRequestHandler):
    server_version = "SwisherStudio/1"

    @property
    def studio(self):
        return self.server.studio

    def valid_host(self):
        return self.headers.get("Host") in getattr(self.server, "allowed_hosts", set()) | {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }

    def send(self, code, payload, content_type="application/json; charset=utf-8", cookie=None):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, ensure_ascii=False).encode()
        if isinstance(payload, str):
            payload = payload.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if cookie:
            secure = "; Secure" if getattr(self.server, "phone", False) else ""
            self.send_header(
                "Set-Cookie", f"swisher={cookie}; HttpOnly; SameSite=Strict; Path=/{secure}"
            )
        self.end_headers()
        self.wfile.write(payload)

    def client_token(self):
        values = self.headers.get("Cookie", "").split("; ")
        return next((v.partition("=")[2] for v in values if v.startswith("swisher=")), "")

    def token_valid(self, token):
        if getattr(self.server, "phone", False):
            return self.studio.phone_sessions.get(token, 0) > time.monotonic()
        return bool(token) and secrets.compare_digest(token, self.studio.token)

    def authorized(self):
        return self.token_valid(self.client_token())

    def do_GET(self):
        if not self.valid_host() or self.headers.get("Sec-Fetch-Site") == "cross-site":
            return self.send(403, {"error": "גישה מקומית בלבד"})
        parsed = urlparse(self.path)
        if parsed.path == "/" and getattr(self.server, "phone", False) and not self.authorized():
            return self.send(
                200, Path(__file__).with_name("pair.html").read_text(), "text/html; charset=utf-8"
            )
        if parsed.path == "/":
            token = (
                self.client_token() if getattr(self.server, "phone", False) else self.studio.token
            )
            page = Path(__file__).with_name("index.html").read_text().replace("__TOKEN__", token)
            return self.send(200, page, "text/html; charset=utf-8", cookie=token)
        if not self.authorized():
            return self.send(403, {"error": "פתחו את מסך הבית תחילה"})
        if parsed.path == "/api/phone" and not getattr(self.server, "phone", False):
            return self.send(
                200,
                {
                    "connection": self.studio.phone_info,
                    "code": self.studio.pair_code,
                    "expires_seconds": max(0, int(self.studio.pair_expires - time.monotonic())),
                },
            )
        if parsed.path == "/api/state":
            state = self.studio.store.snapshot()
            for j in state["jobs"]:
                for key in ("source", "existing_report", "report"):
                    j.pop(key, None)
            return self.send(200, state)
        if parsed.path == "/api/export":
            state = self.studio.store.snapshot()
            rows = [
                {
                    "shot_id": s["id"],
                    "job_id": s["job_id"],
                    "candidate": s["candidate"],
                    "suggestion": s["suggestion"],
                    "reviews": s["reviews"],
                }
                for s in state["shots"]
            ]
            return self.send(
                200,
                {
                    "schema_version": 1,
                    "shots": rows,
                    "jobs": [
                        {
                            key: j.get(key)
                            for key in (
                                "id",
                                "player",
                                "view",
                                "source_sha256",
                                "video",
                                "capture_id",
                                "segment",
                            )
                        }
                        for j in state["jobs"]
                    ],
                },
            )
        parts = parsed.path.strip("/").split("/")
        if (
            len(parts) == 3
            and parts[0] == "media"
            and parts[2] in ("clip.mp4", "frame.jpg", "source.mp4", "preview.jpg")
        ):
            with self.studio.store.lock:
                shot = self.studio.store.shot(parts[1])
                if not shot:
                    return self.send(404, {"error": "לא נמצא"})
                job = next(j for j in self.studio.store.data["jobs"] if j["id"] == shot["job_id"])
                folder = self.studio.store.root / "shots" / shot["id"]
                if parts[2] == "source.mp4":
                    target = Path(job["source"])
                elif parts[2] == "preview.jpg":
                    try:
                        index = int(parse_qs(parsed.query)["frame"][0])
                        if not 0 <= index < job["video"]["frame_count"]:
                            raise ValueError()
                        folder.mkdir(parents=True, exist_ok=True)
                        target = folder / f"clean_{index}.jpg"
                        if not target.exists():
                            media.frame(
                                Path(job["source"]),
                                target,
                                index,
                                self.studio.frame_region(job, index),
                            )
                    except (KeyError, ValueError):
                        return self.send(400, {"error": "פריים לא תקין"})
                else:
                    target = folder / parts[2]
            if target.is_file():
                return self.file(target)
        return self.send(404, {"error": "לא נמצא"})

    def file(self, path):
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        range_header = self.headers.get("Range")
        if range_header:
            try:
                spec = range_header.removeprefix("bytes=").split("-")
                start = int(spec[0])
                end = min(int(spec[1]) if spec[1] else size - 1, size - 1)
                if not 0 <= start <= end < size:
                    raise ValueError()
                code = 206
            except (ValueError, IndexError):
                return self.send(416, {"error": "טווח לא תקין"})
        self.send_response(code)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self):
        parsed = urlparse(self.path)
        if not self.valid_host():
            return self.send(403, {"error": "בקשה לא מורשית"})
        if parsed.path == "/api/pair" and getattr(self.server, "phone", False):
            try:
                origin = self.headers.get("Origin")
                if origin and origin != "https://" + self.headers.get("Host"):
                    raise ValueError("מקור בקשה לא מורשה")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 1000:
                    raise ValueError("בקשה לא תקינה")
                payload = json.loads(self.rfile.read(length))
                address = self.client_address[0]
                with self.studio.store.lock:
                    attempts = [
                        v
                        for v in self.studio.pair_attempts.get(address, [])
                        if v > time.monotonic() - 60
                    ]
                    self.studio.pair_attempts[address] = attempts + [time.monotonic()]
                    if len(attempts) >= 5:
                        return self.send(429, {"error": "נסיונות רבים מדי. המתינו דקה"})
                    if time.monotonic() > self.studio.pair_expires or not secrets.compare_digest(
                        str(payload.get("code", "")), self.studio.pair_code
                    ):
                        raise ValueError("קוד שגוי או שפג תוקפו. צרו קוד חדש במחשב")
                    token = secrets.token_urlsafe(32)
                    self.studio.phone_sessions[token] = time.monotonic() + 86400
                    self.studio.pair_expires = 0
                return self.send(200, {"ok": True}, cookie=token)
            except (ValueError, TypeError):
                return self.send(400, {"error": "לא ניתן לצמד. בדקו את הקוד במחשב"})
        if not self.token_valid(self.headers.get("X-Swisher-Token", "")):
            return self.send(403, {"error": "בקשה לא מורשית"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 200 * 1024 * 1024:
                return self.send(413, {"error": "הקובץ חייב להיות עד 200MB"})
            if parsed.path == "/api/upload":
                query = parse_qs(parsed.query)
                if query.get("consent") != ["yes"]:
                    raise ValueError("נדרש אישור צילום ושימוש לפני העלאה")
                region = json.loads(query.get("region", ["null"])[0])
                if region is not None and (
                    not isinstance(region, list)
                    or len(region) != 4
                    or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in region)
                    or not (region[0] < region[2] and region[1] < region[3])
                ):
                    raise ValueError("אזור השחקן לא תקין")
                segment = int(query.get("segment", ["0"])[0])
                suffix = (
                    ".webm"
                    if "webm" in self.headers.get("Content-Type", "")
                    else Path(query.get("name", ["video.mp4"])[0]).suffix.lower()
                )
                if suffix not in (".mp4", ".mov", ".webm", ".m4v"):
                    raise ValueError("פורמט לא נתמך")
                with self.studio.store.lock:
                    if (
                        sum(
                            j["status"] in ("queued", "processing")
                            for j in self.studio.store.data["jobs"]
                        )
                        >= 40
                    ):
                        return self.send(429, {"error": "התור מלא. עצרו צילום והמתינו לעיבוד"})
                target = self.studio.store.root / "uploads" / (uuid4().hex + suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                remaining = length
                try:
                    with target.open("wb") as f:
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("ההעלאה לא הושלמה")
                            f.write(chunk)
                            remaining -= len(chunk)
                    if suffix != ".webm":
                        facts = inspect_video(target)
                        if facts["frame_count"] / facts["fps"] > 600:
                            raise ValueError("יש להעלות סרטון עד עשר דקות")
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
                job = self.studio.store.add_job(
                    source=str(target.resolve()),
                    name=query.get("name", ["צילום חדש"])[0][:100],
                    player=query.get("player", ["ללא מזהה"])[0][:80],
                    view=query.get("view", ["לא צוין"])[0][:80],
                    consent=True,
                    capture_id=query.get("capture", [None])[0],
                    segment=segment,
                    region=region,
                )
                return self.send(201, {"id": job["id"]})
            if length > (12 * 1024 * 1024 if parsed.path == "/api/import-feedback" else 20000):
                raise ValueError("בקשה גדולה מדי")
            payload = json.loads(self.rfile.read(length))
            if parsed.path == "/api/refresh-suggestions" and not getattr(
                self.server, "phone", False
            ):
                return self.send(200, self.studio.refresh_suggestions())
            if parsed.path == "/api/import-feedback" and not getattr(self.server, "phone", False):
                return self.send(200, self.studio.store.import_feedback(payload))
            if parsed.path == "/api/new-pair" and not getattr(self.server, "phone", False):
                with self.studio.store.lock:
                    self.studio.pair_code = f"{secrets.randbelow(100000000):08d}"
                    self.studio.pair_expires = time.monotonic() + 600
                return self.send(200, {"ok": True})
            if parsed.path == "/api/review":
                review = self.studio.store.review(payload["id"], payload)
                return self.send(200, review)
            if parsed.path == "/api/manual-shot":
                with self.studio.store.lock:
                    job = next(
                        j for j in self.studio.store.data["jobs"] if j["id"] == payload["job_id"]
                    )
                    if job["status"] != "ready":
                        raise ValueError("יש להמתין לסיום הסריקה")
                    start, end = int(payload["start_frame"]), int(payload["end_frame"])
                    release = payload.get("release_frame")
                    release = None if release is None else int(release)
                    if not 0 <= start < end < job["video"]["frame_count"]:
                        raise ValueError("טווח הזריקה מחוץ לסרטון")
                    if release is not None and not start <= release <= end:
                        raise ValueError("השחרור צריך להיות בתוך הזריקה")
                    candidate = {
                        "shot_id": "manual_" + uuid4().hex[:8],
                        "shot_start": start,
                        "shot_end": end,
                        "release_frame": release,
                        "motion_frame": start,
                        "method": "manual_added",
                        "review_required": True,
                    }
                    shot = {
                        "id": uuid4().hex,
                        "job_id": job["id"],
                        "candidate": candidate,
                        "suggestion": suggestion(candidate, None),
                        "measurement": None,
                        "reviews": [],
                        "media_status": "queued",
                        "boundary_warning": False,
                    }
                    self.studio.store.data["shots"].append(shot)
                    self.studio.store.save()
                return self.send(201, {"id": shot["id"]})
            if parsed.path == "/api/job-label":
                with self.studio.store.lock:
                    job = next(
                        j for j in self.studio.store.data["jobs"] if j["id"] == payload["id"]
                    )
                    player = str(payload.get("player", "")).strip()
                    if not player or len(player) > 80:
                        raise ValueError("נדרש מזהה שחקן עד 80 תווים")
                    job["player"] = player
                    self.studio.store.save()
                return self.send(200, {"ok": True})
            if parsed.path == "/api/retry-shot":
                with self.studio.store.lock:
                    shot = self.studio.store.shot(payload["id"])
                    if not shot or shot["media_status"] != "error":
                        raise ValueError("הקטע אינו במצב שגיאה")
                    shot["media_status"] = "queued"
                    self.studio.store.save()
                return self.send(200, {"ok": True})
            if parsed.path == "/api/retry":
                with self.studio.store.lock:
                    job = next(
                        j for j in self.studio.store.data["jobs"] if j["id"] == payload["id"]
                    )
                    if job["status"] != "error":
                        raise ValueError("אפשר לנסות מחדש רק עיבוד שנכשל")
                    job["status"] = "queued"
                    job["error"] = None
                    self.studio.store.save()
                return self.send(200, {"ok": True})
            return self.send(404, {"error": "לא נמצא"})
        except (ValueError, KeyError, TypeError, StopIteration) as e:
            return self.send(400, {"error": str(e) or "בקשה לא תקינה"})

    def log_message(self, format, *args):
        if args and str(args[0]).startswith("POST"):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--root", type=Path, default=Path("data/studio"))
    parser.add_argument("--import-pilot", type=Path)
    parser.add_argument("--phone", action="store_true", help="Enable paired LAN HTTPS capture")
    parser.add_argument("--phone-port", type=int, default=8772)
    parser.add_argument("--certificate-port", type=int, default=8773)
    args = parser.parse_args()
    studio = Studio(args.root.resolve())
    if args.import_pilot:
        studio.import_pilot(args.import_pilot.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.studio = studio
    extra_servers = []
    if args.phone:
        from swisher.studio.phone import bootstrap_server, certificate, lan_ip

        address = lan_ip()
        cert_root = args.root.resolve() / "certificates"
        cert, key = certificate(cert_root, address)
        phone = ThreadingHTTPServer(("0.0.0.0", args.phone_port), Handler)
        phone.phone = True
        phone.allowed_hosts = {f"{address}:{args.phone_port}"}
        phone.studio = studio
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        phone.socket = context.wrap_socket(phone.socket, server_side=True)
        bootstrap = bootstrap_server(cert_root, args.certificate_port)
        extra_servers = [phone, bootstrap]
        for extra in extra_servers:
            threading.Thread(target=extra.serve_forever, daemon=True).start()
        studio.phone_info = {
            "capture_url": f"https://{address}:{args.phone_port}",
            "setup_url": f"http://{address}:{args.certificate_port}",
            "transport": "LAN HTTPS; local CA requires owner trust on phone",
        }
        print(f"Phone setup: {studio.phone_info['setup_url']}", flush=True)
    studio.start()
    print(f"Swisher Studio: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        studio.stopping.set()
        server.server_close()
        for extra in extra_servers:
            extra.shutdown()
            extra.server_close()


if __name__ == "__main__":
    main()
