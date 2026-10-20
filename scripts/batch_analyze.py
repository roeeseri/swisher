"""Run the existing Swisher engine on an explicit, resumable video manifest.

Usage: PYTHONPATH=src .venv/bin/python scripts/batch_analyze.py manifest.json
Manifest: {"output_dir": "artifacts/batch", "videos": [
  {"path": "/absolute/video.mov", "id": "video_001", "player_region": [x1,y1,x2,y2]}
]}
Results are unreviewed model output, never coach-approved annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    output = Path(payload["output_dir"]).resolve()
    videos = payload["videos"]
    ids = [item["id"] for item in videos]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids):
        raise ValueError("Video IDs must be unique and contain letters, digits, _ or -")
    output.mkdir(parents=True, exist_ok=True)
    from swisher.analysis import analyze_video

    summary = []
    for item in videos:
        source = Path(item["path"]).resolve(strict=True)
        target = output / item["id"]
        target.mkdir(exist_ok=True)
        region = tuple(item["player_region"]) if item.get("player_region") else None
        selection = (
            tuple(item["player_selection_region"]) if item.get("player_selection_region") else None
        )
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprint = digest.hexdigest()
        report_path = target / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            provenance = report.get("provenance", {})
            if (
                provenance.get("source_sha256") != fingerprint
                or provenance.get("player_region") != (list(region) if region else None)
                or provenance.get("player_selection_region")
                != (list(selection) if selection else None)
            ):
                raise ValueError(f"Existing output does not match source/settings: {target}")
            print(f"RESUME {item['id']}: report exists", flush=True)
        else:
            last = [-1]

            def progress(value, last=last, video_id=item["id"]):
                bucket = int(value * 10)
                if bucket != last[0]:
                    print(f"{video_id} {value:.0%}", flush=True)
                    last[0] = bucket

            try:
                result = analyze_video(
                    source,
                    target / "overlay.mp4",
                    on_progress=progress,
                    max_duration_seconds=item.get("max_duration_seconds", 120),
                    thorough=item.get("thorough", False),
                    track_camera_motion=item.get("track_camera_motion", True),
                    refine_ball_detection=item.get("refine_ball_detection", True),
                    player_region=region,
                    player_selection_region=selection,
                    trace_path=target / "pose.jsonl",
                )
                report = result.model_dump(mode="json")
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                (target / "error.txt").unlink(missing_ok=True)
            except Exception:
                (target / "error.txt").write_text(traceback.format_exc())
                summary.append({"video": item["id"], "status": "error"})
                continue
        if not (target / "events.mp4").exists():
            try:
                from swisher.event_overlay import render_event_overlay

                render_event_overlay(target / "overlay.mp4", target / "events.mp4", report)
            except Exception:
                (target / "event_error.txt").write_text(traceback.format_exc())
        summary.append(
            {
                "video": item["id"],
                "status": report["status"],
                "pose_coverage": report["pose_confidence"],
                "unreviewed_proposals": len(report["candidates"]),
                "coach_approved": False,
            }
        )
        (output / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    (output / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
