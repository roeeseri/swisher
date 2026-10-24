from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from swisher.pose.export import run_pose_jsonl, run_pose_with_video
from swisher.pose.mediapipe_adapter import MediaPipePoseEstimator
from swisher.video.inspection import VideoInspectionError, VideoInspector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swisher", description="Swisher AI prototype tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect-video", help="Inspect technical video quality")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--json", action="store_true", dest="as_json")
    pose = subparsers.add_parser("run-pose", help="Run the baseline pose adapter")
    pose.add_argument("path", type=Path)
    pose.add_argument("--jsonl", type=Path, required=True)
    pose.add_argument("--video", type=Path, help="Reserved for annotated video export")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-video":
        try:
            metadata, report = VideoInspector().inspect(args.path)
        except VideoInspectionError as error:
            print(f"REJECT: {error}", file=sys.stderr)
            return 2
        payload = {
            "metadata": metadata.model_dump(mode="json"),
            "quality": report.model_dump(mode="json"),
        }
        if args.as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"{report.decision.value.upper()}: "
                f"{metadata.width}x{metadata.height} @ {metadata.fps:.2f}fps"
            )
            for issue in report.issues:
                print(f"- {issue.code}: {issue.message}")
        return 0 if report.decision.value != "reject" else 2
    if args.command == "run-pose":
        try:
            estimator = MediaPipePoseEstimator()
            if args.video:
                count = run_pose_with_video(args.path, args.jsonl, args.video, estimator)
            else:
                count = run_pose_jsonl(args.path, args.jsonl, estimator)
        except RuntimeError as error:
            print(f"REJECT: {error}", file=sys.stderr)
            return 2
        print(f"Wrote {count} pose frames to {args.jsonl}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
