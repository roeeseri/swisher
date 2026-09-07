"""Run from project root. Latency/coverage comparison, NOT an accuracy benchmark."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from swisher.analysis import _extract_record
from swisher.pose_backend import Pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()
    if args.output.resolve() == args.video.resolve():
        raise ValueError("output cannot overwrite source")
    rows = []
    for complexity, name in [(1, "BlazePose Full"), (2, "BlazePose Heavy")]:
        cap = cv2.VideoCapture(str(args.video))
        latencies, valid, count = [], 0, 0
        try:
            with Pose(static_image_mode=True, model_complexity=complexity) as pose:
                while count < args.frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    before = time.perf_counter()
                    output = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    latencies.append((time.perf_counter() - before) * 1000)
                    record = _extract_record(
                        output.pose_landmarks, (0, 0, frame.shape[1], frame.shape[0]), count
                    )
                    valid += int(record["valid"])
                    count += 1
        finally:
            cap.release()
        rows.append(
            {
                "model": name,
                "frames": count,
                "valid_frame_coverage": valid / max(count, 1),
                "median_ms": float(np.median(latencies)) if latencies else None,
                "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "source": args.video.name,
                "results": rows,
                "limitation": "Full-frame smoke benchmark, no ground truth, no accuracy ranking.",
            },
            indent=2,
        )
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
