"""Render review-required event markers without changing source frame numbering."""

from pathlib import Path

import cv2


def render_event_overlay(source: Path, target: Path, report: dict) -> None:
    if source.resolve() == target.resolve():
        raise ValueError("event rendering must not overwrite its source")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("cannot open overlay source")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    preview_scale = min(960 / width, 1.0)
    width = int(width * preview_scale) // 2 * 2
    height = int(height * preview_scale) // 2 * 2
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    for codec in ["avc1", "mp4v"]:
        attempt = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if attempt.isOpened():
            writer = attempt
            break
        attempt.release()
    if writer is None:
        capture.release()
        raise ValueError("cannot encode event overlay")
    proposals = report["candidates"]
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height))
            candidate = next(
                (p for p in proposals if p["shot_start"] <= index <= p["shot_end"]), None
            )
            left = max(10, width - 650)
            cv2.rectangle(frame, (left, 8), (width - 8, 118), (20, 20, 20), -1)
            if candidate:
                release = candidate.get("release_frame")
                text = f"{candidate['shot_id']} | PROPOSAL - REVIEW REQUIRED"
                cv2.putText(
                    frame, text, (left + 12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2
                )
                line = (
                    "Release not detected"
                    if release is None
                    else f"Proposed release: {release / fps:.3f}s (frame {release})"
                )
                cv2.putText(
                    frame, line, (left + 12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 255), 2
                )
                interval = candidate.get("release_interval")
                if interval and interval[0] <= index <= interval[1]:
                    cv2.putText(
                        frame,
                        "BALL-HAND SEPARATION WINDOW",
                        (left + 12, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (80, 220, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "No shot proposal at this frame",
                    (left + 12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                )
            y = height - 25
            cv2.line(frame, (25, y), (width - 25, y), (90, 90, 90), 4)
            for p in proposals:
                x1 = 25 + int((width - 50) * p["shot_start"] / max(count - 1, 1))
                x2 = 25 + int((width - 50) * p["shot_end"] / max(count - 1, 1))
                cv2.line(frame, (x1, y), (x2, y), (200, 130, 60), 5)
                if p.get("release_frame") is not None:
                    x = 25 + int((width - 50) * p["release_frame"] / max(count - 1, 1))
                    cv2.line(frame, (x, y - 8), (x, y + 8), (60, 230, 255), 3)
            x = 25 + int((width - 50) * index / max(count - 1, 1))
            cv2.line(frame, (x, y - 10), (x, y + 10), (255, 255, 255), 2)
            writer.write(frame)
            index += 1
        if index != count:
            raise ValueError(f"event video frame mismatch: {index}/{count}")
    finally:
        capture.release()
        writer.release()
