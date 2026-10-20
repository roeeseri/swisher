"""Conservative ball/visible-wrist separation; not a pose-only release guess."""

from __future__ import annotations

import itertools

import numpy as np

from swisher.contracts import ReleaseEvent


def detect_release(valid, balls, fps):
    if len(valid) < 3 or len(balls) < 3:
        return None
    poses = {int(p["index"]): p for p in valid}
    grouped = {}
    for ball in balls:
        grouped.setdefault(int(ball["frame"]), []).append(ball)
    candidates = []
    for index, detections in sorted(grouped.items()):
        nearby = [i for i in poses if abs(i - index) <= 3]
        if not nearby:
            continue
        pose = poses[min(nearby, key=lambda i: abs(i - index))]
        xy, visibility = pose["xy"], pose.get("visibility_by_name", {})
        if min(visibility.get(k, 0) for k in ("ls", "rs")) < 0.5:
            continue
        width = float(np.linalg.norm(xy["ls"] - xy["rs"]))
        torsos = [
            float(np.linalg.norm(xy[s] - xy[h]))
            for s, h in [("ls", "lh"), ("rs", "rh")]
            if h in xy and min(visibility.get(s, 0), visibility.get(h, 0)) >= 0.4
        ]
        if torsos:
            width = max(width, float(np.median(torsos)) * 0.5)
        if width < 12:
            continue
        arms = [
            (side, wrist)
            for side, wrist, shoulder in [("left", "lw", "ls"), ("right", "rw", "rs")]
            if visibility.get(wrist, 0) >= 0.5 and xy[wrist][1] <= xy[shoulder][1]
        ]
        if not arms:
            continue
        choices = []
        for ball in detections:
            center = np.asarray(ball["center"], dtype=float)
            for side, wrist in arms:
                distance = float(np.linalg.norm(center - xy[wrist])) / width
                if distance <= 4:
                    choices.append((distance, side, center, float(ball["confidence"])))
        if not choices:
            continue
        distance, side, center, confidence = min(choices, key=lambda c: c[0])
        candidates.append(
            {
                "frame": index,
                "distance": distance,
                "side": side,
                "center": center,
                "width": width,
                "confidence": confidence,
                "above_wrist": float(xy["lw" if side == "left" else "rw"][1] - center[1]) / width,
                "both_wrists_visible": min(visibility.get("lw", 0), visibility.get("rw", 0)) >= 0.5,
            }
        )
    if len(candidates) < 3:
        return None
    pairs = []
    max_gap = max(3, int(fps * 0.12))
    for i, (before, after) in enumerate(itertools.pairwise(candidates)):
        gap = after["frame"] - before["frame"]
        scale = (before["width"] + after["width"]) / 2
        growth = after["distance"] - before["distance"]
        if not 0 < gap <= max_gap:
            continue
        displacement = np.linalg.norm(after["center"] - before["center"]) / scale
        if displacement > max(1.5, gap * 0.9):
            continue  # do not jump to a different distant ball
        if (
            before["distance"] > 0.8
            or after["distance"] < 1.0
            or growth < 0.25
            or after["above_wrist"] < 0.5
        ):
            continue
        rise = float(before["center"][1] - after["center"][1])
        if rise < max(
            0.08 * scale, 0.25 * float(np.linalg.norm(after["center"] - before["center"]))
        ):
            continue
        support = []
        if i:
            support.append(candidates[i - 1])
        if i + 2 < len(candidates):
            support.append(candidates[i + 2])
        if not any(
            min(abs(s["frame"] - before["frame"]), abs(s["frame"] - after["frame"])) <= max_gap
            for s in support
        ):
            continue
        pairs.append((growth, i, before, after))
    if not pairs:
        return None
    _, _, before, after = max(pairs, key=lambda p: p[0])
    confidence = float(
        np.clip(
            0.35 + (after["distance"] - before["distance"]) * 0.35 + after["confidence"] * 0.35,
            0,
            0.95,
        )
    )
    # One visible hand can establish separation but cannot establish shooting handedness.
    side = (
        after["side"]
        if before["both_wrists_visible"]
        and after["both_wrists_visible"]
        and before["side"] == after["side"]
        else "unknown"
    )
    return ReleaseEvent(
        frame=after["frame"],
        seconds=after["frame"] / fps,
        confidence=confidence,
        shooting_side=side,
        wrist_ball_distance_norm=after["distance"],
        interval_start_frame=before["frame"],
        interval_end_frame=after["frame"],
        method="visible_wrist_ball_separation_v0.4",
    )
