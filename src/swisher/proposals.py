"""Conservative raised-arm episodes, not a trained shot/action classifier."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np


def propose_shots(
    frames: list[dict],
    balls: list[dict],
    fps: float,
    frame_count: int,
    release_detector: Callable,
) -> list[dict]:
    episodes: list[list[dict]] = []
    active: list[dict] = []
    last_valid = -10000
    for frame in frames:
        index = int(frame["index"])
        if not frame.get("valid"):
            if index - last_valid > max(2, int(fps * 0.15)) and active:
                episodes.append(active)
                active = []
            continue
        xy = frame["xy"]
        visibility = frame.get("visibility_by_name", {})
        raised = False
        for shoulder, elbow, wrist, hip in (("ls", "le", "lw", "lh"), ("rs", "re", "rw", "rh")):
            if min(visibility.get(k, 0) for k in (shoulder, elbow, wrist, hip)) < 0.5:
                continue
            torso = max(float(np.linalg.norm(xy[shoulder] - xy[hip])), 1)
            raised |= xy[wrist][1] < xy[shoulder][1] - torso * 0.12
        last_valid = index
        if raised:
            active.append(frame)
        elif active:
            episodes.append(active)
            active = []
    if active:
        episodes.append(active)
    proposals = []
    for episode in episodes:
        if len(episode) < max(3, int(fps * 0.10)):
            continue
        first, last = int(episode[0]["index"]), int(episode[-1]["index"])
        if last - first > fps * 5:  # stationary raised hands are not repeat shots
            continue
        start, end = max(0, first - int(fps)), min(frame_count - 1, last + int(fps * 1.0))
        local = [f for f in frames if f.get("valid") and start <= f["index"] <= end]
        local_balls = [b for b in balls if first - fps * 0.3 <= b["frame"] <= last]
        release = release_detector(local, local_balls, fps)
        if proposals and first - proposals[-1]["motion_frame"] < fps * 0.8:
            continue
        proposals.append(
            {
                "shot_id": f"shot_{len(proposals) + 1:03d}",
                "shot_start": start,
                "shot_end": end,
                "motion_frame": first,
                "release_frame": release.frame if release else None,
                "release_interval": (
                    [release.interval_start_frame, release.interval_end_frame] if release else None
                ),
                "method": "raised_arm_with_ball_separation_v0.3" if release else "raised_arm_v0.3",
                "review_required": True,
                "outcome": "unknown",
                "evidence": "נמצאה הפרדת כדור–יד" if release else "הרמת יד בלבד; השחרור לא זוהה",
            }
        )
    # Keep proposed shots separate; never allow frame ownership to overlap.
    for previous, current in pairwise(proposals):
        if previous["shot_end"] >= current["shot_start"]:
            boundary = (previous["motion_frame"] + current["motion_frame"]) // 2
            previous["shot_end"] = boundary
            current["shot_start"] = boundary + 1
            for item in (previous, current):
                if item["release_frame"] is not None and not (
                    item["shot_start"] <= item["release_frame"] <= item["shot_end"]
                ):
                    item["release_frame"] = None
                    item["release_interval"] = None
    return proposals
