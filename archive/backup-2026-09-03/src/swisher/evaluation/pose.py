from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from swisher.data.models import PoseFrame


def pose_coverage(frames: Iterable[PoseFrame]) -> dict[str, float]:
    visible: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for frame in frames:
        for point in frame.keypoints:
            total[point.name] += 1
            visible[point.name] += int(point.visible)
    return {name: visible[name] / count for name, count in total.items() if count}
