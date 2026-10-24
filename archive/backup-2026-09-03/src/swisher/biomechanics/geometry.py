from __future__ import annotations

import math

from swisher.data.models import Keypoint


def joint_angle(a: Keypoint, vertex: Keypoint, c: Keypoint) -> float | None:
    if not (a.visible and vertex.visible and c.visible):
        return None
    first = (a.x - vertex.x, a.y - vertex.y)
    second = (c.x - vertex.x, c.y - vertex.y)
    norm_first = math.hypot(*first)
    norm_second = math.hypot(*second)
    if norm_first == 0 or norm_second == 0:
        return None
    cosine = (first[0] * second[0] + first[1] * second[1]) / (norm_first * norm_second)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
