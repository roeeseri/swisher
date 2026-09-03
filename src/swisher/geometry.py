from __future__ import annotations

import math

import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Return the internal angle ABC, or None for degenerate geometry."""
    ba = a - b
    bc = c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator < 1e-8:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def trunk_lean_deg(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    vector = shoulder_mid - hip_mid
    return math.degrees(math.atan2(float(vector[0]), float(-vector[1])))
