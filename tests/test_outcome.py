import numpy as np

from swisher.analysis import _shot_outcome
from swisher.rim import RimDetection


def test_detects_downward_crossing_inside_rim() -> None:
    rim = RimDetection(confidence=0.8, xyxy=(100, 100, 160, 115))
    observations = [
        {"frame": 1, "center": np.array([130.0, 70.0]), "confidence": 0.8},
        {"frame": 2, "center": np.array([132.0, 95.0]), "confidence": 0.8},
        {"frame": 3, "center": np.array([133.0, 120.0]), "confidence": 0.8},
        {"frame": 4, "center": np.array([134.0, 140.0]), "confidence": 0.8},
    ]
    outcome = _shot_outcome(observations, rim)
    assert outcome.status == "unknown"  # 2D overlap is not evidence of depth
    assert outcome.confidence == 0


def test_abstains_without_rim() -> None:
    outcome = _shot_outcome([], None)
    assert outcome.status == "unknown"
    assert outcome.confidence == 0.0
