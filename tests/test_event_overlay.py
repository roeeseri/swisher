import cv2
import numpy as np
import pytest

from swisher.event_overlay import render_event_overlay


def test_event_preview_keeps_frame_count_and_fps_and_preserves_source(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "events.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
    writer.release()
    original = source.read_bytes()
    render_event_overlay(
        source,
        target,
        {
            "candidates": [
                {
                    "shot_id": "shot_001",
                    "shot_start": 0,
                    "shot_end": 7,
                    "release_frame": 3,
                    "release_interval": [2, 3],
                }
            ]
        },
    )
    cap = cv2.VideoCapture(str(target))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 30) < 0.01
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.max() > 30
    assert source.read_bytes() == original
    with pytest.raises(ValueError):
        render_event_overlay(source, source, {"candidates": []})
