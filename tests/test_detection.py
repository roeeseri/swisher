import numpy as np

from swisher.detection import Detection, padded_crop, select_primary_player, smooth_box


def test_selects_person_near_ball() -> None:
    detections = [
        Detection("person", 0.9, (10, 10, 60, 190)),
        Detection("person", 0.8, (190, 20, 250, 190)),
        Detection("sports_ball", 0.7, (225, 30, 240, 45)),
    ]
    selected = select_primary_player(detections, (200, 300, 3))
    assert selected is not None
    assert selected.xyxy == (190, 20, 250, 190)


def test_crop_is_clamped_to_frame() -> None:
    assert padded_crop((0, 0, 100, 100), 200, 150) == (0, 0, 145, 145)


def test_smooth_box_preserves_history() -> None:
    assert smooth_box((0, 0, 10, 10), (10, 10, 20, 20), alpha=0.5) == (
        5.0,
        5.0,
        15.0,
        15.0,
    )


def test_detail_ball_boxes_map_back_to_source_without_importing_people():
    from swisher.detection import detect_region_balls

    class Detector:
        def detect(self, image, threshold):
            assert image.shape == (100, 120, 3)
            return [
                Detection("sports_ball", 0.8, (10.0, 20.0, 30.0, 40.0)),
                Detection("person", 0.9, (0.0, 0.0, 100.0, 100.0)),
            ]

    result = detect_region_balls(
        Detector(), np.zeros((480, 640, 3), dtype=np.uint8), (100, 200, 220, 300)
    )
    assert len(result) == 1
    assert result[0].xyxy == (110.0, 220.0, 130.0, 240.0)
