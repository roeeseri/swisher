from swisher.rim import _plausible_rim


def test_rejects_full_frame_false_positive() -> None:
    assert not _plausible_rim((0, 0, 1920, 1080), 1920, 1080)


def test_accepts_plausible_rim_box() -> None:
    assert _plausible_rim((1400, 250, 1530, 300), 1920, 1080)
