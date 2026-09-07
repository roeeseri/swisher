import numpy as np
import pytest

from swisher.pose_backend import prepare_pixels


def test_metal_requires_rgba_and_preserves_rgb():
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    pixels = prepare_pixels(rgb, True)
    assert pixels.shape == (3, 4, 4)
    assert (pixels[:, :, :3] == rgb).all()
    assert (pixels[:, :, 3] == 255).all()
    assert pixels.flags["C_CONTIGUOUS"]


def test_cpu_rgb_is_contiguous():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)[::2]
    assert prepare_pixels(rgb, False).flags["C_CONTIGUOUS"]


def test_bad_pose_input_is_rejected_before_native_inference():
    with pytest.raises(ValueError):
        prepare_pixels(np.zeros((4, 4)), True)
