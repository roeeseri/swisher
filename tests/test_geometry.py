import numpy as np

from swisher.geometry import angle_deg, trunk_lean_deg


def test_right_angle() -> None:
    value = angle_deg(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    assert value == 90.0


def test_straight_angle() -> None:
    value = angle_deg(np.array([-1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert value == 180.0


def test_vertical_trunk_has_zero_lean() -> None:
    assert trunk_lean_deg(np.array([0.0, 0.0]), np.array([0.0, 10.0])) == 0.0
