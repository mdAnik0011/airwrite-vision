import pytest

from airwrite.config import GestureConfig
from airwrite.gestures import PinchDetector
from airwrite.hand_tracker import HandObservation, Point3D


def hand(ratio, scale=1):
    points = [Point3D(0, 0) for _ in range(21)]
    points[17] = Point3D(scale, 0)
    points[8] = Point3D(ratio * scale, 0)
    return HandObservation(points)


def test_normalization_and_hysteresis():
    detector = PinchDetector(GestureConfig())
    event = detector.update(hand(0.3, 0.2), 640, 480)
    assert event.started and event.ratio == pytest.approx(0.3)
    assert detector.update(hand(0.4)).pinched
    assert detector.update(hand(0.46)).released
    assert not detector.update(hand(0.4)).pinched
    assert detector.update(hand(0.3, 0.7)).started
    assert detector.update(None).released


def test_degenerate_palm():
    assert not PinchDetector(GestureConfig()).update(hand(0, 0)).pinched
