import pytest

from airwrite.smoothing import PointSmoother


def test_smoothing_and_reset():
    smoother = PointSmoother(0.5)
    assert smoother.update((10, 20)) == (10, 20)
    assert smoother.update((20, 40)) == (15, 30)
    smoother.reset()
    assert smoother.update((100, 100)) == (100, 100)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        PointSmoother(0)


def test_adaptive_smoothing_reduces_jitter_but_follows_fast_motion():
    smoother = PointSmoother(0.35, fast_distance=24)
    smoother.update((0, 0))
    small = smoother.update((2, 0))
    assert 0 < small[0] < 2
    assert smoother.update((120, 0)) == (120, 0)
    assert smoother.update((120, 120)) == (120, 120)
    smoother.reset()
    assert smoother.update((350, 50)) == (350, 50)
