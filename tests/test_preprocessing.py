import numpy as np

from airwrite.preprocessing import bounding_box, preprocess_canvas


def test_empty():
    image = np.zeros((480, 640), dtype=np.uint8)
    assert bounding_box(image) is None
    assert preprocess_canvas(image) is None


def test_bounds_shape_polarity_and_aspect_ratio():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:80, 40:50] = 255
    assert bounding_box(image) == (40, 20, 50, 80)
    result = preprocess_canvas(image)
    assert result.shape == (1, 28, 28)
    assert result.dtype == np.float32
    assert result.max() <= 1 and result.min() == 0
    ys, xs = np.nonzero(result[0] > 0.1)
    assert np.ptp(ys) > np.ptp(xs) * 3
    assert abs((xs.min() + xs.max()) / 2 - 13.5) <= 1


def test_single_dot_is_valid():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[5, 5] = 255
    assert preprocess_canvas(image).sum() > 0
