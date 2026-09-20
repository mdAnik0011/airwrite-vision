"""Shared training/runtime normalization: white foreground on a black background."""

import cv2
import numpy as np


def bounding_box(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return exclusive (left, top, right, bottom) ink bounds, or None."""
    ys, xs = np.nonzero(image > 8)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def preprocess_canvas(image: np.ndarray) -> np.ndarray | None:
    """Crop, pad, preserve aspect ratio and center ink in a float32 1x28x28 array."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 grayscale or BGR canvas")
    bounds = bounding_box(image)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right]
    pad = max(1, round(max(crop.shape) * 0.1))
    crop = np.pad(crop, pad)
    scale = 24 / max(crop.shape)
    w, h = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    output = np.zeros((28, 28), dtype=np.float32)
    x, y = (28 - w) // 2, (28 - h) // 2
    output[y : y + h, x : x + w] = resized / 255.0
    return output[np.newaxis, :, :]
