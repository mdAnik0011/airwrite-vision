"""Replaceable point filtering."""

from math import hypot


class PointSmoother:
    """Exponential moving average in pixel coordinates."""

    def __init__(self, alpha: float = 0.35, fast_distance: float | None = None):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        if fast_distance is not None and fast_distance <= 0:
            raise ValueError("fast_distance must be positive")
        self.fast_distance = fast_distance
        self.previous: tuple[float, float] | None = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        """Return the first point unchanged, then smooth subsequent points."""
        if self.previous is None:
            self.previous = point
        else:
            alpha = self.alpha
            if self.fast_distance is not None:
                distance = hypot(point[0] - self.previous[0], point[1] - self.previous[1])
                alpha += (1 - alpha) * min(1.0, distance / self.fast_distance)
            self.previous = tuple(
                alpha * x + (1 - alpha) * old for x, old in zip(point, self.previous, strict=True)
            )
        return self.previous

    def reset(self):
        """Forget the previous track."""
        self.previous = None
