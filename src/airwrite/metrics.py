"""Rolling FPS and anonymous session statistics."""

import time
from collections import deque


class RuntimeMetrics:
    """Count frames and predictions without retaining camera images."""

    def __init__(self, window: int = 30):
        self.times = deque(maxlen=window + 1)
        self.started = time.monotonic()
        self.detected_frames = 0
        self.lost_hand_frames = 0
        self.predictions: list[dict] = []

    def tick(self, detected: bool) -> float:
        self.times.append(time.monotonic())
        self.detected_frames += int(detected)
        self.lost_hand_frames += int(not detected)
        elapsed = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0

    def summary(self, strokes) -> dict:
        """Produce JSON-compatible statistics for technical evaluation."""
        count = len(strokes.strokes)
        return {
            "average_fps": (self.detected_frames + self.lost_hand_frames)
            / max(1e-9, time.monotonic() - self.started),
            "detected_frames": self.detected_frames,
            "lost_hand_frames": self.lost_hand_frames,
            "number_of_strokes": count,
            "average_points_per_stroke": sum(len(s.points) for s in strokes.strokes)
            / max(1, count),
            "predictions": self.predictions,
        }
