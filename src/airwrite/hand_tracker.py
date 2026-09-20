"""MediaPipe Tasks VIDEO adapter; consumers use ordinary dataclasses."""

import time
from dataclasses import dataclass

import cv2

from airwrite.config import HandTrackingConfig


@dataclass(frozen=True)
class Point3D:
    """Normalized image-space landmark."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class HandObservation:
    """One tracked hand, isolated from MediaPipe result types."""

    landmarks: list[Point3D]
    handedness: str | None = None

    def get_landmark(self, index: int) -> Point3D:
        return self.landmarks[index]

    def get_pixel_landmark(self, index: int, width: int, height: int) -> tuple[int, int]:
        point = self.get_landmark(index)
        return (
            max(0, min(width - 1, round(point.x * width))),
            max(0, min(height - 1, round(point.y * height))),
        )


class HandTracker:
    """Own a HandLandmarker with strictly increasing millisecond timestamps."""

    def __init__(self, config: HandTrackingConfig):
        if not config.model_path.is_file():
            raise FileNotFoundError(
                f"Missing {config.model_path}. Run: python scripts/download_models.py"
            )
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.task = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(config.model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=config.num_hands,
                min_hand_detection_confidence=config.min_detection_confidence,
                min_hand_presence_confidence=config.min_presence_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
        )
        self.last_timestamp = -1

    def detect(self, frame) -> HandObservation | None:
        """Detect the first hand in a BGR frame."""
        timestamp = max(self.last_timestamp + 1, time.monotonic_ns() // 1_000_000)
        self.last_timestamp = timestamp
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = self.task.detect_for_video(image, timestamp)
        if not result.hand_landmarks:
            return None
        label = result.handedness[0][0].category_name if result.handedness else None
        return HandObservation([Point3D(p.x, p.y, p.z) for p in result.hand_landmarks[0]], label)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.task.close()
