"""Mirrored webcam capture with deterministic resource cleanup."""

import cv2
import numpy as np

from airwrite.config import CameraConfig


class Camera:
    """Own a VideoCapture; use as a context manager."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture = None
        self.width = config.frame_width
        self.height = config.frame_height

    def __enter__(self):
        self.capture = cv2.VideoCapture(self.config.camera_index)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(
                "Cannot open webcam. Check camera index, permissions and other apps."
            )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Best-effort requests; actual support depends on the camera/backend.
        self.capture.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self

    def read(self) -> np.ndarray:
        """Read a mirrored BGR frame and update actual dimensions."""
        if self.capture is None:
            raise RuntimeError("Camera is not open")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam stopped delivering frames. Check the camera connection.")
        self.height, self.width = frame.shape[:2]
        return cv2.flip(frame, 1)

    def close(self):
        """Release the device, including after failed initialization."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __exit__(self, *_):
        self.close()
