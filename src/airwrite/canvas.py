"""Clean grayscale canvas, independent of camera frames and interface overlays."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from airwrite.strokes import StrokeManager


class DrawingCanvas:
    """Render white ink on black for consistent training and inference polarity."""

    def __init__(self, width: int, height: int, strokes: StrokeManager, thickness: int = 5):
        self.width, self.height = width, height
        self.strokes, self.thickness = strokes, thickness

    def render(self) -> np.ndarray:
        """Render vector strokes, including a single-point dot."""
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        for stroke in self.strokes.strokes:
            points = np.array([(p.x, p.y) for p in stroke.points], dtype=np.int32)
            if len(points) == 1:
                cv2.circle(image, tuple(points[0]), max(1, self.thickness // 2), 255, -1)
            elif len(points) > 1:
                cv2.polylines(image, [points], False, 255, self.thickness, cv2.LINE_AA)
        return image

    def save(self, directory: Path) -> Path:
        """Save a unique clean PNG, never the camera or UI."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"airwrite_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}.png"
        if not cv2.imwrite(str(path), self.render()):
            raise OSError(f"Could not save drawing: {path}")
        return path
