"""Paired clean raster and trajectory capture."""

import json
from pathlib import Path
from string import ascii_uppercase

import cv2

from airwrite.preprocessing import preprocess_canvas


class SampleCollector:
    """Write indexed sample pairs without overwriting previous sessions."""

    def __init__(self, label: str, root: Path = Path("data/custom")):
        if label not in ascii_uppercase or len(label) != 1:
            raise ValueError("Sample label must be a single uppercase A-Z character")
        self.label = label
        self.directory = root / label

    def save(self, canvas) -> Path:
        """Save full-resolution ink and matching pixel trajectories, then clear."""
        image = canvas.render()
        if preprocess_canvas(image) is None:
            raise ValueError("Nothing to save. Write a character first.")
        self.directory.mkdir(parents=True, exist_ok=True)
        index = 1
        while any(
            (self.directory / f"{self.label}_{index:04d}{suffix}").exists()
            for suffix in (".png", ".json")
        ):
            index += 1
        path = self.directory / f"{self.label}_{index:04d}.png"
        trajectory = {
            "label": self.label,
            "width": canvas.width,
            "height": canvas.height,
            "strokes": canvas.strokes.export_trajectory(),
        }
        json_path = path.with_suffix(".json")
        try:
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Could not write {path}")
            json_path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
        except OSError:
            path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
            raise
        canvas.strokes.clear()
        return path
