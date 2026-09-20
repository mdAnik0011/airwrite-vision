import json

import cv2
import numpy as np
import pytest

from airwrite.canvas import DrawingCanvas
from airwrite.collection import SampleCollector
from airwrite.strokes import StrokeManager, StrokePoint


def test_paired_samples_and_unique_names(tmp_path):
    strokes = StrokeManager()
    canvas = DrawingCanvas(100, 100, strokes)
    collector = SampleCollector("A", tmp_path)
    for index in (1, 2):
        strokes.begin_stroke(StrokePoint(20, 20, 10))
        strokes.append_point(StrokePoint(40, 40, 10.1))
        clean = canvas.render()
        path = collector.save(canvas)
        assert path.name == f"A_{index:04d}.png"
        assert np.array_equal(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), clean)
        data = json.loads(path.with_suffix(".json").read_text())
        assert data["label"] == "A"
        assert data["strokes"][0][0] == {"x": 20, "y": 20, "t": 0}
        assert not strokes.strokes
    with pytest.raises(ValueError, match="Nothing to save"):
        collector.save(canvas)


def test_png_export_contains_only_ink(tmp_path):
    strokes = StrokeManager()
    strokes.begin_stroke(StrokePoint(40, 40, 0))
    canvas = DrawingCanvas(80, 80, strokes)
    first, second = canvas.save(tmp_path), canvas.save(tmp_path)
    assert first != second
    image = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (80, 80)
    assert image[0, 0] == 0 and image[40, 40] == 255
