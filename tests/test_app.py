"""Interaction integration tests using synthetic landmarks, never a webcam."""

from pathlib import Path

import numpy as np

from airwrite.app import AirWriteApp
from airwrite.canvas import DrawingCanvas
from airwrite.config import AppConfig
from airwrite.hand_tracker import HandObservation, Point3D
from airwrite.recognizer import Prediction
from airwrite.ui import Toolbar


def make_app():
    config = AppConfig()
    config.gesture.control_mode = "pinch"
    config.recognition.character_model_path = Path("absent-test-model.pt")
    config.drawing.smoothing_alpha = 1
    app = AirWriteApp(config)
    app.canvas = DrawingCanvas(640, 480, app.strokes)
    app.toolbar = Toolbar(640, config.drawing)
    return app


def observation(x, y, pinched=True):
    points = [Point3D(0.5, 0.5) for _ in range(21)]
    points[5], points[17] = Point3D(0.4, 0.5), Point3D(0.6, 0.5)
    points[8] = Point3D(x / 640, y / 480)
    points[4] = Point3D(x / 640 + (0.02 if pinched else 0.15), y / 480)
    return HandObservation(points)


def test_hover_release_loss_and_reacquisition():
    app = make_app()
    app.process_hand(observation(200, 200, False), 640, 480)
    assert not app.strokes.strokes
    app.process_hand(observation(200, 200), 640, 480)
    app.process_hand(observation(210, 210), 640, 480)
    app.process_hand(None, 640, 480)
    assert app.strokes.active is None
    app.process_hand(observation(220, 220), 640, 480)
    assert len(app.strokes.strokes) == 2
    app.process_hand(observation(230, 230, False), 640, 480)
    assert app.strokes.active is None


def test_toolbar_clear_fires_once_and_does_not_draw_when_dragged_out():
    app = make_app()
    actions = []
    original = app.action

    def record(action):
        actions.append(action)
        return original(action)

    app.action = record
    button = next(b for b in app.toolbar.buttons if b.action == "CLEAR")
    x, y, width, height = button.bounds
    center = (x + width // 2, y + height // 2)
    app.process_hand(observation(*center), 640, 480)
    app.process_hand(observation(*center), 640, 480)
    app.process_hand(observation(center[0], 200), 640, 480)
    assert actions == ["CLEAR"]
    assert not app.strokes.strokes
    app.process_hand(observation(center[0], 200, False), 640, 480)
    app.process_hand(observation(center[0], 200), 640, 480)
    assert len(app.strokes.strokes) == 1


def test_text_commit_and_stale_prediction():
    app = make_app()
    app.process_hand(observation(200, 200), 640, 480)
    app.predictions = [Prediction("A", 0.9)]
    app.prediction_revision = app.strokes.revision
    app.action("ADD")
    assert app.text == "A" and not app.strokes.strokes
    app.action("SPACE")
    app.action("BACKSPACE")
    assert app.text == "A"
    app.predictions = [Prediction("B", 0.9)]
    app.prediction_revision = app.strokes.revision
    app.process_hand(observation(210, 210), 640, 480)
    app.action("ADD")
    assert app.text == "A"
    app.action("CLEAR TEXT")
    assert app.text == ""


def test_display_never_contaminates_clean_canvas():
    app = make_app()
    hand = observation(200, 200)
    cursor, event = app.process_hand(hand, 640, 480)
    before = app.canvas.render()
    app._display(np.full((480, 640, 3), 125, dtype=np.uint8), hand, cursor, event)
    assert np.array_equal(app.canvas.render(), before)
    assert not before[:86].any()
