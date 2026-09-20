"""Finger geometry and app transitions without a camera."""

import math

import pytest
from test_app import make_app

from airwrite.config import GestureConfig
from airwrite.gestures import FingerGestureDetector, HandPose, ToolbarDwell
from airwrite.hand_tracker import HandObservation, Point3D
from airwrite.interaction import InteractionState
from airwrite.strokes import StrokePoint


def pose_hand(count, rotation=0, scale=1, mirror=False, offset=(0, 0)):
    points = [(320, 400)] * 21
    points[1:5] = [(270, 350), (240, 330), (210, 310), (180, 290)]
    for finger, base in enumerate((5, 9, 13, 17)):
        x = 260 + 40 * finger
        points[base] = (x, 280)
        points[base + 1] = (x, 240)
        points[base + 2] = (x, 210 if finger < count else 265)
        points[base + 3] = (x, 180 if finger < count else 295)
    if count == 0:
        points[4] = (270, 285)  # Thumb tucked across the fist.
    transformed = []
    for x, y in points:
        x, y = (x - 320) * scale * (-1 if mirror else 1), (y - 240) * scale
        a = x * math.cos(rotation) - y * math.sin(rotation)
        b = x * math.sin(rotation) + y * math.cos(rotation)
        transformed.append(Point3D((a + 320 + offset[0]) / 640, (b + 240 + offset[1]) / 480))
    return HandObservation(transformed)


@pytest.mark.parametrize(
    "rotation,scale,mirror", [(0, 1, False), (1.1, 0.5, True), (math.pi, 0.8, False)]
)
@pytest.mark.parametrize(
    "count,expected",
    [
        (0, HandPose.ERASE),
        (1, HandPose.POINTER),
        (2, HandPose.WRITE),
        (3, HandPose.HOVER),
        (4, HandPose.HOVER),
    ],
)
def test_shapes_are_rotation_scale_and_handedness_independent(
    count, expected, rotation, scale, mirror
):
    detector = FingerGestureDetector(GestureConfig())
    assert detector.classify(pose_hand(count, rotation, scale, mirror), 640, 480) == expected


def test_pose_entry_waits_and_transition_stops_immediately():
    detector = FingerGestureDetector(GestureConfig(pose_hold_seconds=0.12))
    assert detector.update(pose_hand(2), 640, 480, 0) == HandPose.HOVER
    assert detector.update(pose_hand(2), 640, 480, 0.13) == HandPose.WRITE
    assert detector.update(pose_hand(0), 640, 480, 0.14) == HandPose.HOVER
    assert detector.update(pose_hand(0), 640, 480, 0.27) == HandPose.ERASE
    assert detector.update(None, 640, 480, 0.28) == HandPose.HOVER
    assert detector.update(pose_hand(0), 640, 480, 0.29) == HandPose.HOVER


def test_dwell_fires_once_and_resets_when_leaving():
    dwell = ToolbarDwell(0.8)
    assert dwell.update("UNDO", 0) is None
    assert dwell.update("UNDO", 0.4) is None
    assert dwell.progress == pytest.approx(0.5)
    assert dwell.update("UNDO", 0.81) == "UNDO"
    assert dwell.update("UNDO", 2) is None
    assert dwell.update(None, 3) is None
    assert dwell.update("UNDO", 4) is None
    assert dwell.update("UNDO", 4.81) == "UNDO"


def finger_app():
    app = make_app()
    app.config.gesture.control_mode = "fingers"
    app.config.gesture.pose_hold_seconds = 0
    return app


def test_two_write_one_hovers_fist_erases_and_undo_restores():
    app = finger_app()
    app.process_hand(pose_hand(2), 640, 480)
    assert app.interaction.state == InteractionState.DRAWING
    assert len(app.strokes.strokes) == 1
    app.process_hand(pose_hand(1), 640, 480)
    assert app.interaction.state == InteractionState.HOVERING
    assert app.strokes.active is None
    app.strokes.begin_stroke(StrokePoint(300, 280, 0))
    app.strokes.finish_stroke()
    cursor, _ = app.process_hand(pose_hand(0), 640, 480)
    assert cursor == (300, 280)
    assert app.interaction.state == InteractionState.ERASING
    assert len(app.strokes.strokes) == 1
    app.action("UNDO")
    assert len(app.strokes.strokes) == 2


def test_one_finger_toolbar_activation_and_no_ink(monkeypatch):
    app = finger_app()
    now = [0.0]
    monkeypatch.setattr("airwrite.app.time.monotonic", lambda: now[0])
    actions = []
    original = app.action

    def record(action):
        actions.append(action)
        return original(action)

    app.action = record
    button = next(b for b in app.toolbar.buttons if b.action == "CLEAR")
    x, y, width, height = button.bounds
    # Move the index tip from (260, 180) to the inset button's center.
    hand = pose_hand(1, offset=(x + width // 2 - 260, y + height // 2 - 180))
    app.process_hand(hand, 640, 480)
    now[0] = 0.81
    app.process_hand(hand, 640, 480)
    now[0] = 2
    app.process_hand(hand, 640, 480)
    assert actions == ["CLEAR"]
    assert not app.strokes.strokes


def test_switch_back_to_finger_mode():
    app = finger_app()
    app.action("ERASER")
    assert app.config.gesture.control_mode == "pinch"
    app.action("GESTURES")
    assert app.config.gesture.control_mode == "fingers"


def test_fist_pinching_does_not_disable_eraser_or_click_toolbar():
    app = finger_app()
    _, event = app.process_hand(pose_hand(0), 640, 480)
    assert event.pinched
    assert app.interaction.state == InteractionState.ERASING
    app.process_hand(None, 640, 480)
    app.process_hand(pose_hand(0, offset=(64, -195)), 640, 480)
    assert app.config.gesture.control_mode == "fingers"
    assert app.interaction.state == InteractionState.HOVERING


def test_fist_sweep_clears_fast_motion_but_not_across_tracking_loss():
    for lose_hand in (True, False):
        app = finger_app()
        app.config.drawing.smoothing_alpha = 0.1
        app.strokes.begin_stroke(StrokePoint(350, 280, 0))
        app.strokes.finish_stroke()
        app.process_hand(pose_hand(0), 640, 480)
        if lose_hand:
            app.process_hand(None, 640, 480)
        cursor, _ = app.process_hand(pose_hand(0, offset=(100, 0)), 640, 480)
        assert cursor == (400, 280)
        assert bool(app.strokes.strokes) == lose_hand


def test_open_palm_and_thumb_out_do_not_erase():
    detector = FingerGestureDetector(GestureConfig())
    assert detector.classify(pose_hand(4), 640, 480) == HandPose.HOVER
    hand = pose_hand(0)
    hand.landmarks[4] = Point3D(180 / 640, 290 / 480)
    assert detector.classify(hand, 640, 480) == HandPose.HOVER


def test_fast_writing_stays_connected_and_tracks_corners():
    app = finger_app()
    app.smoother.alpha = 0.35
    for offset in ((0, 0), (110, 0), (110, 100), (0, 100)):
        app.process_hand(pose_hand(2, offset=offset), 640, 480)
    assert len(app.strokes.strokes) == 1
    assert [(p.x, p.y) for p in app.strokes.strokes[0].points] == [
        (260, 180),
        (370, 180),
        (370, 280),
        (260, 280),
    ]
    app.process_hand(pose_hand(1, offset=(0, 100)), 640, 480)
    assert app.strokes.active is None
    app.process_hand(pose_hand(2, offset=(10, 100)), 640, 480)
    assert len(app.strokes.strokes) == 2


def test_fast_writing_still_breaks_on_tracking_loss_and_extreme_jump():
    app = finger_app()
    app.process_hand(pose_hand(2), 640, 480)
    app.process_hand(pose_hand(2, offset=(300, 0)), 640, 480)
    assert len(app.strokes.strokes) == 2
    app.process_hand(None, 640, 480)
    app.process_hand(pose_hand(2, offset=(290, 0)), 640, 480)
    assert len(app.strokes.strokes) == 3


def test_writing_starts_quickly_and_tolerates_slight_finger_bend():
    detector = FingerGestureDetector(GestureConfig())
    assert detector.update(pose_hand(2), 640, 480, 0) == HandPose.HOVER
    assert detector.update(pose_hand(2), 640, 480, 0.04) == HandPose.WRITE
    bent = pose_hand(2)
    bent.landmarks[12] = Point3D(334 / 640, 190 / 480)
    assert detector.update(bent, 640, 480, 0.06) == HandPose.WRITE
    assert detector.update(pose_hand(1), 640, 480, 0.07) == HandPose.HOVER
    assert detector.update(pose_hand(1), 640, 480, 0.20) == HandPose.POINTER
