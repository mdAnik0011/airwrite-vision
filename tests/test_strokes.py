from airwrite.strokes import StrokeManager
from airwrite.strokes import StrokePoint as P


def test_swept_eraser_removes_ink_between_frames_and_preserves_outside():
    strokes = StrokeManager(maximum_jump=200)
    strokes.begin_stroke(P(50, -40, 0))
    strokes.append_point(P(50, 40, 1))
    strokes.finish_stroke()
    strokes.erase((0, 0), 10)
    strokes.erase((100, 0), 10)
    assert len(strokes.strokes) == 2
    assert strokes.strokes[0].points[-1].y == -10
    assert strokes.strokes[1].points[0].y == 10
    strokes.finish_stroke()
    strokes.undo()
    assert len(strokes.strokes) == 1
    assert strokes.strokes[0].points == [P(50, -40, 0), P(50, 40, 1)]


def test_sweep_covers_dots_at_edge_without_sampling_gaps():
    strokes = StrokeManager()
    strokes.begin_stroke(P(51, 9, 0))
    strokes.finish_stroke()
    strokes.erase((0, 0), 10)
    strokes.erase((100, 0), 10)
    assert not strokes.strokes


def test_sweep_does_not_bridge_gestures_or_large_jumps():
    for end_gesture in (True, False):
        strokes = StrokeManager()
        strokes.begin_stroke(P(50, 0, 0))
        strokes.finish_stroke()
        strokes.erase((0, 0), 10)
        if end_gesture:
            strokes.finish_stroke()
        strokes.erase((100, 0), 10, maximum_jump=40 if not end_gesture else 240)
        assert len(strokes.strokes) == 1


def test_lifecycle_jitter_jump_and_undo():
    strokes = StrokeManager()
    strokes.begin_stroke(P(0, 0, 1))
    strokes.append_point(P(1, 1, 2))
    assert len(strokes.active.points) == 1
    strokes.append_point(P(10, 0, 3))
    strokes.append_point(P(200, 0, 4))
    assert len(strokes.strokes) == 2
    strokes.finish_stroke()
    assert strokes.active is None
    assert strokes.export_trajectory()[0][0]["t"] == 0
    strokes.undo()
    assert not strokes.strokes


def test_clear_can_be_undone():
    strokes = StrokeManager()
    strokes.begin_stroke(P(10, 10, 0))
    strokes.clear()
    assert not strokes.strokes
    strokes.undo()
    assert len(strokes.strokes) == 1


def test_eraser_intersects_between_endpoints_and_undoes_gesture():
    strokes = StrokeManager(maximum_jump=200)
    strokes.begin_stroke(P(0, 20, 0))
    strokes.append_point(P(100, 20, 1))
    strokes.finish_stroke()
    strokes.erase((50, 20), 10)
    assert len(strokes.strokes) == 2
    assert strokes.strokes[0].points[-1].x == 40
    assert strokes.strokes[1].points[0].x == 60
    strokes.erase((70, 20), 10)
    strokes.finish_stroke()
    strokes.undo()
    assert len(strokes.strokes) == 1
    assert strokes.strokes[0].points[-1].x == 100
