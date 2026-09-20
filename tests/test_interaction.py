from airwrite.interaction import InteractionController, ToolMode
from airwrite.interaction import InteractionState as S


def test_state_transitions():
    controller = InteractionController()
    assert controller.update(False, False, False) == S.NO_HAND
    assert controller.update(True, False, False) == S.HOVERING
    assert controller.update(True, True, False) == S.DRAWING
    assert controller.update(True, False, False) == S.HOVERING
    controller.tool = ToolMode.ERASER
    assert controller.update(True, True, False) == S.ERASING
    assert controller.update(False, True, False) == S.NO_HAND


def test_toolbar_pinch_captured_until_release():
    controller = InteractionController()
    assert controller.update(True, True, True) == S.HOVERING
    assert controller.update(True, True, False) == S.HOVERING
    controller.update(True, False, False)
    assert controller.update(True, True, False) == S.DRAWING
