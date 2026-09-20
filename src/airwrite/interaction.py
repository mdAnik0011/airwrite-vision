"""Explicit drawing states and gesture capture for toolbar interactions."""

from enum import Enum, auto


class InteractionState(Enum):
    """Mutually exclusive live interaction states."""

    NO_HAND = auto()
    HOVERING = auto()
    DRAWING = auto()
    ERASING = auto()


class ToolMode(Enum):
    """Persistent selected editing tool."""

    PEN = auto()
    ERASER = auto()


class InteractionController:
    """Capture UI pinches until release, including when the cursor leaves the UI."""

    def __init__(self):
        self.tool = ToolMode.PEN
        self.state = InteractionState.NO_HAND
        self.ui_captured = False

    def update(self, hand_present: bool, pinched: bool, over_ui: bool) -> InteractionState:
        if not hand_present or not pinched:
            self.ui_captured = False
        elif over_ui:
            self.ui_captured = True
        if not hand_present:
            self.state = InteractionState.NO_HAND
        elif not pinched or over_ui or self.ui_captured:
            self.state = InteractionState.HOVERING
        elif self.tool == ToolMode.PEN:
            self.state = InteractionState.DRAWING
        else:
            self.state = InteractionState.ERASING
        return self.state
