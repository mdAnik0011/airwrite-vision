"""OpenCV toolbar, status panel and optional landmark diagnostics."""

from dataclasses import dataclass

import cv2

from airwrite.config import DrawingConfig

CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
KEY_ACTIONS = {
    ord(key): action
    for key, action in [
        ("p", "PEN"),
        ("e", "ERASER"),
        ("u", "UNDO"),
        ("c", "CLEAR"),
        ("r", "RECOGNIZE"),
        ("s", "SAVE"),
        ("a", "ADD"),
        (" ", "SPACE"),
        ("x", "CLEAR TEXT"),
        ("d", "DEBUG"),
        ("g", "GESTURES"),
        ("q", "QUIT"),
    ]
}
KEY_ACTIONS.update({8: "BACKSPACE", 127: "BACKSPACE", 13: "ADD"})


@dataclass(frozen=True)
class Button:
    """Action and its pixel hit region."""

    action: str
    bounds: tuple[int, int, int, int]

    def contains(self, point):
        x, y, width, height = self.bounds
        return x <= point[0] < x + width and y <= point[1] < y + height


class Toolbar:
    """Four essential actions with a quiet header and compact result area."""

    def __init__(self, width: int, config: DrawingConfig, height: int = 480):
        self.config = config
        self.buttons = []
        rows = [("UNDO", "CLEAR", "RECOGNIZE", "SAVE")]
        # Keep the user's whole hand away from frame edges while reaching controls.
        top = round(config.toolbar_top_margin * height / 480)
        margin = round(config.toolbar_side_margin * width / 640)
        usable_width = width - 2 * margin
        row_height = max(1, round(config.toolbar_height * height / 480))
        self.bottom = top + row_height * len(rows)
        for row, actions in enumerate(rows):
            for column, action in enumerate(actions):
                left = margin + column * usable_width // len(actions)
                right = margin + (column + 1) * usable_width // len(actions)
                self.buttons.append(
                    Button(action, (left, top + row * row_height, right - left, row_height))
                )

    def hit_test(self, point):
        return next((b.action for b in self.buttons if b.contains(point)), None)

    def draw(self, frame, cursor, status: "DisplayStatus"):
        """Render essentials only; diagnostic details appear when debug is enabled."""
        height, width = frame.shape[:2]
        scale = min(width / 640, height / 480)
        padding = max(8, round(20 * scale))
        background, muted, foreground = (28, 25, 23), (158, 151, 143), (242, 239, 233)
        accent = (179, 222, 111)
        header_height = round(50 * height / 480)
        cv2.rectangle(frame, (0, 0), (width, header_height), background, -1)
        _text(
            frame,
            "airwrite",
            (padding, round(30 * height / 480)),
            0.65 * scale,
            foreground,
            width // 2,
        )
        state_text = status.state + (f"  {status.fps:.0f} fps" if status.debug else "")
        state_width = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, 1)[0][0]
        state_x = max(width // 2, width - padding - state_width)
        cv2.circle(
            frame,
            (state_x - 12, round(25 * height / 480)),
            max(2, round(3 * scale)),
            accent if status.hand_present else muted,
            -1,
        )
        _text(
            frame,
            state_text,
            (state_x, round(29 * height / 480)),
            0.4 * scale,
            muted,
            width - state_x - padding,
        )
        for button in self.buttons:
            x, y, w, h = button.bounds
            hovered = cursor is not None and button.contains(cursor)
            color = accent if hovered else background
            _rounded_panel(frame, (x + 3, y + 2, w - 6, h - 4), color, max(2, round(8 * scale)))
            label = {"UNDO": "Undo", "CLEAR": "Clear", "RECOGNIZE": "Read", "SAVE": "Save"}[
                button.action
            ]
            font_scale = 0.46 * scale
            text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
            _text(
                frame,
                label,
                (x + (w - text_width) // 2, y + h // 2 + round(5 * scale)),
                font_scale,
                background if hovered else foreground,
                w - 12,
            )
        top = height - self.config.status_height
        cv2.rectangle(frame, (0, top), (width, height), background, -1)
        cv2.line(frame, (padding, top), (width - padding, top), (57, 50, 44), 1)
        if status.message:
            primary = status.message
        elif status.prediction:
            primary = f"Read: {status.prediction}    A / Enter to add"
        elif status.text:
            primary = "Text: " + status.text[-45:]
        else:
            primary = "Write a letter, then choose Read"
        if status.prediction and status.text and not status.message:
            primary = f"{status.text[-30:]}   + {status.prediction}   |   Enter to add"
        _text(frame, primary, (padding, top + 24), 0.44 * scale, foreground, width - 2 * padding)
        hint = (
            "2 fingers write / 1 selects / fist erases"
            if status.finger_mode
            else "Pinch to use tool / G for finger controls"
        )
        _text(frame, hint, (padding, height - 13), 0.34 * scale, muted, width - 90)
        _text(
            frame,
            "Q quit",
            (width - padding - round(40 * scale), height - 13),
            0.34 * scale,
            muted,
            60,
        )
        if cursor is not None:
            cv2.circle(frame, cursor, 6, accent, 2, cv2.LINE_AA)


@dataclass(frozen=True)
class DisplayStatus:
    """Presentation data without exposing internal state-machine names."""

    state: str = "Show your hand"
    hand_present: bool = False
    prediction: str = ""
    text: str = ""
    message: str = ""
    finger_mode: bool = True
    debug: bool = False
    fps: float = 0.0


def _text(frame, text, origin, scale, color, max_width):
    """Fit a single line with an ellipsis instead of spilling into adjacent UI."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, scale, 1)[0][0] > max_width:
        while text and cv2.getTextSize(text + "...", font, scale, 1)[0][0] > max_width:
            text = text[:-1]
        text += "..."
    cv2.putText(frame, text, origin, font, scale, color, 1, cv2.LINE_AA)


def _rounded_panel(frame, bounds, color, radius):
    """Small solid rounded panel without external assets or expensive compositing."""
    x, y, w, h = bounds
    radius = max(1, min(radius, w // 2, h // 2))
    cv2.rectangle(frame, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(frame, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx in (x + radius, x + w - radius):
        for cy in (y + radius, y + h - radius):
            cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)


def draw_debug(frame, hand, event):
    """Show all landmarks, bones and aspect-correct pinch measurements."""
    height, width = frame.shape[:2]
    points = [hand.get_pixel_landmark(i, width, height) for i in range(21)]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (160, 110, 60), 1)
    for index, point in enumerate(points):
        cv2.circle(frame, point, 5 if index == 8 else 2, (30, 220, 250), -1)
    text = f"tip={event.tip_distance:.1f} palm={event.palm_distance:.1f} ratio={event.ratio:.2f}"
    cv2.putText(
        frame, text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 255), 1, cv2.LINE_AA
    )
