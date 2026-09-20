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
    """Two rows provide gesture access to editing and text composition."""

    def __init__(self, width: int, config: DrawingConfig, height: int = 480):
        self.config = config
        self.buttons = []
        rows = [
            ("PEN", "ERASER", "UNDO", "CLEAR", "RECOGNIZE", "SAVE"),
            ("ADD", "SPACE", "BACKSPACE", "CLEAR TEXT", "GESTURES"),
        ]
        # Keep the user's whole hand away from frame edges while reaching controls.
        top = round(config.toolbar_top_margin * height / 480)
        margin = round(config.toolbar_side_margin * width / 640)
        usable_width = width - 2 * margin
        row_height = max(1, round(config.toolbar_height * height / 480) // 2)
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

    def draw(self, frame, cursor, tool, status: list[str]):
        """Render only onto the supplied display frame."""
        height, width = frame.shape[:2]
        for button in self.buttons:
            x, y, w, h = button.bounds
            selected = button.action == tool.name
            hovered = cursor is not None and button.contains(cursor)
            color = (95, 85, 35) if selected else (42, 35, 29)
            cv2.rectangle(frame, (x + 2, y + 2), (x + w - 2, y + h - 2), color, -1)
            if hovered:
                cv2.rectangle(frame, (x + 2, y + 2), (x + w - 2, y + h - 2), (100, 230, 210), 1)
            scale = min(0.43, (w - 10) / max(1, len(button.action) * 20))
            cv2.putText(
                frame,
                button.action,
                (x + 6, y + h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        top = height - self.config.status_height
        cv2.rectangle(frame, (0, top), (width, height), (28, 24, 20), -1)
        for index, line in enumerate(status[:4]):
            cv2.putText(
                frame,
                line[:95],
                (10, top + 19 + index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (220, 230, 230),
                1,
                cv2.LINE_AA,
            )
        if cursor is not None:
            cv2.circle(frame, cursor, 7, (80, 240, 190), 2, cv2.LINE_AA)


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
