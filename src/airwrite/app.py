"""Application orchestration; CV, interaction, editing, UI and ML remain separate."""

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from airwrite.camera import Camera
from airwrite.canvas import DrawingCanvas
from airwrite.collection import SampleCollector
from airwrite.config import AppConfig
from airwrite.gestures import FingerGestureDetector, HandPose, PinchDetector, ToolbarDwell
from airwrite.hand_tracker import HandTracker
from airwrite.interaction import InteractionController, InteractionState, ToolMode
from airwrite.metrics import RuntimeMetrics
from airwrite.recognizer import CharacterRecognizer
from airwrite.smoothing import PointSmoother
from airwrite.strokes import StrokeManager, StrokePoint
from airwrite.ui import KEY_ACTIONS, DisplayStatus, Toolbar, draw_debug

LOGGER = logging.getLogger(__name__)


class AirWriteApp:
    """Own one drawing session and dispatch explicit editing/recognition actions."""

    def __init__(self, config: AppConfig, collector: SampleCollector | None = None):
        self.config, self.collector = config, collector
        self.strokes = StrokeManager(
            config.drawing.minimum_point_distance, config.drawing.maximum_jump
        )
        self.interaction = InteractionController()
        self.smoother = PointSmoother(
            config.drawing.smoothing_alpha, config.drawing.smoothing_fast_distance
        )
        self.pinch = PinchDetector(config.gesture)
        self.fingers = FingerGestureDetector(config.gesture)
        self.dwell = ToolbarDwell(config.gesture.toolbar_dwell_seconds)
        self.pose = HandPose.HOVER
        self.cursor_anchor = 8
        self.recognizer = CharacterRecognizer(config.recognition.character_model_path)
        self.metrics = RuntimeMetrics()
        self.predictions = []
        self.prediction_revision = -1
        self.text = ""
        self.message = (
            f"Collecting {collector.label}: SAVE stores sample and clears." if collector else ""
        )
        self.canvas = None
        self.toolbar = None
        self.last_hand_time = 0.0
        self.previous_raw = None

    @property
    def message(self) -> str:
        """Latest action feedback; retained for logging/tests after the toast fades."""
        return self._message

    @message.setter
    def message(self, value: str):
        self._message = value
        self.message_until = time.monotonic() + 5.0

    def _invalidate_prediction(self):
        if self.predictions and self.prediction_revision != self.strokes.revision:
            self.predictions = []
            self.message = "Canvas changed. Recognize again before ADD."

    def action(self, action: str | None) -> bool:
        """Execute one command; return False only for quit."""
        if action is None:
            return True
        self.strokes.finish_stroke()
        try:
            if action == "QUIT":
                return False
            if action in ("PEN", "ERASER"):
                self.interaction.tool = ToolMode[action]
                self.config.gesture.control_mode = "pinch"
                self.message = "Pinch mode: pinch to use the tool. G returns to finger controls."
            elif action == "GESTURES":
                current = self.config.gesture.control_mode
                self.config.gesture.control_mode = "pinch" if current == "fingers" else "fingers"
                self.message = f"Control mode: {self.config.gesture.control_mode.upper()}"
                self.fingers = FingerGestureDetector(self.config.gesture)
            elif action == "UNDO":
                self.strokes.undo()
            elif action == "CLEAR":
                self.strokes.clear()
            elif action == "SAVE":
                path = (
                    self.collector.save(self.canvas)
                    if self.collector
                    else self.canvas.save(self.config.drawing.output_dir)
                )
                self.message = f"Saved {path.name}"
            elif action == "RECOGNIZE":
                self.predictions = self.recognizer.predict(self.canvas.render())
                self.prediction_revision = self.strokes.revision
                self.message = self.recognizer.status
                self.metrics.predictions.append(
                    {
                        "elapsed": time.monotonic() - self.metrics.started,
                        "top3": [asdict(p) for p in self.predictions],
                    }
                )
            elif action == "ADD":
                self._invalidate_prediction()
                if self.predictions:
                    self.text += self.predictions[0].character
                    self.strokes.clear()
                    self.predictions = []
                    self.message = "Character added. Write the next letter."
                else:
                    self.message = "Recognize a character before ADD."
            elif action == "SPACE":
                self.text += " "
            elif action == "BACKSPACE":
                self.text = self.text[:-1]
            elif action == "CLEAR TEXT":
                self.text = ""
            elif action == "DEBUG":
                self.config.debug = not self.config.debug
            self._invalidate_prediction()
        except (OSError, ValueError, RuntimeError) as error:
            self.message = str(error)
            LOGGER.warning("Action %s failed: %s", action, error)
        return True

    def process_hand(self, hand, width, height):
        """Advance interaction once; split immediately on tracking loss."""
        now = time.monotonic()
        event = self.pinch.update(hand, width, height)
        self.pose = self.fingers.update(hand, width, height, now)
        finger_mode = self.config.gesture.control_mode == "fingers"
        cursor = None
        if hand is not None:
            anchor = 9 if finger_mode and self.pose == HandPose.ERASE else 8
            if anchor != self.cursor_anchor:
                self.strokes.finish_stroke()
                self.smoother.reset()
                self.previous_raw = None
                self.cursor_anchor = anchor
            raw = hand.get_pixel_landmark(anchor, width, height)
            if self.previous_raw is not None:
                dx, dy = raw[0] - self.previous_raw[0], raw[1] - self.previous_raw[1]
                jump_limit = (
                    self.config.drawing.eraser_maximum_jump
                    if self.pose == HandPose.ERASE and finger_mode
                    else self.config.drawing.maximum_jump
                )
                if (dx * dx + dy * dy) ** 0.5 > jump_limit:
                    self.strokes.finish_stroke()
                    self.smoother.reset()
            if self.previous_raw is None:
                self.smoother.reset()
            # A fist follows the knuckle directly; pen EMA lag otherwise causes missed ink.
            cursor = (
                raw
                if finger_mode and self.pose == HandPose.ERASE
                else tuple(round(v) for v in self.smoother.update(raw))
            )
            self.previous_raw, self.last_hand_time = raw, now
        else:
            self.previous_raw = None
            if now - self.last_hand_time >= self.config.drawing.smoothing_reset_seconds:
                self.smoother.reset()
        over_ui = cursor is not None and (
            cursor[1] < self.toolbar.bottom
            or cursor[1] >= height - self.config.drawing.status_height
        )
        previous_state = self.interaction.state
        active = event.pinched
        if finger_mode:
            # A closed fist naturally resembles a pinch: do not let pinch cancel erasing.
            active = self.pose in (HandPose.WRITE, HandPose.ERASE)
            if self.pose == HandPose.WRITE:
                self.interaction.tool = ToolMode.PEN
            elif self.pose == HandPose.ERASE:
                self.interaction.tool = ToolMode.ERASER
        state = self.interaction.update(hand is not None, active, over_ui)
        if state != previous_state:
            self.strokes.finish_stroke()
        target = (
            self.toolbar.hit_test(cursor)
            if cursor is not None and finger_mode and self.pose == HandPose.POINTER
            else None
        )
        dwell_action = self.dwell.update(target, now)
        if event.started and over_ui and not finger_mode:
            self.action(self.toolbar.hit_test(cursor))
        elif dwell_action:
            self.action(dwell_action)
        if state == InteractionState.DRAWING:
            self.strokes.append_point(StrokePoint(*cursor, now))
        elif state == InteractionState.ERASING:
            self.strokes.erase(
                cursor,
                self.config.drawing.eraser_radius + self.config.drawing.pen_thickness / 2,
                self.config.drawing.eraser_maximum_jump,
            )
        else:
            self.strokes.finish_stroke()
        self._invalidate_prediction()
        return cursor, event

    def _display(self, frame, hand, cursor, event):
        # Quiet the camera feed slightly so handwriting and controls remain legible.
        frame = cv2.convertScaleAbs(frame, alpha=0.72)
        ink = self.canvas.render()
        frame[ink > 0] = (235, 250, 250)
        if self.config.debug and hand is not None:
            draw_debug(frame, hand, event)
        fps = self.metrics.tick(hand is not None)
        prediction = (
            f"{self.predictions[0].character}  {self.predictions[0].confidence:.0%}"
            if self.predictions
            else ""
        )
        message = self.message if time.monotonic() < self.message_until else ""
        if message == "Recognition complete.":
            message = ""
        if self.collector and not message:
            message = f"Collecting {self.collector.label} / Save to capture the next sample"
        status = DisplayStatus(
            state={
                InteractionState.NO_HAND: "Show your hand",
                InteractionState.HOVERING: "Ready",
                InteractionState.DRAWING: "Writing",
                InteractionState.ERASING: "Erasing",
            }[self.interaction.state],
            hand_present=hand is not None,
            prediction=prediction,
            text=self.text,
            message=message,
            finger_mode=self.config.gesture.control_mode == "fingers",
            debug=self.config.debug,
            fps=fps,
        )
        self.toolbar.draw(frame, cursor, status)
        if cursor is not None and self.dwell.progress > 0:
            cv2.ellipse(
                frame,
                cursor,
                (13, 13),
                -90,
                0,
                360 * self.dwell.progress,
                (80, 240, 190),
                3,
                cv2.LINE_AA,
            )
        if cursor is not None and self.interaction.state == InteractionState.ERASING:
            cv2.circle(frame, cursor, self.config.drawing.eraser_radius, (100, 170, 255), 1)
        return frame

    def run(self):
        """Run capture until Q/window close; always release native resources."""
        try:
            with HandTracker(self.config.tracking) as tracker, Camera(self.config.camera) as camera:
                cv2.namedWindow("AirWrite Vision", cv2.WINDOW_AUTOSIZE)
                while True:
                    frame = camera.read()
                    height, width = frame.shape[:2]
                    if self.canvas is None:
                        self.toolbar = Toolbar(width, self.config.drawing, height)
                        if height <= (self.toolbar.bottom + self.config.drawing.status_height):
                            raise RuntimeError("Camera resolution is too short for the toolbar.")
                        self.canvas = DrawingCanvas(
                            width, height, self.strokes, self.config.drawing.pen_thickness
                        )
                    elif (width, height) != (self.canvas.width, self.canvas.height):
                        raise RuntimeError(
                            "Camera resolution changed during capture. Restart the app."
                        )
                    hand = tracker.detect(frame)
                    cursor, event = self.process_hand(hand, width, height)
                    cv2.imshow("AirWrite Vision", self._display(frame, hand, cursor, event))
                    key = cv2.waitKey(1) & 0xFF
                    if not self.action(KEY_ACTIONS.get(key)):
                        break
                    if cv2.getWindowProperty("AirWrite Vision", cv2.WND_PROP_VISIBLE) < 1:
                        break
        finally:
            self.strokes.finish_stroke()
            cv2.destroyAllWindows()
            self._save_metrics()

    def _save_metrics(self):
        if not (self.metrics.detected_frames + self.metrics.lost_hand_frames):
            return
        try:
            directory = self.config.recognition.output_dir
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"session_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
            path.write_text(
                json.dumps(self.metrics.summary(self.strokes), indent=2), encoding="utf-8"
            )
        except OSError as error:
            LOGGER.warning("Cannot save session statistics: %s", error)


def main(argv=None):
    """Parse tuning options and launch the desktop app."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hand-model", type=Path, default=Path("models/hand_landmarker.task"))
    parser.add_argument("--character-model", type=Path, default=Path("models/airwrite_cnn.pt"))
    parser.add_argument("--pinch-start", type=float, default=0.35)
    parser.add_argument("--pinch-release", type=float, default=0.45)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--control-mode", choices=("fingers", "pinch"), default="fingers")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = AppConfig(debug=args.debug)
    config.camera.camera_index = args.camera
    config.tracking.model_path = args.hand_model
    config.recognition.character_model_path = args.character_model
    config.gesture.pinch_start_ratio = args.pinch_start
    config.gesture.pinch_release_ratio = args.pinch_release
    config.drawing.smoothing_alpha = args.alpha
    config.gesture.control_mode = args.control_mode
    try:
        config.gesture.__post_init__()
        config.drawing.__post_init__()
        AirWriteApp(config).run()
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"AirWrite: {error}\n")
    except KeyboardInterrupt:
        LOGGER.info("Stopped.")


if __name__ == "__main__":
    main()
