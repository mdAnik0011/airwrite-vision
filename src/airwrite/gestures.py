"""Scale-normalized pinch geometry and hysteresis."""

from dataclasses import dataclass
from enum import Enum
from math import acos, degrees, hypot

from airwrite.config import GestureConfig
from airwrite.hand_tracker import HandObservation


class HandPose(Enum):
    """Direct hand commands; unrecognized shapes safely hover."""

    HOVER = "REST"
    WRITE = "2 FINGERS: WRITE"
    POINTER = "1 FINGER: POINTER"
    ERASE = "CLOSED FIST: ERASE"


class FingerGestureDetector:
    """Rotation/mirror-independent joint geometry with delayed action entry."""

    def __init__(self, config: GestureConfig):
        self.config = config
        self.candidate = HandPose.HOVER
        self.accepted = HandPose.HOVER
        self.since = 0.0

    def classify(self, hand: HandObservation | None, width: int, height: int) -> HandPose:
        """Use straight joints and wrist-relative reach, never screen-up tests."""
        if hand is None:
            return HandPose.HOVER
        points = [(p.x * width, p.y * height) for p in hand.landmarks]

        def distance(a, b):
            return hypot(points[a][0] - points[b][0], points[a][1] - points[b][1])

        def angle(a, b, c):
            ux, uy = points[a][0] - points[b][0], points[a][1] - points[b][1]
            vx, vy = points[c][0] - points[b][0], points[c][1] - points[b][1]
            length = hypot(ux, uy) * hypot(vx, vy)
            if length < 1e-8:
                return 0.0
            return degrees(acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / length))))

        palm = distance(5, 17)
        if palm < 1e-8:
            return HandPose.HOVER
        threshold = (
            self.config.finger_release_angle
            if self.accepted == HandPose.WRITE
            else self.config.finger_straight_angle
        )
        extended = [
            angle(base, base + 1, base + 3) >= threshold
            and distance(0, base + 3) > distance(0, base + 1) * self.config.finger_reach_ratio
            for base in (5, 9, 13, 17)
        ]
        curled = [
            angle(base, base + 1, base + 3) < self.config.fist_curl_angle
            and distance(0, base + 3) < distance(0, base + 1) * 1.15
            for base in (5, 9, 13, 17)
        ]
        thumb_tucked = distance(4, 9) < palm and distance(4, 5) < palm
        if all(curled) and thumb_tucked:
            return HandPose.ERASE
        if extended == [True, False, False, False]:
            return HandPose.POINTER
        if extended == [True, True, False, False]:
            return HandPose.WRITE
        return HandPose.HOVER

    def update(self, hand: HandObservation | None, width: int, height: int, now: float) -> HandPose:
        """Stop immediately on shape changes; resume only after a stable hold."""
        pose = self.classify(hand, width, height)
        if hand is None or pose != self.candidate:
            self.candidate, self.since = pose, now
        hold = (
            min(self.config.write_hold_seconds, self.config.pose_hold_seconds)
            if pose == HandPose.WRITE
            else self.config.pose_hold_seconds
        )
        if now - self.since < hold:
            self.accepted = HandPose.HOVER
            return HandPose.HOVER
        self.accepted = pose
        return pose


class ToolbarDwell:
    """Trigger once after pointer hovering; leave/release to repeat."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.target: str | None = None
        self.since = 0.0
        self.fired = False
        self.progress = 0.0

    def update(self, target: str | None, now: float) -> str | None:
        """Return a command on completion, with progress for a visible timer."""
        if target != self.target or target is None:
            self.target, self.since, self.fired = target, now, False
        self.progress = min(1.0, max(0.0, (now - self.since) / self.seconds)) if target else 0.0
        if target and not self.fired and self.progress >= 1:
            self.fired = True
            return target
        return None


@dataclass(frozen=True)
class PinchEvent:
    """Current gesture plus edges for single-activation controls."""

    pinched: bool = False
    started: bool = False
    released: bool = False
    ratio: float = float("inf")
    tip_distance: float = 0.0
    palm_distance: float = 0.0


class PinchDetector:
    """Measure in aspect-correct image coordinates, normalized by palm width."""

    def __init__(self, config: GestureConfig):
        self.config = config
        self.pinched = False

    def update(self, hand: HandObservation | None, width=1, height=1) -> PinchEvent:
        old = self.pinched
        tip = palm = 0.0
        ratio = float("inf")
        if hand is not None:

            def distance(a, b):
                p, q = hand.get_landmark(a), hand.get_landmark(b)
                return hypot((p.x - q.x) * width, (p.y - q.y) * height)

            tip, palm = distance(4, 8), distance(5, 17)
            if palm > 1e-8:
                ratio = tip / palm
        threshold = self.config.pinch_release_ratio if old else self.config.pinch_start_ratio
        self.pinched = ratio < threshold
        return PinchEvent(
            self.pinched, self.pinched and not old, old and not self.pinched, ratio, tip, palm
        )
