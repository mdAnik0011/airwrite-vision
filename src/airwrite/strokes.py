"""Vector trajectories, jump segmentation and undoable geometric erasing."""

from copy import deepcopy
from dataclasses import dataclass, field
from math import hypot, sqrt


@dataclass(frozen=True)
class StrokePoint:
    """Pixel position and monotonic timestamp in seconds."""

    x: int
    y: int
    timestamp: float


@dataclass
class Stroke:
    """One uninterrupted section of handwriting."""

    points: list[StrokePoint] = field(default_factory=list)


def _distance_to_sweep(point, start, end):
    """Distance to the closest point on the eraser's travel segment."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length2 = dx * dx + dy * dy
    t = (
        max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2))
        if length2
        else 0.0
    )
    return hypot(point[0] - start[0] - t * dx, point[1] - start[1] - t * dy)


def _outside_segments(a: StrokePoint, b: StrokePoint, center, radius, previous=None):
    """Clip against a swept disk (capsule), retaining exact outside intervals."""
    previous = center if previous is None else previous
    dx, dy = b.x - a.x, b.y - a.y
    aa = dx * dx + dy * dy
    if aa == 0:
        return [(a, b)] if _distance_to_sweep((a.x, a.y), previous, center) > radius else []
    cuts = [0.0, 1.0]
    for cx, cy in (previous, center):
        fx, fy = a.x - cx, a.y - cy
        bb = 2 * (fx * dx + fy * dy)
        cc = fx * fx + fy * fy - radius * radius
        discriminant = bb * bb - 4 * aa * cc
        if discriminant > 0:
            cuts.extend(
                t
                for t in (
                    (-bb - sqrt(discriminant)) / (2 * aa),
                    (-bb + sqrt(discriminant)) / (2 * aa),
                )
                if 0 < t < 1
            )
    sx, sy = center[0] - previous[0], center[1] - previous[1]
    length = hypot(sx, sy)
    if length > 0:
        # Crossings of the parallel sides of the capsule's connecting rectangle.
        normal_x, normal_y = -sy / length, sx / length
        offset = (a.x - previous[0]) * normal_x + (a.y - previous[1]) * normal_y
        slope = dx * normal_x + dy * normal_y
        if abs(slope) > 1e-9:
            cuts.extend(
                t for t in ((radius - offset) / slope, (-radius - offset) / slope) if 0 < t < 1
            )
    cuts = sorted(set(cuts))

    def point(t):
        return StrokePoint(
            round(a.x + t * dx), round(a.y + t * dy), a.timestamp + t * (b.timestamp - a.timestamp)
        )

    intervals = []
    for lo, hi in zip(cuts, cuts[1:], strict=False):
        middle = (a.x + (lo + hi) / 2 * dx, a.y + (lo + hi) / 2 * dy)
        if _distance_to_sweep(middle, previous, center) > radius:
            if intervals and intervals[-1][1] == lo:
                intervals[-1] = (intervals[-1][0], hi)
            else:
                intervals.append((lo, hi))
    return [(point(lo), point(hi)) for lo, hi in intervals]


class StrokeManager:
    """Retain vector strokes; one pen gesture or eraser gesture is one undo action."""

    def __init__(self, minimum_distance: float = 3, maximum_jump: float = 85):
        self.minimum_distance = minimum_distance
        self.maximum_jump = maximum_jump
        self.strokes: list[Stroke] = []
        self.active: Stroke | None = None
        self._history: list[list[Stroke]] = []
        self._erasing = False
        self._eraser_previous = None
        self.revision = 0

    def _snapshot(self):
        self._history.append(deepcopy(self.strokes))
        self._history = self._history[-100:]

    def begin_stroke(self, point: StrokePoint):
        self.finish_stroke()
        self._snapshot()
        self.active = Stroke([point])
        self.strokes.append(self.active)
        self.revision += 1

    def append_point(self, point: StrokePoint):
        """Discard jitter; split implausible jumps without drawing a connecting line."""
        if self.active is None:
            self.begin_stroke(point)
            return
        previous = self.active.points[-1]
        distance = hypot(point.x - previous.x, point.y - previous.y)
        if distance > self.maximum_jump:
            self.active = Stroke([point])
            self.strokes.append(self.active)
        elif distance >= self.minimum_distance:
            self.active.points.append(point)
        else:
            return
        self.revision += 1

    def finish_stroke(self):
        """End a drawing or erasing gesture without discarding its points."""
        self.active = None
        self._erasing = False
        self._eraser_previous = None

    def undo(self):
        """Restore the state before the last edit, including clear and erase."""
        self.finish_stroke()
        if self._history:
            self.strokes = self._history.pop()
            self.revision += 1

    def clear(self):
        self.finish_stroke()
        if self.strokes:
            self._snapshot()
            self.strokes = []
            self.revision += 1

    def erase(self, center: tuple[int, int], radius: float, maximum_jump: float = 240):
        """Erase the entire travel corridor; keep one undo per continuous gesture."""
        previous = self._eraser_previous or center
        if hypot(center[0] - previous[0], center[1] - previous[1]) > maximum_jump:
            previous = center
        self._eraser_previous = center
        result = []
        for stroke in self.strokes:
            if len(stroke.points) == 1:
                p = stroke.points[0]
                if _distance_to_sweep((p.x, p.y), previous, center) > radius:
                    result.append(stroke)
                continue
            fragment = []
            for a, b in zip(stroke.points, stroke.points[1:], strict=False):
                for start, end in _outside_segments(a, b, center, radius, previous):
                    if fragment and fragment[-1] != start:
                        result.append(Stroke(fragment))
                        fragment = []
                    if not fragment:
                        fragment.append(start)
                    fragment.append(end)
                if _distance_to_sweep((b.x, b.y), previous, center) <= radius and fragment:
                    result.append(Stroke(fragment))
                    fragment = []
            if fragment:
                result.append(Stroke(fragment))
        if result != self.strokes:
            if not self._erasing:
                self._snapshot()
                self._erasing = True
            self.active = None
            self.strokes = result
            self.revision += 1

    def export_trajectory(self) -> list[list[dict]]:
        """Export timestamps relative to the earliest remaining point."""
        origin = min((p.timestamp for s in self.strokes for p in s.points), default=0)
        return [
            [{"x": p.x, "y": p.y, "t": p.timestamp - origin} for p in s.points]
            for s in self.strokes
        ]
