"""Validated configuration, independent of camera and ML dependencies."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CameraConfig:
    """Capture device and requested resolution."""

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 60


@dataclass
class HandTrackingConfig:
    """MediaPipe task options."""

    model_path: Path = Path("models/hand_landmarker.task")
    num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class GestureConfig:
    """Scale-independent pinch thresholds."""

    pinch_start_ratio: float = 0.35
    pinch_release_ratio: float = 0.45
    control_mode: str = "fingers"
    finger_straight_angle: float = 155.0
    finger_release_angle: float = 140.0
    finger_reach_ratio: float = 1.12
    fist_curl_angle: float = 130.0
    pose_hold_seconds: float = 0.12
    write_hold_seconds: float = 0.035
    toolbar_dwell_seconds: float = 0.8

    def __post_init__(self):
        if not 0 < self.pinch_start_ratio < self.pinch_release_ratio:
            raise ValueError("Pinch thresholds must satisfy 0 < start < release")
        if self.control_mode not in ("fingers", "pinch"):
            raise ValueError("Control mode must be fingers or pinch")
        if not 90 < self.finger_straight_angle < 180 or self.finger_reach_ratio <= 1:
            raise ValueError("Invalid finger extension thresholds")
        if self.pose_hold_seconds < 0 or self.toolbar_dwell_seconds <= 0:
            raise ValueError("Invalid gesture hold times")
        if not 0 < self.fist_curl_angle < self.finger_straight_angle:
            raise ValueError("Fist curl angle must be below the finger extension angle")
        if not self.fist_curl_angle < self.finger_release_angle < self.finger_straight_angle:
            raise ValueError("Finger release angle must lie between curled and straight angles")
        if self.write_hold_seconds < 0:
            raise ValueError("Write hold time must be non-negative")


@dataclass
class DrawingConfig:
    """Drawing, filtering and layout options in capture pixels."""

    smoothing_alpha: float = 0.35
    smoothing_fast_distance: float = 24.0
    smoothing_reset_seconds: float = 0.2
    pen_thickness: int = 5
    eraser_radius: int = 32
    eraser_maximum_jump: float = 240.0
    minimum_point_distance: float = 3.0
    maximum_jump: float = 240.0
    toolbar_height: int = 48
    toolbar_top_margin: int = 64
    toolbar_side_margin: int = 48
    status_height: int = 64
    output_dir: Path = Path("outputs/drawings")

    def __post_init__(self):
        if not 0 < self.smoothing_alpha <= 1:
            raise ValueError("Smoothing alpha must be in (0, 1]")
        if not 0 <= self.minimum_point_distance < self.maximum_jump:
            raise ValueError("Point distances must satisfy 0 <= minimum < maximum")
        if min(self.pen_thickness, self.eraser_radius) <= 0:
            raise ValueError("Pen thickness and eraser radius must be positive")
        if self.smoothing_fast_distance <= 0:
            raise ValueError("Fast smoothing distance must be positive")


@dataclass
class RecognitionConfig:
    """Optional CNN weights and prediction export."""

    character_model_path: Path = Path("models/airwrite_cnn.pt")
    output_dir: Path = Path("outputs/predictions")


@dataclass
class AppConfig:
    """Complete runtime configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    debug: bool = False
