from types import SimpleNamespace

import numpy as np
import pytest

from airwrite.camera import Camera
from airwrite.config import CameraConfig, HandTrackingConfig
from airwrite.hand_tracker import HandTracker


def test_camera_mirror_dimensions_and_cleanup(monkeypatch):
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    released = []
    capture = SimpleNamespace(
        isOpened=lambda: True,
        set=lambda *_: None,
        read=lambda: (True, frame),
        release=lambda: released.append(True),
    )
    monkeypatch.setattr("airwrite.camera.cv2.VideoCapture", lambda _: capture)
    with Camera(CameraConfig()) as camera:
        assert np.array_equal(camera.read(), frame[:, ::-1])
        assert (camera.width, camera.height) == (3, 2)
    assert released == [True]


def test_camera_open_failure_releases(monkeypatch):
    released = []
    capture = SimpleNamespace(isOpened=lambda: False, release=lambda: released.append(True))
    monkeypatch.setattr("airwrite.camera.cv2.VideoCapture", lambda _: capture)
    with pytest.raises(RuntimeError, match="Cannot open webcam"), Camera(CameraConfig()):
        pytest.fail("Failed camera must not enter context")
    assert released == [True]


def test_strictly_increasing_task_timestamps(monkeypatch):
    timestamps = []

    def detect(_, timestamp):
        timestamps.append(timestamp)
        return SimpleNamespace(hand_landmarks=[])

    tracker = HandTracker.__new__(HandTracker)
    tracker.last_timestamp = -1
    tracker.mp = SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=1))
    tracker.task = SimpleNamespace(detect_for_video=detect)
    monkeypatch.setattr("airwrite.hand_tracker.time.monotonic_ns", lambda: 1_000_000)
    for _ in range(3):
        assert tracker.detect(np.zeros((2, 2, 3), dtype=np.uint8)) is None
    assert timestamps == [1, 2, 3]


def test_missing_task_has_setup_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/download_models.py"):
        HandTracker(HandTrackingConfig(model_path=tmp_path / "missing.task"))
