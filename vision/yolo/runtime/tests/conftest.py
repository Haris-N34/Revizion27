from pathlib import Path

import cv2
import numpy as np
import pytest

from shared.contracts import Detection

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FakeDetector:
    """Finds the 'can' as the red pixels in a frame. No YOLO, no camera."""

    confirm_frames = 3

    def __init__(self):
        self.calls = 0

    def detect(self, frame: np.ndarray, target: str) -> list[Detection]:
        frame_id = self.calls
        self.calls += 1
        if target != "can":
            return []
        b, g, r = frame[..., 0], frame[..., 1], frame[..., 2]
        ys, xs = np.nonzero((r > 200) & (g < 60) & (b < 60))
        if len(xs) == 0:
            return []
        return [Detection("can", 0.9, xs.min(), ys.min(), xs.max(), ys.max(), frame_id)]


def _load(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIXTURES / name))
    assert frame is not None, f"missing fixture {name}"
    return frame


@pytest.fixture
def can_frame() -> np.ndarray:
    return _load("can.png")


@pytest.fixture
def empty_frame() -> np.ndarray:
    return _load("empty.png")


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()
