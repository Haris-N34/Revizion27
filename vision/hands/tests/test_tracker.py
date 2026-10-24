from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from shared.config import DEFAULTS
from vision.hands.draw import draw_hand
from vision.hands.tracker import HandTracker, model_path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
W, H = 640, 480


def _result(score=0.9, side="Right", hands=True):
    """MediaPipe-shaped result: landmark i at normalized (i/40, i/30)."""
    if not hands:
        return SimpleNamespace(hand_landmarks=[], handedness=[])
    lms = [SimpleNamespace(x=i / 40, y=i / 30) for i in range(21)]
    return SimpleNamespace(
        hand_landmarks=[lms],
        handedness=[[SimpleNamespace(category_name=side, score=score)]],
    )


class FakeLandmarker:
    def __init__(self, result):
        self.result = result
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, image, ts):
        self.timestamps.append(ts)
        return self.result

    def close(self):
        self.closed = True


def make(result, mirror=False):
    cfg = dict(DEFAULTS, mirror=mirror)
    fake = FakeLandmarker(result)
    return HandTracker(cfg, landmarker=fake), fake


def frame():
    return np.zeros((H, W, 3), np.uint8)


def test_fingertip_and_wrist_in_pixels():
    tracker, _ = make(_result())
    hand = tracker.track(frame())
    assert (hand.x, hand.y) == pytest.approx((8 / 40 * W, 8 / 30 * H))
    assert (hand.wrist_x, hand.wrist_y) == (0, 0)
    assert len(tracker.landmarks) == 21


def test_no_hand_is_none():
    tracker, _ = make(_result(hands=False))
    assert tracker.track(frame()) is None
    assert tracker.landmarks is None


def test_low_confidence_is_none():
    tracker, _ = make(_result(score=0.3))
    assert tracker.track(frame()) is None


def test_mirror_flips_handedness():
    assert make(_result(side="Right"))[0].track(frame()).handedness == "Right"
    assert make(_result(side="Right"), mirror=True)[0].track(frame()).handedness == "Left"


def test_frame_ids_and_timestamps_increase():
    tracker, fake = make(_result())
    ids = [tracker.track(frame()).frame_id for _ in range(5)]
    assert ids == [0, 1, 2, 3, 4]
    assert all(b > a for a, b in zip(fake.timestamps, fake.timestamps[1:]))


def test_close_releases_landmarker():
    tracker, fake = make(_result())
    tracker.close()
    assert fake.closed


def test_draw_hand_draws_and_tolerates_none():
    tracker, _ = make(_result())
    tracker.track(frame())
    img = frame()
    draw_hand(img, tracker.landmarks)
    assert img.any()
    blank = frame()
    draw_hand(blank, None)
    assert not blank.any()


def test_real_model_finds_no_hand_in_empty_scene():
    """Skipped unless mediapipe and models/hand_landmarker.task are present."""
    pytest.importorskip("mediapipe")
    if not model_path(DEFAULTS).exists():
        pytest.skip("hand model not downloaded")
    tracker = HandTracker(DEFAULTS)
    try:
        assert tracker.track(cv2.imread(str(FIXTURES / "empty.png"))) is None
    finally:
        tracker.close()
