import itertools

from vision.yolo.runtime.scan import scan


class StepClock:
    """Fake clock that advances `step` seconds every time it is read."""

    def __init__(self, step: float):
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        now = self.t
        self.t += self.step
        return now


def test_found_after_confirm_frames(fake_detector, can_frame):
    det = scan([can_frame] * 5, fake_detector, "can", timeout_s=5)
    assert det is not None
    assert det.label == "can"
    assert det.frame_id == 2  # third consecutive frame
    assert det.x1 < det.x2 and det.y1 < det.y2


def test_streak_resets_on_a_miss(fake_detector, can_frame, empty_frame):
    frames = [can_frame, can_frame, empty_frame, can_frame, can_frame, can_frame]
    det = scan(frames, fake_detector, "can", timeout_s=5)
    assert det is not None
    assert det.frame_id == 5


def test_not_found_is_none(fake_detector, empty_frame):
    assert scan([empty_frame] * 20, fake_detector, "can", timeout_s=5) is None


def test_unsupported_target_is_none(fake_detector, can_frame):
    assert scan([can_frame] * 20, fake_detector, "bottle", timeout_s=5) is None


def test_too_few_frames_is_none(fake_detector, can_frame):
    assert scan([can_frame] * 2, fake_detector, "can", timeout_s=5) is None


def test_timeout_stops_endless_stream(fake_detector, empty_frame):
    frames = itertools.repeat(empty_frame)  # like a live camera
    det = scan(frames, fake_detector, "can", timeout_s=5, clock=StepClock(1.0))
    assert det is None
    assert fake_detector.calls < 10


def test_timeout_before_confirmation(fake_detector, can_frame):
    # 1 s per frame, 1.5 s timeout: only one frame is checked, never 3 in a row.
    det = scan([can_frame] * 5, fake_detector, "can", timeout_s=1.5, clock=StepClock(1.0))
    assert det is None


def test_confirm_frames_override(fake_detector, can_frame):
    det = scan([can_frame] * 5, fake_detector, "can", timeout_s=5, confirm_frames=1)
    assert det.frame_id == 0
