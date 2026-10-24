import numpy as np

from app import voice_out
from app.fakes import FakeDetector, FakeHandTracker, PrintBand, fake_compute
from app.loop import Guide, State
from shared.config import DEFAULTS
from shared.contracts import Direction

FRAME = np.zeros((480, 640, 3), np.uint8)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(found_after=0, visible=True):
    said, clock = [], Clock()
    tracker = FakeHandTracker(speed=60, visible=visible)
    band = PrintBand()
    guide = Guide(
        DEFAULTS,
        FakeDetector(found_after=found_after),
        tracker,
        fake_compute,
        band,
        say=said.append,
        clock=clock,
    )
    return guide, tracker, band, said, clock


def test_found_path_guides_to_done():
    guide, tracker, band, said, _ = make()
    det = guide.find([FRAME] * 10, "can")
    assert det is not None and guide.state == State.GUIDING
    assert said == [voice_out.LOOKING.format("can")]

    tracker.goal = det.center
    for _ in range(50):
        guide.step(FRAME)
        if guide.state == State.DONE:
            break
    assert guide.state == State.DONE
    assert said[-1] == voice_out.ARRIVED
    directions = [c.direction for c in band.sent]
    assert Direction.ARRIVED in directions
    assert directions[-1] == Direction.STOP


def test_not_found_says_exact_line():
    guide, _, band, said, _ = make(found_after=None)
    assert guide.find([FRAME] * 10, "can") is None
    assert guide.state == State.NOT_FOUND
    assert said[-1] == "I can't find a can"
    assert guide.step(FRAME) is None  # no guiding without a target
    assert band.sent == []


def test_hand_lost_warns_once_after_two_seconds():
    guide, _, _, said, clock = make(visible=False)
    guide.find([FRAME] * 10, "can")
    for t in (0.0, 1.0, 2.5, 3.0, 4.0):
        clock.t = t
        assert guide.step(FRAME).direction == Direction.LOST
    assert said.count(voice_out.HAND_LOST) == 1


def test_keeps_last_box_when_target_drops_out():
    guide, _, _, _, _ = make()
    det = guide.find([FRAME] * 10, "can")
    guide.detector.found_after = None  # target disappears
    guide.step(FRAME)
    assert guide.det == det


def test_stop_always_sends_stop_and_closes():
    guide, _, band, _, _ = make()
    guide.stop()
    assert band.sent[-1].direction == Direction.STOP
    assert band.closed
