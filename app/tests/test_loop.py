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


def test_cancel_stops_guiding():
    guide, _, band, _, _ = make()
    guide.find([FRAME] * 10, "can")
    guide.cancel()
    assert guide.state == State.IDLE
    assert band.sent[-1].direction == Direction.STOP
    assert guide.step(FRAME) is None


def test_stop_always_sends_stop_and_closes():
    guide, _, band, _, _ = make()
    guide.stop()
    assert band.sent[-1].direction == Direction.STOP
    assert band.closed


class TwoFrameArrive:
    """Minimal stabilizer: ARRIVED must be seen twice in a row."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.seen = 0
        self.resets = getattr(self, "resets", 0) + 1

    def update(self, cmd):
        from dataclasses import replace

        if cmd.direction == Direction.ARRIVED:
            self.seen += 1
            return cmd if self.seen >= 2 else replace(cmd, direction=Direction.UP)
        self.seen = 0
        return cmd


def test_stabilizer_sits_between_compute_and_band():
    said = []
    stab = TwoFrameArrive()
    band = PrintBand()
    tracker = FakeHandTracker(speed=1000)  # jumps straight onto the can
    guide = Guide(
        DEFAULTS, FakeDetector(), tracker, fake_compute, band, said.append, stabilizer=stab
    )
    det = guide.find([FRAME] * 10, "can")
    assert stab.resets == 2  # reset on each find()
    tracker.goal = det.center
    guide.step(FRAME)  # raw ARRIVED once -> held
    assert guide.state == State.GUIDING
    guide.step(FRAME)  # twice -> DONE
    assert guide.state == State.DONE


def test_object_distance_kept_while_hand_covers_it():
    from dataclasses import replace

    guide, tracker, _, _, _ = make()
    det = guide.find([FRAME] * 10, "can")
    guide.det = replace(det, distance_m=0.6)  # last good estimate

    class ShrunkDetector(FakeDetector):  # hand hides half the can: looks farther
        def detect(self, frame, target):
            return [replace(d, y2=(d.y1 + d.y2) / 2, distance_m=1.2) for d in super().detect(frame, target)]

    guide.detector = ShrunkDetector()
    tracker.pos = det.center  # fingertip over the can
    tracker.speed = 0
    guide.step(FRAME)
    assert guide.det.distance_m == 0.6

    guide.state = State.GUIDING  # fingertip on the can made the fake guidance arrive
    tracker.pos = (5.0, 5.0)  # hand moves away: trust new estimates again
    guide.step(FRAME)
    assert guide.det.distance_m == 1.2


def test_hand_over_phone_keeps_full_box():
    """Hand covers the middle of a phone: YOLO returns two fragments; keep the full box."""
    from shared.contracts import Detection

    guide, tracker, _, _, _ = make()
    guide.find([FRAME] * 10, "can")
    full = Detection("can", 0.9, 300, 100, 380, 300, 0, distance_m=0.5)
    guide.det = full

    class Fragments(FakeDetector):
        def detect(self, frame, target):
            return [
                Detection("can", 0.8, 300, 100, 380, 170, 1, distance_m=1.4),  # top half
                Detection("can", 0.7, 300, 240, 380, 300, 1, distance_m=1.6),  # bottom
            ]

    guide.detector = Fragments()
    tracker.pos, tracker.speed = (340, 205), 0  # fingertip on the middle
    guide.step(FRAME)
    kept = guide.det
    assert (kept.x1, kept.y1, kept.x2, kept.y2, kept.distance_m) == (300, 100, 380, 300, 0.5)


def test_follows_the_same_object_when_several_are_seen():
    from shared.contracts import Detection

    guide, tracker, _, _, _ = make()
    guide.find([FRAME] * 10, "can")
    guide.det = Detection("can", 0.6, 100, 100, 160, 200, 0)

    class TwoCans(FakeDetector):
        def detect(self, frame, target):
            return [
                Detection("can", 0.95, 500, 100, 560, 200, 1),  # other can, more confident
                Detection("can", 0.60, 104, 102, 164, 202, 1),  # ours, moved slightly
            ]

    guide.detector = TwoCans()
    tracker.pos, tracker.speed = (20, 460), 0  # hand far away
    guide.step(FRAME)
    assert guide.det.x1 == 104


# --- target leaves the view ---------------------------------------------------
def _edge_turn(last, frame_shape, config):
    """Stand-in for guidance.compute.turn_toward: left edge -> LEFT, else None."""
    from shared.contracts import GuidanceCommand

    return GuidanceCommand(Direction.LEFT, 200, 0.0) if last.x1 <= 10 else None


def _gone_setup(box):
    from shared.contracts import Detection

    said, clock = [], Clock()
    tracker = FakeHandTracker(speed=0)
    tracker.pos = (600.0, 450.0)  # hand far from the target, not covering it
    band = PrintBand()
    guide = Guide(
        DEFAULTS, FakeDetector(), tracker, fake_compute, band, said.append, clock, turn=_edge_turn
    )
    guide.find([FRAME] * 10, "can")
    guide.det = Detection("can", 0.9, *box, 0)
    guide.detector.found_after = None  # target no longer visible
    return guide, band, said, clock


def test_turns_toward_edge_the_target_left_from():
    from app.loop import MISSING_FRAMES

    guide, _, _, _ = _gone_setup((0, 200, 40, 300))
    for _ in range(MISSING_FRAMES - 1):
        guide.step(FRAME)
    assert not guide.turning
    guide.step(FRAME)
    assert guide.turning and guide.cmd.direction == Direction.LEFT


def test_vanishing_mid_frame_is_not_a_turn():
    guide, _, _, _ = _gone_setup((300, 200, 360, 300))
    for _ in range(20):
        guide.step(FRAME)
    assert not guide.turning
    assert guide.det.x1 == 300  # still guiding toward the last box


def test_turning_stops_when_target_is_back():
    from app.loop import MISSING_FRAMES

    guide, _, _, _ = _gone_setup((0, 200, 40, 300))
    for _ in range(MISSING_FRAMES):
        guide.step(FRAME)
    guide.detector.found_after = 0
    guide.step(FRAME)
    assert not guide.turning


def test_lost_too_long_says_so_and_requests_rescan():
    from app.loop import TARGET_LOST_S

    guide, band, said, clock = _gone_setup((0, 200, 40, 300))
    guide.step(FRAME)
    clock.t = TARGET_LOST_S + 0.1
    guide.step(FRAME)
    assert said[-1] == "I lost the can. Looking again."
    assert guide.rescan_requested and guide.state == State.IDLE
    assert band.sent[-1].direction == Direction.STOP


def test_hand_covering_target_never_counts_as_lost():
    from app.loop import TARGET_LOST_S

    guide, _, said, clock = _gone_setup((300, 200, 360, 300))
    guide.tracker.pos = (330.0, 250.0)  # fingertip on the target; YOLO can't see it
    guide.config = dict(DEFAULTS, guidance={"arrive_px": 0, "dead_zone_px": 0})
    guide.compute = lambda det, hand, shape, cfg: fake_compute(det, None, shape, cfg)  # no arrive
    for t in range(int(TARGET_LOST_S * 3)):
        clock.t = t
        guide.step(FRAME)
    assert not guide.rescan_requested
    assert "I lost the can. Looking again." not in said


def test_hand_outline_covering_object_freezes_box():
    """Palm point outside the box, but the hand's outline covers most of it."""
    from dataclasses import replace

    from shared.contracts import Detection, HandPosition

    guide, _, _, _, _ = make()
    guide.find([FRAME] * 10, "can")
    full = Detection("can", 0.9, 300, 100, 380, 300, 0)
    guide.det = full

    class Partial(FakeDetector):
        def detect(self, frame, target):
            return [Detection("can", 0.6, 300, 100, 380, 150, 1)]  # only the top shows

    class HoldingTracker:
        landmarks = None

        def track(self, frame):
            return HandPosition(340, 330, 340, 400, "Right", 0, box=(280, 140, 400, 420))

    guide.detector, guide.tracker = Partial(), HoldingTracker()
    guide.step(FRAME)
    assert replace(guide.det, confidence=0.9, frame_id=0) == full


def test_continuous_detection_recovers_after_target_disappears():
    guide, tracker, band, _, clock = make()
    tracker.speed = 0
    guide.start("can")
    guide.detector.found_after = None
    for _ in range(20):
        guide.follow(FRAME)
    assert guide.state == State.SCANNING
    assert guide.det is None
    assert band.sent[-1].direction == Direction.STOP

    guide.detector.found_after = 0
    for _ in range(guide.detector.confirm_frames):
        guide.follow(FRAME)
    assert guide.state == State.GUIDING
    assert band.sent[-1].direction != Direction.STOP

    guide.detector.found_after = None
    clock.t = 2.0
    guide.follow(FRAME)
    assert guide.state == State.SCANNING
    assert guide.det is None
    assert band.sent[-1].direction == Direction.STOP


def test_continuous_detection_resumes_after_arrival_without_repeating_grab():
    from dataclasses import replace
    from app.loop import CONTACT_HOLD_S

    guide, tracker, band, said, clock = make()
    detect, track = guide.detector.detect, tracker.track
    guide.detector.detect = lambda *a: [replace(d, distance_m=1.0) for d in detect(*a)]
    tracker.track = lambda *a: replace(track(*a), distance_m=1.0)
    guide.start("can")
    for _ in range(guide.detector.confirm_frames):
        guide.follow(FRAME)
    tracker.pos, tracker.speed = guide.det.center, 0
    guide.follow(FRAME)
    assert guide.state != State.DONE
    clock.t += CONTACT_HOLD_S + 0.1
    guide.follow(FRAME)
    assert guide.state == State.DONE
    guide.follow(FRAME)
    assert band.sent[-1].direction == Direction.STOP
    assert said.count(voice_out.ARRIVED) == 1
    tracker.pos = (5.0, 5.0)
    guide.follow(FRAME)
    assert guide.state == State.GUIDING
    assert band.sent[-1].direction not in (Direction.STOP, Direction.ARRIVED)


def test_continuous_occlusion_remembers_box_without_confirming_contact():
    from app.loop import CONTACT_HOLD_S, OCCLUSION_MEMORY_S
    from shared.contracts import GuidanceCommand

    guide, tracker, band, said, clock = make()
    tracker.speed = 0
    guide.start("can")
    for _ in range(guide.detector.confirm_frames):
        guide.follow(FRAME)
    remembered = guide.det
    tracker.pos = remembered.center
    guide.compute = lambda *args: GuidanceCommand(Direction.FORWARD, 180, 0)
    guide.detector.found_after = None
    guide.follow(FRAME)
    assert guide.det == remembered
    assert band.sent[-1].direction == Direction.FORWARD
    clock.t = CONTACT_HOLD_S + 0.1
    guide.follow(FRAME)
    assert guide.state != State.DONE
    assert band.sent[-1].direction == Direction.FORWARD
    guide.follow(FRAME)
    assert guide.state != State.DONE
    assert voice_out.ARRIVED not in said
    assert band.sent[-1].direction == Direction.FORWARD
    clock.t = OCCLUSION_MEMORY_S + 0.1
    guide.follow(FRAME)
    assert guide.state == State.SCANNING
    assert guide.det is None


def test_brief_miss_away_from_hand_keeps_memory_but_stops_motors():
    guide, tracker, band, _, clock = make()
    tracker.speed = 0
    guide.start("can")
    for _ in range(guide.detector.confirm_frames):
        guide.follow(FRAME)
    remembered = guide.det
    guide.detector.found_after = None
    clock.t = 0.2
    guide.follow(FRAME)
    assert guide.det == remembered
    assert band.sent[-1].direction == Direction.STOP
    guide.detector.found_after = 0
    guide.follow(FRAME)
    assert guide.state == State.GUIDING
    assert band.sent[-1].direction != Direction.STOP


def test_passing_hand_does_not_confirm_contact_or_override_missing_hand():
    from app.loop import CONTACT_HOLD_S
    from shared.contracts import GuidanceCommand

    guide, tracker, _, _, clock = make()
    tracker.speed = 0
    guide.start("can")
    for _ in range(guide.detector.confirm_frames):
        guide.follow(FRAME)
    guide.compute = lambda *args: GuidanceCommand(Direction.BACK, 180, 0)
    tracker.pos = guide.det.center
    guide.follow(FRAME)
    clock.t = CONTACT_HOLD_S / 2
    tracker.pos = (5, 5)
    guide.follow(FRAME)
    assert guide.state != State.DONE
    clock.t = CONTACT_HOLD_S + 1
    tracker.pos = guide.det.center
    guide.follow(FRAME)
    assert guide.state != State.DONE
    tracker.visible = False
    clock.t += CONTACT_HOLD_S + 1
    guide.follow(FRAME)
    assert guide.state != State.DONE


def test_continuous_search_announces_timeout_once_then_found_once():
    guide, tracker, _, said, clock = make(found_after=None)
    tracker.speed = 0
    guide.start("can")
    clock.t = float(guide.config.get("scan_timeout_s", 5)) + 0.1
    guide.follow(FRAME)
    guide.follow(FRAME)
    assert said.count(voice_out.NOT_FOUND.format("can")) == 1
    guide.detector.found_after = 0
    for _ in range(guide.detector.confirm_frames + 3):
        guide.follow(FRAME)
    assert said.count(voice_out.FOUND) == 1
    guide.detector.found_after = None
    clock.t += 10
    guide.follow(FRAME)
    guide.detector.found_after = 0
    for _ in range(guide.detector.confirm_frames):
        guide.follow(FRAME)
    assert said.count(voice_out.FOUND) == 1
    guide.start("can")
    for _ in range(guide.detector.confirm_frames):
        guide.follow(FRAME)
    assert said.count(voice_out.FOUND) == 2
