from dataclasses import replace

import cv2
import numpy as np
import pytest

from app import voice_out
from app.loop import (
    CONTACT_HOLD_S, HAND_LOST_AFTER_S, MISSING_NOTICE_S, NOTICE_GAP_S,
    OCCLUSION_MEMORY_S, REACQUIRE_HOLD_S, SEARCH_REMINDER_S, State,
)
from app.scene_motion import SceneMotion
from app.tests.test_loop import FRAME, make
from shared.contracts import Direction, GuidanceCommand


def ready():
    guide, tracker, band, said, clock = make()
    tracker.speed = 0
    guide.start("can")
    for _ in range(guide.detector.confirm_frames):
        guide.follow(FRAME)
    return guide, tracker, band, said, clock


@pytest.mark.parametrize("direction", [Direction.FORWARD, Direction.BACK])
def test_hand_in_front_never_overrides_depth_guidance(direction):
    guide, tracker, band, said, clock = ready()
    tracker.pos = guide.det.center
    guide.compute = lambda *a: GuidanceCommand(direction, 180, 0)
    for t in (0, 1, 3):
        clock.t = t
        guide.follow(FRAME)
        assert guide.state != State.DONE
        assert band.sent[-1].direction == direction
    assert voice_out.ARRIVED not in said


def test_overlap_without_depth_asks_for_check_once_and_never_arrives():
    guide, tracker, band, said, clock = ready()
    tracker.pos = guide.det.center
    for t in (0, 1, 3):
        clock.t = t
        guide.follow(FRAME)
        assert band.sent[-1].direction == Direction.STOP
    assert said.count(voice_out.CHECK_DEPTH) == 1
    assert voice_out.ARRIVED not in said


def test_covered_memory_does_not_arrive_even_with_depth():
    guide, tracker, band, said, clock = ready()
    guide.det = replace(guide.det, distance_m=1.0)
    tracker.pos = guide.det.center
    track = tracker.track
    tracker.track = lambda *a: replace(track(*a), distance_m=1.0)
    guide.detector.found_after = None
    guide.follow(FRAME)
    remembered = guide.det
    for t in (MISSING_NOTICE_S + 0.1, OCCLUSION_MEMORY_S - 0.1):
        clock.t = t
        guide.follow(FRAME)
        assert guide.det == remembered
        assert guide.visibility == "occluded"
        assert band.sent[-1].direction == Direction.FORWARD
    assert said.count(voice_out.TARGET_COVERED.format("can")) == 1
    assert voice_out.ARRIVED not in said
    clock.t = OCCLUSION_MEMORY_S + 0.1
    guide.follow(FRAME)
    assert guide.det is None
    assert guide.state == State.SCANNING
    assert band.sent[-1].direction == Direction.STOP
    clock.t = MISSING_NOTICE_S + NOTICE_GAP_S + 0.2
    guide.follow(FRAME)
    assert said[-1] == voice_out.LOOK_AROUND.format("can")


def test_lost_target_prompts_search_and_announces_reacquisition():
    guide, _, band, said, clock = ready()
    guide.detector.found_after = None
    guide.follow(FRAME)
    assert voice_out.LOOK_AROUND.format("can") not in said
    clock.t = MISSING_NOTICE_S + 0.1
    guide.follow(FRAME)
    guide.follow(FRAME)
    assert said.count(voice_out.LOOK_AROUND.format("can")) == 1
    assert band.sent[-1].direction == Direction.STOP
    clock.t += SEARCH_REMINDER_S + 0.1
    guide.follow(FRAME)
    assert said.count(voice_out.LOOK_AROUND.format("can")) == 2
    guide.detector.found_after = 0
    for _ in range(guide.detector.confirm_frames + 1):
        guide.follow(FRAME)
    assert voice_out.TARGET_AGAIN.format("can") not in said
    clock.t += NOTICE_GAP_S + REACQUIRE_HOLD_S + 0.1
    guide.follow(FRAME)
    assert said.count(voice_out.TARGET_AGAIN.format("can")) == 1
    assert guide.visibility == "visible"
    assert guide.state == State.GUIDING


def test_camera_motion_invalidates_coverage_at_old_image_position():
    guide, tracker, band, said, clock = ready()
    tracker.pos = guide.det.center
    guide.detector.found_after = None
    guide._motion.update = lambda *a: (0.04, 0)
    clock.t = 0.1
    guide.follow(FRAME)
    assert guide.det is None
    assert guide.visibility == "missing"
    assert band.sent[-1].direction == Direction.STOP
    clock.t += MISSING_NOTICE_S + 0.1
    guide.follow(FRAME)
    assert said[-1] == voice_out.LOOK_AROUND.format("can")
    assert voice_out.TARGET_COVERED.format("can") not in said


@pytest.mark.parametrize("mirror,side", [(False, "left"), (True, "right")])
def test_remembers_side_target_left_from(mirror, side):
    guide, _, _, said, clock = ready()
    guide.config = dict(guide.config, mirror=mirror)
    guide.det = replace(guide.det, x1=0, x2=40)
    guide.detector.found_after = None
    guide.follow(FRAME)
    clock.t = MISSING_NOTICE_S + 0.1
    guide.follow(FRAME)
    assert said[-1] == voice_out.LOOK_BACK.format("can", side)


def test_depth_agreement_must_be_continuous_and_fresh():
    guide, tracker, band, said, clock = ready()
    detect, track = guide.detector.detect, tracker.track
    guide.detector.detect = lambda *a: [replace(d, distance_m=1.0) for d in detect(*a)]
    tracker.track = lambda *a: replace(track(*a), distance_m=1.0)
    tracker.pos = guide.det.center
    guide.follow(FRAME)
    clock.t = CONTACT_HOLD_S / 2
    tracker.visible = False
    tracker.track = lambda *a: None
    guide.follow(FRAME)
    clock.t = CONTACT_HOLD_S + 0.1
    tracker.visible = True
    tracker.track = lambda *a: replace(track(*a), distance_m=1.0)
    guide.follow(FRAME)
    assert guide.state != State.DONE
    clock.t += CONTACT_HOLD_S + 0.1
    guide.follow(FRAME)
    assert guide.state == State.DONE
    assert band.sent[-1].direction == Direction.ARRIVED
    assert said.count(voice_out.ARRIVED) == 1


def test_arrival_rejected_when_depth_disagrees_or_only_fragment_visible():
    guide, tracker, band, said, clock = ready()
    guide.det = replace(guide.det, distance_m=1.0)
    tracker.pos = guide.det.center
    detect, track = guide.detector.detect, tracker.track
    guide.compute = lambda *a: GuidanceCommand(Direction.ARRIVED, 255, 0)
    guide.detector.detect = lambda *a: [replace(d, distance_m=1.0) for d in detect(*a)]
    tracker.track = lambda *a: replace(track(*a), distance_m=0.5)
    for t in (0, 1):
        clock.t = t
        guide.follow(FRAME)
        assert band.sent[-1].direction == Direction.STOP
    tracker.track = lambda *a: replace(track(*a), distance_m=1.0)
    guide.detector.detect = lambda *a: [
        replace(d, y2=(d.y1 + d.y2) / 2, distance_m=1.0) for d in detect(*a)
    ]
    for t in (2, 3):
        clock.t = t
        guide.follow(FRAME)
        assert band.sent[-1].direction == Direction.STOP
    assert voice_out.ARRIVED not in said


def test_scene_motion_detects_background_translation():
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    motion = SceneMotion()
    assert motion.update(frame) is None
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 4], [0, 1, 0]]), (160, 120))
    dx, dy = motion.update(shifted)
    assert dx == pytest.approx(4 / 160, abs=0.003)
    assert dy == pytest.approx(0, abs=0.003)


def test_scene_motion_ignores_masked_hand_and_handles_blank_frames():
    motion = SceneMotion()
    assert motion.update(FRAME) is None
    assert motion.update(FRAME) is None
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    motion.reset()
    motion.update(frame)
    moved_hand = frame.copy()
    moved_hand[30:90, 40:110] = 255
    shift = motion.update(moved_hand, [(35, 25, 115, 95)])
    assert shift is None or (abs(shift[0]) < 0.005 and abs(shift[1]) < 0.005)


def test_scene_cut_invalidates_memory_even_without_features():
    motion = SceneMotion()
    motion.update(FRAME)
    assert motion.update(np.full_like(FRAME, 255)) == (1.0, 1.0)


def test_stabilizer_cannot_replay_arrival_from_memory():
    guide, tracker, band, said, _ = ready()
    tracker.pos = guide.det.center
    guide.detector.found_after = None
    guide.compute = lambda *a: GuidanceCommand(Direction.FORWARD, 180, 0)

    class HoldingArrival:
        def reset(self):
            pass

        def update(self, cmd):
            return GuidanceCommand(Direction.ARRIVED, 255, 0)

    guide.stabilizer = HoldingArrival()
    guide.follow(FRAME)
    assert band.sent[-1].direction == Direction.FORWARD
    assert voice_out.ARRIVED not in said


def test_occlusion_forward_ends_when_hand_moves_or_target_returns():
    guide, tracker, band, said, clock = ready()
    tracker.pos = guide.det.center
    guide.detector.found_after = None
    guide.follow(FRAME)
    assert band.sent[-1].direction == Direction.FORWARD
    clock.t = MISSING_NOTICE_S + 0.1
    guide.follow(FRAME)
    assert said[-1] == voice_out.TARGET_COVERED.format("can")
    assert voice_out.LOOK_AROUND.format("can") not in said
    tracker.pos = (5, 5)
    guide.follow(FRAME)
    assert band.sent[-1].direction == Direction.STOP
    tracker.pos = guide.det.center
    guide.follow(FRAME)
    assert band.sent[-1].direction == Direction.FORWARD
    guide.detector.found_after = 0
    guide.compute = lambda *args: GuidanceCommand(Direction.RIGHT, 180, 50)
    guide.follow(FRAME)
    assert band.sent[-1].direction == Direction.RIGHT


def test_hand_loss_stops_occlusion_forward():
    guide, tracker, band, _, _ = ready()
    tracker.pos = guide.det.center
    guide.detector.found_after = None
    guide.follow(FRAME)
    assert band.sent[-1].direction == Direction.FORWARD
    tracker.visible = False
    guide.follow(FRAME)
    assert band.sent[-1].direction == Direction.STOP


def test_repeated_target_flicker_does_not_repeat_lost_and_found_speech():
    guide, _, _, said, clock = ready()
    for cycle in range(6):
        clock.t = cycle * 1.0
        guide.detector.found_after = None
        guide.follow(FRAME)
        clock.t += MISSING_NOTICE_S + 0.1
        guide.follow(FRAME)
        guide.detector.found_after = 0
        guide.follow(FRAME)
    assert said.count(voice_out.LOOK_AROUND.format("can")) == 1
    assert voice_out.TARGET_AGAIN.format("can") not in said


def test_lifting_covering_hand_is_silent_even_after_long_pause():
    guide, tracker, _, said, clock = ready()
    tracker.pos = guide.det.center
    guide.detector.found_after = None
    guide.follow(FRAME)
    clock.t = MISSING_NOTICE_S + 0.1
    guide.follow(FRAME)
    guide.detector.found_after = 0
    clock.t = NOTICE_GAP_S + 0.1
    tracker.pos = (5, 5)
    guide.follow(FRAME)
    clock.t += REACQUIRE_HOLD_S + 0.1
    for _ in range(4):
        guide.follow(FRAME)
    assert voice_out.TARGET_AGAIN.format("can") not in said
    assert guide._last_notice is None


def test_quick_stable_return_clears_search_notice_silently():
    guide, _, _, said, clock = ready()
    guide.detector.found_after = None
    guide.follow(FRAME)
    clock.t = MISSING_NOTICE_S + 0.1
    guide.follow(FRAME)
    guide.detector.found_after = 0
    guide.follow(FRAME)
    clock.t += REACQUIRE_HOLD_S + 0.1
    for _ in range(3):
        guide.follow(FRAME)
    assert voice_out.TARGET_AGAIN.format("can") not in said
    assert guide._last_notice is None


def test_one_frame_hand_recovery_does_not_rearm_lost_warning():
    guide, tracker, _, said, clock = ready()
    tracker.visible = False
    guide.follow(FRAME)
    clock.t = HAND_LOST_AFTER_S + 0.1
    guide.follow(FRAME)
    assert said.count(voice_out.HAND_LOST) == 1
    tracker.visible = True
    clock.t += 0.1
    guide.follow(FRAME)
    tracker.visible = False
    clock.t += 0.1
    guide.follow(FRAME)
    clock.t += NOTICE_GAP_S + HAND_LOST_AFTER_S
    guide.follow(FRAME)
    assert said.count(voice_out.HAND_LOST) == 1


def test_depth_check_does_not_rearm_when_alignment_flickers():
    guide, tracker, _, said, clock = ready()
    center = guide.det.center
    for i in range(8):
        tracker.pos = center if i % 2 else (5, 5)
        clock.t = i
        guide.follow(FRAME)
    assert said.count(voice_out.CHECK_DEPTH) == 1
