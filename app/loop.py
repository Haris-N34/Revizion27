"""The find-and-guide state machine. No camera or window here; main.py feeds frames."""

import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from enum import Enum

import numpy as np

from app import voice_out
from shared.contracts import Detection, Direction, GuidanceCommand, HandPosition
from vision.yolo.runtime.scan import scan

STOP = GuidanceCommand(Direction.STOP, 0, 0.0)
HAND_LOST_AFTER_S = 2.0
COVER_MARGIN = 0.15  # hand point within the box grown by 15% counts as covering it
COVER_OVERLAP = 0.3  # ...or the hand's outline covering 30% of the box
MISSING_FRAMES = 8  # target unseen this many frames (not under the hand) = out of view
TARGET_LOST_S = 6.0  # out of view this long: give up and scan again
TARGET_MEMORY_S = 1.0  # tolerate brief detector misses without discarding the box
OCCLUSION_MEMORY_S = 4.0  # longer memory while the hand covers the remembered box
CONTACT_HOLD_S = 0.6  # sustained overlap confirms visual contact


def _inside(x: float, y: float, d: Detection, margin: float = 0.0) -> bool:
    mx, my = (d.x2 - d.x1) * margin, (d.y2 - d.y1) * margin
    return d.x1 - mx <= x <= d.x2 + mx and d.y1 - my <= y <= d.y2 + my


def _covering(hand: HandPosition | None, det: Detection) -> bool:
    """Is the hand over the object: hand point inside the box (+COVER_MARGIN), or the
    hand's outline covering at least COVER_OVERLAP of it?"""
    if hand is None:
        return False
    if _inside(hand.x, hand.y, det, COVER_MARGIN):
        return True
    if hand.box is None:
        return False
    hx1, hy1, hx2, hy2 = hand.box
    w = min(det.x2, hx2) - max(det.x1, hx1)
    h = min(det.y2, hy2) - max(det.y1, hy1)
    area = (det.x2 - det.x1) * (det.y2 - det.y1)
    return w > 0 and h > 0 and area > 0 and (w * h) / area >= COVER_OVERLAP


def _iou(a: Detection, b: Detection) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


class State(str, Enum):
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    NOT_FOUND = "NOT_FOUND"
    GUIDING = "GUIDING"
    DONE = "DONE"


class Guide:
    def __init__(
        self,
        config: dict,
        detector,
        tracker,
        compute: Callable,
        band,
        say: Callable[[str], None] = voice_out.speak,
        clock: Callable[[], float] = time.monotonic,
        stabilizer=None,
        turn: Callable | None = None,
    ):
        """`stabilizer`: optional object with update(cmd) -> cmd and reset()
        (guidance.stabilize.Stabilizer); smooths commands between compute and band.
        `turn`: optional turn_toward(last_det, frame_shape, config) -> cmd | None
        (guidance.compute.turn_toward); used when the target leaves the view."""
        self.config = config
        self.stabilizer = stabilizer
        self.turn = turn
        self.detector = detector
        self.tracker = tracker
        self.compute = compute
        self.band = band
        self.say = say
        self.clock = clock
        self.state = State.IDLE
        self.target: str | None = None
        self.det: Detection | None = None  # last known box of the target
        self.hand: HandPosition | None = None
        self.cmd: GuidanceCommand | None = None
        self._lost_since: float | None = None
        self._warned_lost = False
        self.turning = False  # True while telling the user to turn toward the target
        self.rescan_requested = False  # set when the target was lost; main.py rescans
        self._missing = 0
        self._missing_since: float | None = None
        self._confirmed = 0
        self._seen_at: float | None = None
        self._contact_since: float | None = None
        self._search_started = self.clock()
        self._announced_found = False
        self._announced_not_found = False

    def start(self, target: str, announce: bool = True) -> None:
        """Continuously look for a target until the user changes it or quits."""
        self.cancel()
        self.target, self.det, self.hand = target, None, None
        self._confirmed = 0
        self._seen_at, self._contact_since = None, None
        self._search_started = self.clock()
        self._announced_found = self._announced_not_found = False
        self._lost_since, self._warned_lost = None, False
        self.turning, self.rescan_requested = False, False
        if self.stabilizer is not None:
            self.stabilizer.reset()
        self.state = State.SCANNING
        if announce:
            self.say(voice_out.LOOKING.format(target))

    def follow(self, frame: np.ndarray) -> GuidanceCommand | None:
        """Track continuously, retaining a bounded memory during hand occlusion."""
        if self.target is None:
            return None
        found = self.detector.detect(frame, self.target)
        self.hand = self.tracker.track(frame)
        now = self.clock()
        covering = self.det is not None and _covering(self.hand, self.det)
        if found:
            self._seen_at = now
        age = now - self._seen_at if self._seen_at is not None else float("inf")
        memory = OCCLUSION_MEMORY_S if covering else TARGET_MEMORY_S
        if not found and (self.det is None or age > memory):
            self.det, self.cmd = None, STOP
            self._confirmed = 0
            self._contact_since = None
            self.state = State.SCANNING
            self._lost_since, self._warned_lost = None, False
            if self.stabilizer is not None:
                self.stabilizer.reset()
            self.band.send(STOP)
            if (
                not self._announced_found
                and not self._announced_not_found
                and now - self._search_started >= float(self.config.get("scan_timeout_s", 5))
            ):
                self.say(voice_out.NOT_FOUND.format(self.target))
                self._announced_not_found = True
            return self.cmd

        self.det = self._update_target(found)
        if found:
            self._confirmed += 1
        if self._confirmed < max(1, self.detector.confirm_frames):
            self.cmd = STOP
            self.band.send(STOP)
            return self.cmd
        if not self._announced_found:
            self.say(voice_out.FOUND)
            self._announced_found = True
        arrived = self.state == State.DONE
        # A missing target without a covering hand is uncertain: keep its box
        # briefly, but never drive toward an unseen object in that case.
        if not found and not covering:
            self._contact_since = None
            self.cmd = STOP
            self.band.send(STOP)
            return self.cmd
        self.cmd = self.compute(self.det, self.hand, frame.shape, self.config)
        # Size-based monocular depth can become unreliable near a grasp. Use
        # sustained visual contact instead of oscillating FORWARD/BACK forever.
        contact = self.hand is not None and _covering(self.hand, self.det)
        if contact:
            if self._contact_since is None:
                self._contact_since = now
            if arrived or now - self._contact_since >= CONTACT_HOLD_S:
                self.cmd = GuidanceCommand(Direction.ARRIVED, 255, self.cmd.distance_px)
            elif self.cmd.direction in (Direction.FORWARD, Direction.BACK):
                self.cmd = STOP
        else:
            self._contact_since = None
        if self.stabilizer is not None:
            if self.cmd.direction == Direction.STOP:
                self.stabilizer.reset()
            else:
                self.cmd = self.stabilizer.update(self.cmd)
        if self.cmd.direction == Direction.ARRIVED:
            if not arrived:
                self.band.send(self.cmd)
                self.say(voice_out.ARRIVED)
            else:
                self.band.send(STOP)
            self.state = State.DONE
        else:
            self.state = State.GUIDING
            self.band.send(self.cmd)
        self._check_lost()
        return self.cmd

    def find(self, frames: Iterable[np.ndarray], target: str, detector=None) -> Detection | None:
        """SCANNING: returns the detection (state GUIDING) or None (state NOT_FOUND).

        `detector` overrides self.detector for the scan only (e.g. a drawing wrapper).
        """
        self.target, self.det, self.hand, self.cmd = target, None, None, None
        self._lost_since, self._warned_lost = None, False
        self.turning, self.rescan_requested = False, False
        self._missing, self._missing_since = 0, None
        if self.stabilizer is not None:
            self.stabilizer.reset()
        self.state = State.SCANNING
        self.say(voice_out.LOOKING.format(target))
        timeout = float(self.config.get("scan_timeout_s", 5))
        self.det = scan(frames, detector or self.detector, target, timeout)
        if self.det is None:
            self.say(voice_out.NOT_FOUND.format(target))
            self.state = State.NOT_FOUND
        else:
            self.state = State.GUIDING
        return self.det

    def step(self, frame: np.ndarray) -> GuidanceCommand | None:
        """GUIDING: one frame of detect -> track -> compute -> send."""
        if self.state != State.GUIDING:
            return None
        found = self.detector.detect(frame, self.target)
        self.hand = self.tracker.track(frame)
        self._track_missing(found)
        self.det = self._update_target(found)

        turn = None
        if self._missing >= MISSING_FRAMES and self.turn is not None and self.det is not None:
            turn = self.turn(self.det, frame.shape, self.config)
        self.turning = turn is not None
        self.cmd = turn or self.compute(self.det, self.hand, frame.shape, self.config)
        if self.stabilizer is not None:
            self.cmd = self.stabilizer.update(self.cmd)
        self.band.send(self.cmd)
        self._check_lost()

        if self._missing_since is not None and self.clock() - self._missing_since > TARGET_LOST_S:
            self.say(voice_out.LOST_TARGET.format(self.target))
            self.band.send(STOP)
            self.state, self.turning, self.rescan_requested = State.IDLE, False, True
            return self.cmd
        if self.cmd.direction == Direction.ARRIVED:
            self.say(voice_out.ARRIVED)
            self.band.send(STOP)
            self.state = State.DONE
        return self.cmd

    def _update_target(self, found: list[Detection]) -> Detection | None:
        """Pick this frame's target box, robust to the hand covering the object.

        - Nothing found: keep the last box (object hidden or a missed frame).
        - Hand over the object (hand point inside the last box + COVER_MARGIN): keep the
          last full box. The hand hides part of it, so YOLO sees a shrunken box or two
          fragments (e.g. a phone split down the middle) whose center and size-based
          distance are wrong. Objects don't move while you reach for them.
        - Otherwise: the detection overlapping the last box most (stays on the same
          object when there are fragments or several cans), else the most confident.
        """
        old, hand = self.det, self.hand
        if not found:
            return old
        if old is None:
            return found[0]
        if _covering(hand, old):
            best = found[0]
            return replace(old, confidence=best.confidence, frame_id=best.frame_id)
        overlapping = [d for d in found if _iou(d, old) > 0]
        return max(overlapping, key=lambda d: _iou(d, old)) if overlapping else found[0]

    def _track_missing(self, found: list[Detection]) -> None:
        """Count frames the target is out of sight. Under the hand doesn't count."""
        old, hand = self.det, self.hand
        if found or (old is not None and _covering(hand, old)):
            self._missing, self._missing_since = 0, None
            return
        self._missing += 1
        if self._missing_since is None:
            self._missing_since = self.clock()

    def _check_lost(self) -> None:
        if self.cmd.direction != Direction.LOST:
            self._lost_since, self._warned_lost = None, False
            return
        now = self.clock()
        if self._lost_since is None:
            self._lost_since = now
        elif not self._warned_lost and now - self._lost_since > HAND_LOST_AFTER_S:
            self.say(voice_out.HAND_LOST)
            self._warned_lost = True

    def cancel(self) -> None:
        """Abandon the current guidance (user asked for something else)."""
        self.band.send(STOP)
        self.state = State.IDLE
        self.cmd = None

    def stop(self) -> None:
        """Always safe to call: motors off, band closed."""
        try:
            self.band.send(STOP)
        finally:
            self.band.close()
