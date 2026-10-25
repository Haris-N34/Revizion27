"""The find-and-guide state machine. No camera or window here; main.py feeds frames."""

import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from enum import Enum

import numpy as np

from app import voice_out
from app.scene_motion import SceneMotion
from shared.contracts import Detection, Direction, GuidanceCommand, HandPosition
from vision.yolo.runtime.scan import scan

STOP = GuidanceCommand(Direction.STOP, 0, 0.0)
HAND_LOST_AFTER_S = 2.0
COVER_MARGIN = 0.15  # hand point within the box grown by 15% counts as covering it
COVER_OVERLAP = 0.3  # ...or the hand's outline covering 30% of the box
MISSING_FRAMES = 8  # target unseen this many frames (not under the hand) = out of view
TARGET_LOST_S = 6.0  # out of view this long: give up and scan again
TARGET_MEMORY_S = 1.0  # tolerate brief detector misses without discarding the box
OCCLUSION_MEMORY_S = 6.0  # bounded memory while the hand covers the remembered box
CONTACT_HOLD_S = 0.6  # fresh, depth-supported arrival must persist this long
MISSING_NOTICE_S = 0.6
SEARCH_REMINDER_S = 8.0
REACQUIRE_HOLD_S = 1.0
REACQUIRE_FRAMES = 3
NOTICE_GAP_S = 8.0
HAND_RECOVERY_S = 1.0


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
        self.visibility = "searching"
        self._motion = SceneMotion()
        self._camera_drift = np.zeros(2)
        self._memory_moved = False
        self._last_notice = None
        self._notice_at = float("-inf")
        self._target_notice_at = float("-inf")
        self._hand_notice_at = float("-inf")
        self._visible_since = None
        self._visible_frames = 0
        self._hand_visible_since = None
        self._depth_warned = False

    def start(self, target: str, announce: bool = True) -> None:
        """Continuously look for a target until the user changes it or quits."""
        self.cancel()
        self.target, self.det, self.hand = target, None, None
        self._confirmed = 0
        self._seen_at, self._contact_since = None, None
        self._search_started = self.clock()
        self._announced_found = self._announced_not_found = False
        self.visibility = "searching"
        self._motion.reset()
        self._camera_drift[:] = 0
        self._memory_moved = False
        self._missing_since = None
        self._last_notice = None
        self._notice_at = float("-inf")
        self._target_notice_at = self._hand_notice_at = float("-inf")
        self._visible_since = self._hand_visible_since = None
        self._visible_frames = 0
        self._depth_warned = False
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
        old = self.det
        exclusions = [self.hand.box if self.hand else None]
        if old is not None:
            exclusions.append((old.x1, old.y1, old.x2, old.y2))
        shift = self._motion.update(frame, exclusions)
        covering = self.det is not None and _covering(self.hand, self.det)
        if found:
            if self._visible_since is None:
                self._visible_since = now
            self._visible_frames += 1
            self._seen_at = now
            self._camera_drift[:] = 0
            self._memory_moved = False
            self._missing_since = None
            stable_return = (
                self._visible_frames >= REACQUIRE_FRAMES
                and now - self._visible_since >= REACQUIRE_HOLD_S
            )
            if stable_return and self._last_notice in ("covered", "search"):
                # Lifting a covering hand needs no "found again" announcement.
                # Quick returns after a search warning stay quiet as well.
                if self._last_notice == "search" and now - self._target_notice_at >= NOTICE_GAP_S:
                    self.say(voice_out.TARGET_AGAIN.format(self.target))
                    self._target_notice_at = now
                self._last_notice = None
            self.visibility = "visible"
        else:
            self._visible_since, self._visible_frames = None, 0
            if self._missing_since is None:
                self._missing_since = now
            if shift is not None:
                self._camera_drift += shift
                if np.linalg.norm(self._camera_drift) > 0.025:
                    self._memory_moved = True
            covering = covering and not self._memory_moved
        age = now - self._seen_at if self._seen_at is not None else float("inf")
        memory = OCCLUSION_MEMORY_S if covering else TARGET_MEMORY_S
        if not found and self._announced_found:
            covered = covering and age <= memory
            self.visibility = "occluded" if covered else "missing"
            if now - self._missing_since >= MISSING_NOTICE_S:
                notice = "covered" if covered else "search"
                reminder = notice == "search" and now - self._notice_at >= SEARCH_REMINDER_S
                if (notice != self._last_notice or reminder) and now - self._target_notice_at >= NOTICE_GAP_S:
                    text = (
                        voice_out.TARGET_COVERED.format(self.target)
                        if covered else self._search_message(old, frame.shape)
                    )
                    self.say(text)
                    self._last_notice, self._notice_at = notice, now
                    self._target_notice_at = now
        if not found and self._memory_moved:
            memory = 0  # stale image coordinates are unsafe after the camera moves
        if not found and (self.det is None or age > memory):
            self.det, self.cmd = None, STOP
            self._confirmed = 0
            self._contact_since = None
            self.state = State.SCANNING
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
            self.state = State.SCANNING
            self.cmd = STOP
            self.band.send(STOP)
            return self.cmd
        if not found and covering:
            # Bounded hand-occlusion memory drives only the forward actuator.
            # Fresh detection, hand loss, camera motion or expiry ends this path.
            self._contact_since = None
            if self.stabilizer is not None:
                self.stabilizer.reset()
            self.cmd = GuidanceCommand(Direction.FORWARD, 100, 0.0)
            self.state = State.GUIDING
            self.band.send(self.cmd)
            return self.cmd
        self.cmd = self.compute(self.det, self.hand, frame.shape, self.config)
        # Overlap only proves alignment in the image. It must never override a
        # FORWARD/BACK command or confirm arrival from a remembered detection.
        contact = self.cmd.direction == Direction.ARRIVED
        depth_known = (
            self.hand is not None and self.det.distance_m is not None
            and self.hand.distance_m is not None and self.det.distance_m > 0
        )
        tolerance = float((self.config.get("guidance") or {}).get("depth_tolerance", 0.15))
        depth_close = depth_known and (
            abs(self.det.distance_m - self.hand.distance_m) <= self.det.distance_m * tolerance
        )
        candidate = None
        if found:
            candidate = max(found, key=lambda d: _iou(d, old)) if old else found[0]
        fresh_full_box = candidate is not None and (
            old is None or (
                (candidate.x2 - candidate.x1) * (candidate.y2 - candidate.y1)
                >= 0.8 * (old.x2 - old.x1) * (old.y2 - old.y1)
            )
        )
        if contact and fresh_full_box and depth_close:
            if self._contact_since is None:
                self._contact_since = now
            if not arrived and now - self._contact_since < CONTACT_HOLD_S:
                self.cmd = STOP
        else:
            self._contact_since = None
            if contact:
                self.cmd = STOP
                if found and not self._depth_warned:
                    self.say(voice_out.CHECK_DEPTH)
                    self._depth_warned = True
        arrival_allowed = self.cmd.direction == Direction.ARRIVED
        if self.stabilizer is not None:
            if self.cmd.direction == Direction.STOP:
                self.stabilizer.reset()
            else:
                self.cmd = self.stabilizer.update(self.cmd)
            if self.cmd.direction == Direction.ARRIVED and not arrival_allowed:
                self.cmd = STOP
                self.stabilizer.reset()
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

    def _search_message(self, last, frame_shape):
        if last is not None and not self._memory_moved:
            width = frame_shape[1]
            side = None
            if last.x1 <= width * 0.08:
                side = "left"
            elif last.x2 >= width * 0.92:
                side = "right"
            if side:
                if self.config.get("mirror"):
                    side = "right" if side == "left" else "left"
                return voice_out.LOOK_BACK.format(self.target, side)
        return voice_out.LOOK_AROUND.format(self.target)

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
        overlapping = [d for d in found if _iou(d, old) > 0]
        if _covering(hand, old) and overlapping:
            best = max(overlapping, key=lambda d: _iou(d, old))
            area = (best.x2 - best.x1) * (best.y2 - best.y1)
            if area < 0.8 * (old.x2 - old.x1) * (old.y2 - old.y1):
                return replace(old, confidence=best.confidence, frame_id=best.frame_id)
            return best
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
        now = self.clock()
        if self.cmd.direction != Direction.LOST:
            self._lost_since = None
            if self.hand is not None:
                if self._hand_visible_since is None:
                    self._hand_visible_since = now
                if now - self._hand_visible_since >= HAND_RECOVERY_S:
                    self._warned_lost = False
            else:
                self._hand_visible_since = None
            return
        self._hand_visible_since = None
        if self._lost_since is None:
            self._lost_since = now
        elif (
            not self._warned_lost and now - self._lost_since > HAND_LOST_AFTER_S
            and now - self._hand_notice_at >= NOTICE_GAP_S
        ):
            self.say(voice_out.HAND_LOST)
            self._warned_lost = True
            self._hand_notice_at = now

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
