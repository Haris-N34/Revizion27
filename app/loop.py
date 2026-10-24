"""The find-and-guide state machine. No camera or window here; main.py feeds frames."""

import time
from collections.abc import Callable, Iterable
from enum import Enum

import numpy as np

from app import voice_out
from shared.contracts import Detection, Direction, GuidanceCommand, HandPosition
from vision.yolo.runtime.scan import scan

STOP = GuidanceCommand(Direction.STOP, 0, 0.0)
HAND_LOST_AFTER_S = 2.0


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
    ):
        self.config = config
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

    def find(self, frames: Iterable[np.ndarray], target: str, detector=None) -> Detection | None:
        """SCANNING: returns the detection (state GUIDING) or None (state NOT_FOUND).

        `detector` overrides self.detector for the scan only (e.g. a drawing wrapper).
        """
        self.target, self.det, self.hand, self.cmd = target, None, None, None
        self._lost_since, self._warned_lost = None, False
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
        if found:
            self.det = found[0]  # else keep the last known box
        self.hand = self.tracker.track(frame)
        self.cmd = self.compute(self.det, self.hand, frame.shape, self.config)
        self.band.send(self.cmd)
        self._check_lost()
        if self.cmd.direction == Direction.ARRIVED:
            self.say(voice_out.ARRIVED)
            self.band.send(STOP)
            self.state = State.DONE
        return self.cmd

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
