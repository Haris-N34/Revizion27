"""Stand-ins that satisfy the root contracts, so the app runs before other branches merge."""

import math

import numpy as np

from shared.contracts import Detection, Direction, GuidanceCommand, HandPosition


class FakeDetector:
    """Returns a fixed box from the `found_after`-th call on; never if found_after is None."""

    def __init__(self, config: dict | None = None, found_after: int | None = 0, label="can"):
        self.confirm_frames = 3
        self.found_after = found_after
        self.label = label
        self.calls = 0

    def detect(self, frame: np.ndarray, target: str) -> list[Detection]:
        frame_id = self.calls
        self.calls += 1
        if self.found_after is None or frame_id < self.found_after or target != self.label:
            return []
        h, w = frame.shape[:2]
        return [Detection(self.label, 0.9, w * 0.6, h * 0.5, w * 0.7, h * 0.8, frame_id)]


class FakeHandTracker:
    """A fingertip that starts bottom-left and moves `speed` px per frame toward `goal`."""

    def __init__(self, config: dict | None = None, speed: float = 40.0, visible: bool = True):
        self.speed = speed
        self.visible = visible
        self.goal: tuple[float, float] | None = None
        self.pos: tuple[float, float] | None = None
        self.landmarks = None
        self.calls = 0

    def track(self, frame: np.ndarray) -> HandPosition | None:
        frame_id = self.calls
        self.calls += 1
        if not self.visible:
            return None
        h, w = frame.shape[:2]
        if self.pos is None:
            self.pos = (w * 0.1, h * 0.9)
        if self.goal:
            dx, dy = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
            dist = math.hypot(dx, dy)
            step = min(self.speed, dist)
            if dist:
                self.pos = (self.pos[0] + dx / dist * step, self.pos[1] + dy / dist * step)
        x, y = self.pos
        return HandPosition(x, y, x, y + 60, "Right", frame_id)

    def close(self) -> None:
        pass


_DIRS = {
    (0, -1): Direction.UP,
    (0, 1): Direction.DOWN,
    (-1, 0): Direction.LEFT,
    (1, 0): Direction.RIGHT,
    (-1, -1): Direction.UP_LEFT,
    (1, -1): Direction.UP_RIGHT,
    (-1, 1): Direction.DOWN_LEFT,
    (1, 1): Direction.DOWN_RIGHT,
}


def fake_compute(det: Detection, hand: HandPosition | None, frame_shape, config: dict):
    """Simple stand-in for guidance.compute: 8 directions, intensity grows with distance."""
    if hand is None:
        return GuidanceCommand(Direction.LOST, 0, 0.0)
    cx, cy = det.center
    dx, dy = cx - hand.x, cy - hand.y
    dist = math.hypot(dx, dy)
    g = config.get("guidance", {})
    if dist <= g.get("arrive_px", 40):
        return GuidanceCommand(Direction.ARRIVED, 255, dist)
    if config.get("mirror"):
        dx = -dx
    dead = g.get("dead_zone_px", 25)
    sx = 0 if abs(dx) < dead else (1 if dx > 0 else -1)
    sy = 0 if abs(dy) < dead else (1 if dy > 0 else -1)
    diag = math.hypot(frame_shape[1], frame_shape[0])
    intensity = int(min(255, 80 + 175 * dist / diag))
    return GuidanceCommand(_DIRS.get((sx, sy), Direction.STOP), intensity, dist)


class PrintBand:
    """Band stand-in: prints each command once when it changes."""

    def __init__(self, config: dict | None = None):
        self.sent: list[GuidanceCommand] = []
        self.closed = False
        self._last = None

    def send(self, cmd: GuidanceCommand) -> None:
        self.sent.append(cmd)
        line = f"{cmd.direction.value},{cmd.intensity}"
        if cmd.direction != self._last:
            print(f"[band] {line}")
            self._last = cmd.direction

    def close(self) -> None:
        self.closed = True
