"""Stabilizer: smooths compute() output over frames so the band doesn't flicker.

compute() stays pure; this holds the little bit of memory guidance needs:
- a new direction must repeat `switch_frames` times before it replaces the current one
- LOST must repeat `lost_frames` times (one dropped tracking frame shouldn't buzz)
- ARRIVED must repeat `arrive_frames` times (a hand sweeping past isn't a grab)
Intensity and distance always come from the latest frame.
"""

from dataclasses import replace

from shared.contracts import Direction, GuidanceCommand

DEFAULT_SWITCH_FRAMES = 3
DEFAULT_LOST_FRAMES = 5
DEFAULT_ARRIVE_FRAMES = 6


class Stabilizer:
    def __init__(self, config: dict | None = None):
        g = (config or {}).get("guidance") or {}
        self.switch_frames = int(g.get("switch_frames", DEFAULT_SWITCH_FRAMES))
        self.lost_frames = int(g.get("lost_frames", DEFAULT_LOST_FRAMES))
        self.arrive_frames = int(g.get("arrive_frames", DEFAULT_ARRIVE_FRAMES))
        self.reset()

    def reset(self) -> None:
        self.current: Direction | None = None
        self._last = GuidanceCommand(Direction.STOP, 0, 0.0)
        self._candidate: Direction | None = None
        self._count = 0

    def _needed(self, direction: Direction) -> int:
        if direction == Direction.LOST:
            return self.lost_frames
        if direction == Direction.ARRIVED:
            return self.arrive_frames
        return self.switch_frames

    def update(self, raw: GuidanceCommand) -> GuidanceCommand:
        """Feed one frame's compute() result; returns the command to send."""
        if raw.direction == self.current:
            self._candidate, self._count = None, 0
            self._last = raw
            return raw
        if raw.direction == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = raw.direction, 1

        first = self.current is None and raw.direction != Direction.ARRIVED
        if first or self._count >= self._needed(raw.direction):
            self.current = raw.direction
            self._candidate, self._count = None, 0
            self._last = raw
            return raw
        # Not confirmed yet: repeat the last sent command with the latest distance.
        return replace(self._last, distance_px=raw.distance_px)
