"""Narrator: speaks guidance directions ("left", "up right") while guiding.

A stand-in for the band (and a demo fallback): fed the stabilized command each
frame, it speaks when the direction changes (at most every MIN_GAP_S) and repeats
the current one every REPEAT_S. It never queues: if the speaker is still talking it
waits, so speech can't fall behind the hand. ARRIVED / LOST have their own lines in
voice_out and are not narrated here.
"""

import time
from collections.abc import Callable

from shared.contracts import Direction, GuidanceCommand

MIN_GAP_S = 1.2
REPEAT_S = 3.0

WORDS = {
    Direction.UP: "up",
    Direction.DOWN: "down",
    Direction.LEFT: "left",
    Direction.RIGHT: "right",
    Direction.UP_LEFT: "up left",
    Direction.UP_RIGHT: "up right",
    Direction.DOWN_LEFT: "down left",
    Direction.DOWN_RIGHT: "down right",
    Direction.FORWARD: "forward",
    Direction.BACK: "back",
}


class Narrator:
    def __init__(
        self,
        speaker,
        min_gap_s: float = MIN_GAP_S,
        repeat_s: float = REPEAT_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        """`speaker`: object with say(text) and busy() (voice_out.Speaker)."""
        self.speaker = speaker
        self.min_gap_s = min_gap_s
        self.repeat_s = repeat_s
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self._last_dir: Direction | None = None
        self._last_time = float("-inf")

    def update(self, cmd: GuidanceCommand | None) -> str | None:
        """Call once per guiding frame; returns the word spoken, if any."""
        if cmd is None or cmd.direction not in WORDS:
            # LOST / ARRIVED / STOP: remember it so the next direction counts as new.
            self._last_dir = cmd.direction if cmd else None
            return None
        if self.speaker.busy():
            return None
        now = self.clock()
        since = now - self._last_time
        changed = cmd.direction != self._last_dir
        if (changed and since >= self.min_gap_s) or since >= self.repeat_s:
            word = WORDS[cmd.direction]
            self.speaker.say(word)
            self._last_dir, self._last_time = cmd.direction, now
            return word
        return None
