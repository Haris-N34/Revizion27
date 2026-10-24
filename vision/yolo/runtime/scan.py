"""scan(): look for a target across frames until it is confirmed or time runs out."""

import time
from collections.abc import Callable, Iterable

import numpy as np

from shared.contracts import Detection

DEFAULT_CONFIRM_FRAMES = 3


def scan(
    frames: Iterable[np.ndarray],
    detector,
    target: str,
    timeout_s: float,
    *,
    confirm_frames: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Detection | None:
    """Return the target once it is seen in `confirm_frames` consecutive frames.

    Returns None if `timeout_s` passes or the frames run out first. Never returns
    an unconfirmed guess. `confirm_frames` defaults to detector.confirm_frames.
    """
    if confirm_frames is None:
        confirm_frames = getattr(detector, "confirm_frames", DEFAULT_CONFIRM_FRAMES)
    confirm_frames = max(1, int(confirm_frames))

    deadline = clock() + timeout_s
    streak = 0
    for frame in frames:
        if clock() > deadline:
            return None
        detections = detector.detect(frame, target)
        if not detections:
            streak = 0
            continue
        streak += 1
        if streak >= confirm_frames:
            return detections[0]
    return None
