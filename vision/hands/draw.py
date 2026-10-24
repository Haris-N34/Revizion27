"""Draw a hand skeleton from HandTracker.landmarks (21 pixel points)."""

import cv2
import numpy as np

# MediaPipe hand topology: thumb, index, middle, ring, pinky, palm.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]  # fmt: skip


def draw_hand(
    frame: np.ndarray,
    landmarks: list[tuple[float, float]] | None,
    color: tuple[int, int, int] = (255, 200, 0),
    grab: tuple[float, float] | None = None,
) -> np.ndarray:
    """Draw bones and joints in place, and a big red dot on the grab point
    (HandPosition.x, y; defaults to the palm center)."""
    if not landmarks:
        return frame
    pts = [(int(x), int(y)) for x, y in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
    if grab is None:
        palm = [pts[i] for i in (0, 5, 9, 13, 17)]
        grab = (sum(p[0] for p in palm) / 5, sum(p[1] for p in palm) / 5)
    cv2.circle(frame, (int(grab[0]), int(grab[1])), 9, (0, 0, 255), -1, cv2.LINE_AA)
    return frame
