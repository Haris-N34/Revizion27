"""Small background-motion check used to invalidate stale screen positions."""

import cv2
import numpy as np


class SceneMotion:
    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, frame, exclude=()):
        """Return background translation in frame fractions, or None if uncertain.

        Mask the hand and target so reaching alone doesn't count as camera motion.
        This is a camera-motion hint, not a position or depth measurement.
        """
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        previous, self.previous = self.previous, gray
        if previous is None:
            return None
        mask = np.full(gray.shape, 255, np.uint8)
        height, width = frame.shape[:2]
        for box in exclude:
            if box is None:
                continue
            x1, y1, x2, y2 = box
            a, b = max(0, int(x1 * 160 / width) - 5), max(0, int(y1 * 120 / height) - 5)
            c, d = min(160, int(x2 * 160 / width) + 6), min(120, int(y2 * 120 / height) + 6)
            mask[b:d, a:c] = 0

        def uncertain():
            background = cv2.absdiff(previous, gray)[mask > 0]
            if background.size and np.mean(background > 25) > 0.5 and np.mean(background) > 35:
                return (1.0, 1.0)  # scene changed too much to trust the old box
            return None

        points = cv2.goodFeaturesToTrack(previous, 60, 0.02, 8, mask=mask)
        if points is None or len(points) < 8:
            return uncertain()
        moved, valid, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None, winSize=(11, 11), maxLevel=2
        )
        if moved is None:
            return uncertain()
        shifts = (moved - points).reshape(-1, 2)[valid.ravel() == 1]
        if len(shifts) < 8:
            return uncertain()
        median = np.median(shifts, axis=0)
        # A majority of the background must agree; local hand motion is not enough.
        if np.mean(np.linalg.norm(shifts - median, axis=1) < 2) < 0.65:
            return uncertain()
        return float(median[0] / 160), float(median[1] / 120)
