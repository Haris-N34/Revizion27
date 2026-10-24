"""OpenCV overlay: target box, hand skeleton, fingertip-to-target line, status."""

import cv2
import numpy as np

from shared.contracts import Detection, GuidanceCommand, HandPosition

try:
    from vision.hands.draw import draw_hand
except ImportError:  # hands branch not merged yet: fingertip dot only
    draw_hand = None

GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def _text(frame, text, org, color=RED, scale=0.7):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def render(
    frame: np.ndarray,
    status: str,
    det: Detection | None = None,
    hand: HandPosition | None = None,
    landmarks=None,
    cmd: GuidanceCommand | None = None,
    fps: float = 0.0,
    hint: str = "r: rescan   q: quit",
) -> np.ndarray:
    """Draw onto a copy of frame and return it."""
    out = frame.copy()
    if det is not None:
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(out, p1, p2, GREEN, 2)
        _text(out, f"{det.label} {det.confidence:.2f}", (p1[0], max(20, p1[1] - 8)), GREEN, 0.6)
    if landmarks and draw_hand is not None:
        draw_hand(out, landmarks)
    if hand is not None:
        tip = (int(hand.x), int(hand.y))
        cv2.circle(out, tip, 9, RED, -1, cv2.LINE_AA)
        if det is not None:
            center = tuple(int(v) for v in det.center)
            cv2.arrowedLine(out, tip, center, YELLOW, 2, cv2.LINE_AA, tipLength=0.05)
    _text(out, f"{status}   {fps:.0f} fps", (10, 28))
    if cmd is not None:
        line = f"band: {cmd.direction.name} ({cmd.direction.value},{cmd.intensity})"
        _text(out, f"{line}  {cmd.distance_px:.0f}px", (10, 58), YELLOW)
    _text(out, hint, (10, out.shape[0] - 12), WHITE, 0.55)
    return out
