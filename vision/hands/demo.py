"""python -m vision.hands.demo [--video FILE]

Live webcam (or video) with the hand skeleton drawn. Press q or Esc to quit.
"""

import argparse
import time

import cv2

from shared.config import load_config
from vision.hands.draw import draw_hand
from vision.hands.tracker import HandTracker

WINDOW = "octaby hands"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="video file instead of the webcam")
    args = parser.parse_args()

    config = load_config()
    source = args.video if args.video else int(config["camera_index"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source!r}")

    tracker = HandTracker(config)
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            hand = tracker.track(frame)
            dt = time.perf_counter() - start
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt

            draw_hand(frame, tracker.landmarks, grab=(hand.x, hand.y) if hand else None)
            if hand:
                status = f"{hand.handedness} grab=({hand.x:.0f}, {hand.y:.0f})"
            else:
                status = "no hand"
            cv2.putText(
                frame,
                f"{status}  {fps:.0f} fps",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
