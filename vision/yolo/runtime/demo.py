"""python -m vision.yolo.runtime.demo --target can [--video FILE] [--no-window]

Scans the webcam (or a video) for the target, draws boxes, prints FOUND or NOT FOUND.
"""

import argparse
import time

import cv2

from shared.config import load_config
from vision.yolo.runtime.detector import ObjectDetector
from vision.yolo.runtime.scan import scan

WINDOW = "octaby yolo"


def read_frames(cap: cv2.VideoCapture):
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def draw(frame, detections, status: str, fps: float):
    for d in detections:
        cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{d.label} {d.confidence:.2f}",
            (int(d.x1), max(15, int(d.y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        frame, f"{status}  {fps:.1f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )
    return frame


class DrawingDetector:
    """Wraps a detector so every scanned frame is shown with its boxes and FPS."""

    def __init__(self, detector, show: bool):
        self.detector = detector
        self.confirm_frames = detector.confirm_frames
        self.show = show
        self.fps = 0.0
        self.last_frame = None

    def detect(self, frame, target):
        start = time.perf_counter()
        detections = self.detector.detect(frame, target)
        dt = time.perf_counter() - start
        self.fps = 0.9 * self.fps + 0.1 * (1 / dt) if self.fps else 1 / dt
        self.last_frame = draw(frame.copy(), detections, f"scanning for {target}", self.fps)
        if self.show:
            cv2.imshow(WINDOW, self.last_frame)
            cv2.waitKey(1)
        return detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="can")
    parser.add_argument("--video", help="video file instead of the webcam")
    parser.add_argument("--timeout", type=float, help="override scan_timeout_s")
    parser.add_argument("--no-window", action="store_true", help="print only, no OpenCV window")
    args = parser.parse_args()

    config = load_config()
    timeout = args.timeout if args.timeout is not None else float(config["scan_timeout_s"])
    source = args.video if args.video else int(config["camera_index"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source!r}")

    detector = DrawingDetector(ObjectDetector(config), show=not args.no_window)
    try:
        # Warm up: the first inference is slow and shouldn't eat the scan timeout.
        ok, frame = cap.read()
        if ok:
            detector.detector.detect(frame, args.target)

        print(f"Scanning for '{args.target}' for up to {timeout:.1f}s ...")
        found = scan(read_frames(cap), detector, args.target, timeout)
        if found:
            cx, cy = found.center
            print(
                f"FOUND {found.label} conf={found.confidence:.2f} "
                f"center=({cx:.0f}, {cy:.0f}) frame={found.frame_id} ~{detector.fps:.1f} fps"
            )
        else:
            print(f"NOT FOUND: I can't find a {args.target}  ~{detector.fps:.1f} fps")

        if not args.no_window and detector.last_frame is not None:
            frame = detector.last_frame
            label = "FOUND" if found else "NOT FOUND"
            cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow(WINDOW, frame)
            print("Press any key in the window to exit.")
            cv2.waitKey(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
