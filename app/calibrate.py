"""python -m app.calibrate [--video FILE] [--image FILE --cm 50]

Calibrates camera_hfov_deg (the camera's field of view), which turns pixel sizes into
distances. Needs a standard 355 ml can and a tape measure. Once per camera.

1. Stand the can upright, fully in view, facing the camera.
2. Measure from the camera lens to the FRONT of the can.
3. Press SPACE, type that distance in cm in the terminal, Enter.
   Do it at 2-3 distances (e.g. 40, 60, 80 cm); the result is the median.
4. Press w to save to config.yaml, q to quit.
The window shows the estimated distance with the current value, to check against the tape.
"""

import argparse
import math
import re
import shutil
import statistics
from collections import deque
from pathlib import Path

import cv2

from app.main import camera_source
from shared.config import CONFIG_PATH, REPO_ROOT, load_config
from vision.yolo.runtime.detector import ObjectDetector, focal_px

CAN_HEIGHT_M = 0.122
CAN_RADIUS_M = 0.033  # tape goes to the can's front; the box height is at its middle
WINDOW = "octaby calibrate"
EXAMPLE_CONFIG = REPO_ROOT / "shared" / "config.example.yaml"


def hfov_from_sample(front_distance_m: float, box_h_px: float, frame_width: int) -> float:
    """Field of view (degrees) that makes a can box_h_px tall sit at that distance."""
    distance = front_distance_m + CAN_RADIUS_M
    focal = distance * box_h_px / CAN_HEIGHT_M
    return math.degrees(2 * math.atan((frame_width / 2) / focal))


def can_distance_m(box_h_px: float, frame_width: int, hfov_deg: float) -> float:
    """Distance to the can's front with a given field of view (inverse of the above)."""
    return focal_px(frame_width, hfov_deg) * CAN_HEIGHT_M / box_h_px - CAN_RADIUS_M


def save_hfov(hfov_deg: float, path: Path = CONFIG_PATH) -> None:
    """Set camera_hfov_deg in config.yaml, keeping every other line and comment.
    Creates config.yaml from the example if it doesn't exist."""
    if not path.exists():
        shutil.copy(EXAMPLE_CONFIG, path)
    text = path.read_text(encoding="utf-8")
    line = f"camera_hfov_deg: {hfov_deg:.1f}"
    if re.search(r"^camera_hfov_deg:.*$", text, flags=re.MULTILINE):
        text = re.sub(r"^camera_hfov_deg:.*$", line, text, flags=re.MULTILINE)
    else:
        text = text.rstrip("\n") + f"\n\n# Set by python -m app.calibrate\n{line}\n"
    path.write_text(text, encoding="utf-8")


def _can_height(detector: ObjectDetector, frame) -> float | None:
    dets = detector.detect(frame, "can")
    return (dets[0].y2 - dets[0].y1) if dets else None


def _draw(frame, heights, samples, hfov_now: float, hfov_new: float | None) -> None:
    lines = []
    if heights:
        h = statistics.median(heights)
        dist = can_distance_m(h, frame.shape[1], hfov_now) * 100
        lines.append(f"can {h:.0f}px tall -> {dist:.0f} cm (with {hfov_now:.1f} deg)")
    else:
        lines.append("no can seen: stand it upright, fully in view")
    lines.append(
        f"samples: {len(samples)}" + (f"   new fov: {hfov_new:.1f} deg" if hfov_new else "")
    )
    lines.append("SPACE: add sample   w: save   q: quit")
    for i, text in enumerate(lines):
        org = (10, 30 + 32 * i)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def run_live(config: dict, video: str | None) -> None:
    detector = ObjectDetector(config)
    hfov_now = float(config.get("camera_hfov_deg", 70.0))
    cap = cv2.VideoCapture(video if video else camera_source(config))
    if not cap.isOpened():
        raise SystemExit("Could not open the camera")
    heights: deque[float] = deque(maxlen=15)  # median of recent frames: steadier than one
    samples: list[float] = []
    hfov_new = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detector.detect(frame, "can")
            if dets:
                d = dets[0]
                heights.append(d.y2 - d.y1)
                cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
            else:
                heights.clear()
            _draw(frame, heights, samples, hfov_new or hfov_now, hfov_new)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if len(heights) < 5:
                    print("Can not seen steadily yet; hold still and try again.")
                    continue
                height = statistics.median(heights)
                cm = float(input(f"Can is {height:.0f}px tall. Distance lens -> can front (cm): "))
                samples.append(hfov_from_sample(cm / 100, height, frame.shape[1]))
                hfov_new = statistics.median(samples)
                print(f"sample {len(samples)}: {samples[-1]:.1f} deg -> using {hfov_new:.1f} deg")
            if key == ord("w") and hfov_new:
                save_hfov(hfov_new)
                print(f"Saved camera_hfov_deg: {hfov_new:.1f} to {CONFIG_PATH.name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", help="video file instead of the camera")
    parser.add_argument("--image", help="one still image (with --cm): print the result only")
    parser.add_argument("--cm", type=float, help="lens -> can front distance for --image")
    args = parser.parse_args()
    config = load_config()
    if args.image:
        frame = cv2.imread(args.image)
        h = _can_height(ObjectDetector(config), frame)
        if h is None or args.cm is None:
            raise SystemExit("need a visible can and --cm")
        print(
            f"can {h:.0f}px tall at {args.cm:.0f} cm -> camera_hfov_deg {hfov_from_sample(args.cm / 100, h, frame.shape[1]):.1f}"
        )
        return
    run_live(config, args.video)


if __name__ == "__main__":
    main()
