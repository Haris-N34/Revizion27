"""python -m app.main [--target can] [--mock-band] [--video FILE] [--no-voice] [--no-window]

Scans for the target, then keeps a live view: target box, hand skeleton, the line
from fingertip to target and the band command. r rescans, q or Esc quits.
"""

import argparse
import time

import cv2

from app import voice_out
from app.debug_view import render
from app.fakes import FakeHandTracker, PrintBand, fake_compute
from app.loop import Guide, State
from shared.config import load_config
from vision.yolo.runtime.detector import ObjectDetector

WINDOW = "octaby"


class Quit(Exception):
    pass


def build(config: dict, mock_band: bool):
    """Real modules where available, fakes otherwise. Returns parts + names of fakes."""
    fakes = []
    try:
        from vision.hands.tracker import HandTracker

        tracker = HandTracker(config)
    except ImportError:
        tracker = FakeHandTracker(config)
        fakes.append("FakeHandTracker")
    try:
        from guidance.compute import compute
    except ImportError:
        compute = fake_compute
        fakes.append("fake_compute")
    try:
        from hardware.host.band import Band, MockBand

        band = MockBand(config) if mock_band or not config.get("serial_port") else Band(config)
    except ImportError:
        band = PrintBand(config)
        fakes.append("PrintBand")
    try:
        from vision.yolo.model.classes import normalize
    except ImportError:

        def normalize(spoken: str) -> str | None:
            return spoken.strip().lower() or None

        fakes.append("normalize (lowercase only)")
    return ObjectDetector(config), tracker, compute, band, normalize, fakes


class Viewer:
    """Owns the window: shows frames, tracks fps, handles keys."""

    def __init__(self, show: bool):
        self.show = show
        self.fps = 0.0
        self._last = time.perf_counter()
        self.rescan = False

    def tick(self) -> None:
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt

    def show_frame(self, image) -> None:
        self.tick()
        if not self.show:
            return
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise Quit
        if key == ord("r"):
            self.rescan = True


class ScanView:
    """Wraps the detector during scan() so each scanned frame is drawn live."""

    def __init__(self, detector, tracker, viewer: Viewer, target: str):
        self.detector, self.tracker, self.viewer, self.target = detector, tracker, viewer, target
        self.confirm_frames = detector.confirm_frames

    def detect(self, frame, target):
        dets = self.detector.detect(frame, target)
        hand = self.tracker.track(frame)
        landmarks = getattr(self.tracker, "landmarks", None)
        status = f"scanning for {self.target}"
        image = render(
            frame, status, dets[0] if dets else None, hand, landmarks, fps=self.viewer.fps
        )
        self.viewer.show_frame(image)
        return dets


def read_frames(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def ask_target(args, normalize) -> str | None:
    spoken = args.target or input("What should I find? ").strip()
    target = normalize(spoken)
    if target is None:
        voice_out.speak(voice_out.CANT_LOOK)
    return target


def run(args) -> None:
    config = load_config()
    detector, tracker, compute, band, normalize, fakes = build(config, args.mock_band)
    if fakes:
        print(f"[app] fakes active: {', '.join(fakes)}")
    if not args.no_voice:
        print("[app] voice not wired up yet; using keyboard / --target")

    target = ask_target(args, normalize)
    if target is None:
        return
    source = args.video if args.video else int(config["camera_index"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source!r}")

    viewer = Viewer(show=not args.no_window)
    guide = Guide(config, detector, tracker, compute, band)
    try:
        while True:
            viewer.rescan = False
            view = ScanView(detector, tracker, viewer, target)
            found = guide.find(read_frames(cap), target, detector=view)
            if found and isinstance(tracker, FakeHandTracker):
                tracker.goal = found.center
            if not live(cap, guide, viewer, target):
                break
    except (Quit, KeyboardInterrupt):
        pass
    finally:
        guide.stop()
        if hasattr(tracker, "close"):
            tracker.close()
        cap.release()
        cv2.destroyAllWindows()


def live(cap, guide: Guide, viewer: Viewer, target: str) -> bool:
    """Keep the view running after scanning. Returns True to rescan, False when video ends."""
    for frame in read_frames(cap):
        if guide.state == State.GUIDING:
            guide.step(frame)
        else:
            # NOT_FOUND / DONE: keep showing the hand and the target live, no band output.
            found = guide.detector.detect(frame, target)
            guide.det = found[0] if found else guide.det
            guide.hand = guide.tracker.track(frame)
        status = {
            State.GUIDING: f"guiding to {target}",
            State.DONE: "ARRIVED - you're there",
            State.NOT_FOUND: f"NOT FOUND: {target}",
        }.get(guide.state, guide.state.value)
        landmarks = getattr(guide.tracker, "landmarks", None)
        cmd = guide.cmd if guide.state == State.GUIDING else None
        image = render(frame, status, guide.det, guide.hand, landmarks, cmd, viewer.fps)
        viewer.show_frame(image)
        if viewer.rescan:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="object to find; asked on the keyboard if omitted")
    parser.add_argument("--video", help="video file instead of the webcam")
    parser.add_argument("--mock-band", action="store_true", help="print band commands")
    parser.add_argument("--no-voice", action="store_true", help="keyboard in, printed speech")
    parser.add_argument("--no-window", action="store_true", help="no OpenCV window")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
