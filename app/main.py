"""python -m app.main [--target can] [--mock-band] [--video FILE] [--no-voice] [--no-window]

Live view: target box, hand skeleton, fingertip-to-target line and band command.
SPACE: ask out loud ("where is the can?").  r: rescan.  q / Esc: quit.
Without --target it starts listening right away. --no-voice types instead of talking.
"""

import argparse
import time
from types import SimpleNamespace

import cv2

from app import voice_out
from app.debug_view import render
from app.fakes import FakeHandTracker, PrintBand, fake_compute
from app.loop import Guide, State
from app.voice_in import Listener, extract_target
from shared.config import load_config
from vision.yolo.runtime.detector import ObjectDetector

WINDOW = "octaby"
SPACE, RESCAN, QUIT_KEYS = ord(" "), ord("r"), (ord("q"), 27)


class Quit(Exception):
    pass


def build(config: dict, mock_band: bool) -> SimpleNamespace:
    """Real modules where available, fakes otherwise; .fakes lists the fakes in use."""
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
        from vision.yolo.model.classes import SUPPORTED, normalize
    except ImportError:
        from vision.yolo.runtime.detector import FALLBACK_CLASSES as SUPPORTED

        def normalize(spoken: str) -> str | None:
            phrase = spoken.strip().lower()
            return phrase if phrase in SUPPORTED else None

        fakes.append("normalize (exact labels only)")
    return SimpleNamespace(
        detector=ObjectDetector(config),
        tracker=tracker,
        compute=compute,
        band=band,
        normalize=normalize,
        supported=list(SUPPORTED),
        fakes=fakes,
    )


class Viewer:
    """Owns the window: shows frames, tracks fps, returns the key pressed (-1 if none)."""

    def __init__(self, show: bool):
        self.show = show
        self.fps = 0.0
        self._last = time.perf_counter()

    def show_frame(self, image) -> int:
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt
        if not self.show:
            return -1
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            raise Quit
        return key


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


class App:
    def __init__(self, args):
        self.args = args
        self.config = load_config()
        self.parts = build(self.config, args.mock_band)
        if self.parts.fakes:
            print(f"[app] fakes active: {', '.join(self.parts.fakes)}")
        self.speaker = voice_out.configure(enabled=not args.no_voice)
        self.listener = None
        if not args.no_voice:
            self.listener = Listener(hint_words=self.parts.supported)
            self.listener.preload()
        p = self.parts
        self.guide = Guide(self.config, p.detector, p.tracker, p.compute, p.band, self.speaker.say)
        self.viewer = Viewer(show=not args.no_window)
        self.target: str | None = None
        self.pending: str | None = None  # target to scan for on the next frame

    # --- input -------------------------------------------------------------
    def ask(self) -> None:
        """Start listening (voice) or read the keyboard (--no-voice)."""
        if self.guide.state == State.GUIDING:
            self.guide.cancel()
        if self.listener is None:
            self.handle(input("What should I find? "))
        elif self.listener.state == "idle":
            self.speaker.say(voice_out.LISTENING)
            self.speaker.wait()  # don't record our own voice
            self.listener.start()

    def handle(self, text: str) -> None:
        target = extract_target(text, self.parts.normalize)
        if target:
            self.pending = target
        elif text.strip():
            self.speaker.say(voice_out.CANT_LOOK)
        else:
            self.speaker.say(voice_out.DIDNT_HEAR)

    # --- loop --------------------------------------------------------------
    def status(self) -> str:
        if self.listener and self.listener.state == "recording":
            return "listening..."
        if self.listener and self.listener.state == "transcribing":
            return "thinking..."
        return {
            State.GUIDING: f"guiding to {self.target}",
            State.DONE: "ARRIVED - you're there",
            State.NOT_FOUND: f"NOT FOUND: {self.target}",
            State.IDLE: "press SPACE and ask",
        }.get(self.guide.state, self.guide.state.value)

    def run(self) -> None:
        a = self.args
        if a.target:
            self.handle(a.target)
        else:
            self.ask()
        source = a.video if a.video else int(self.config["camera_index"])
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open {source!r}")
        frames = read_frames(cap)
        try:
            for frame in frames:
                if self.pending:
                    self.scan(frames)
                    continue
                self.frame(frame)
        except (Quit, KeyboardInterrupt):
            pass
        finally:
            self.guide.stop()
            if hasattr(self.parts.tracker, "close"):
                self.parts.tracker.close()
            cap.release()
            cv2.destroyAllWindows()
            self.speaker.close()

    def scan(self, frames) -> None:
        self.target, self.pending = self.pending, None
        view = ScanView(self.parts.detector, self.parts.tracker, self.viewer, self.target)
        found = self.guide.find(frames, self.target, detector=view)
        if found and isinstance(self.parts.tracker, FakeHandTracker):
            self.parts.tracker.goal = found.center

    def frame(self, frame) -> None:
        g = self.guide
        if g.state == State.GUIDING:
            g.step(frame)
        else:
            # Not guiding: keep showing the hand (and target, if any) live; no band output.
            if self.target:
                found = g.detector.detect(frame, self.target)
                g.det = found[0] if found else g.det
            g.hand = g.tracker.track(frame)
        landmarks = getattr(g.tracker, "landmarks", None)
        cmd = g.cmd if g.state == State.GUIDING else None
        hint = "SPACE: ask   r: rescan   q: quit"
        image = render(frame, self.status(), g.det, g.hand, landmarks, cmd, self.viewer.fps, hint)
        key = self.viewer.show_frame(image)

        if key == SPACE:
            self.ask()
        elif key == RESCAN and self.target:
            self.pending = self.target
        if self.listener:
            text = self.listener.result()
            if text is not None:
                self.handle(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="object to find; otherwise ask by voice/keyboard")
    parser.add_argument("--video", help="video file instead of the webcam")
    parser.add_argument("--mock-band", action="store_true", help="print band commands")
    parser.add_argument("--no-voice", action="store_true", help="keyboard in, printed speech")
    parser.add_argument("--no-window", action="store_true", help="no OpenCV window")
    App(parser.parse_args()).run()


if __name__ == "__main__":
    main()
