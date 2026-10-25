"""python -m app.main [--target can] [--mock-band] [--video FILE] [--no-voice] [--no-window]
                   [--say-directions auto/on/off]

Live view: target box, hand skeleton, hand-to-target line and band command.
Arduino button: ask out loud ("where is the can?").  q / Esc: quit.
Detection and guidance continue automatically while the target is visible.
Without --target press the wristband button to talk. --no-voice types instead of talking.
"""

import argparse
import faulthandler
import sys
import time
from types import SimpleNamespace

import cv2

from app import voice_out
from app.debug_view import GREEN, RED, render
from app.fakes import FakeHandTracker, PrintBand, fake_compute
from app.loop import Guide, State
from app.narrator import Narrator
from app.voice_in import Listener, extract_target
from shared.config import load_config
from vision.yolo.runtime.detector import ObjectDetector

WINDOW = "REVISION"
STALL_S = 3.0  # a frame slower than this dumps every thread's stack to the terminal
QUIT_KEYS = (ord("q"), 27)


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
        from guidance.compute import compute, turn_toward
    except ImportError:
        compute, turn_toward = fake_compute, None
        fakes.append("fake_compute")
    try:
        from guidance.stabilize import Stabilizer

        stabilizer = Stabilizer(config)
    except ImportError:
        stabilizer = None
        fakes.append("no stabilizer")
    try:
        from hardware.host.band import Band, MockBand

        real_band = not mock_band and bool(config.get("serial_port"))
        band = Band(config) if real_band else MockBand(config)
    except ImportError:
        band, real_band = PrintBand(config), False
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
        turn=turn_toward,
        stabilizer=stabilizer,
        band=band,
        real_band=real_band,
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
        # Stall detector: re-armed every frame; only fires if the loop stops for STALL_S.
        faulthandler.dump_traceback_later(STALL_S, repeat=True, file=sys.stderr)
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


def camera_source(config: dict) -> int | str:
    """camera_index as a device number (webcam, iPhone via Continuity Camera) or a
    stream URL (e.g. an iPhone webcam app's http:// / rtsp:// address)."""
    value = config.get("camera_index", 0)
    if isinstance(value, str) and not value.strip().isdigit():
        return value.strip()
    return int(value)


def read_frames(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


class App:
    def __init__(self, args, build_parts=None):
        self.args = args
        self.config = load_config()
        self.parts = (build_parts or build)(self.config, args.mock_band)
        if self.parts.fakes:
            print(f"[app] fakes active: {', '.join(self.parts.fakes)}")
        self.speaker = voice_out.configure(enabled=not args.no_voice)
        self.listener = None
        if not args.no_voice:
            self.listener = Listener(hint_words=self.parts.supported)
            self.listener.preload()
        p = self.parts
        self.guide = Guide(
            self.config,
            p.detector,
            p.tracker,
            p.compute,
            p.band,
            self.speaker.say,
            stabilizer=p.stabilizer,
            turn=p.turn,
        )
        self.viewer = Viewer(show=not args.no_window)
        # Speak search results and arrival; directions are opt-in only.
        mode = getattr(args, "say_directions", "off")
        narrate = mode == "on"
        self.narrator = Narrator(self.speaker) if narrate else None
        self.target: str | None = None
        self.pending: str | None = None  # target to scan for on the next frame

    # --- input -------------------------------------------------------------
    def poll_button(self) -> None:
        poll = getattr(self.parts.band, "poll_button", None)
        if poll is not None and poll() and self.listener is not None:
            self.ask()

    def ask(self) -> None:
        """Start listening (voice) or read the keyboard (--no-voice)."""
        if self.listener is not None and self.listener.state != "idle":
            return
        self.guide.cancel()
        self.target, self.pending = None, None
        if self.listener is None:
            self.handle(input("What should I find? "))
        elif self.listener.state == "idle":
            self.speaker.say(voice_out.LISTENING)
            self.speaker.wait()  # don't record our own voice
            self.listener.start()

    def handle(self, text: str) -> None:
        target = extract_target(text, self.parts.normalize)
        if target:
            self.speaker.say(voice_out.LOOKING.format(target))
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
        visibility = getattr(self.guide, "visibility", "visible")
        if self.target and visibility == "occluded":
            return f"hand may cover {self.target}; remembering its position"
        if self.target and visibility == "missing":
            return f"look around for {self.target}"
        return {
            State.GUIDING: (
                f"turn {self.guide.cmd.direction.name.lower()} to find {self.target}"
                if self.guide.turning and self.guide.cmd
                else f"guiding to {self.target}"
            ),
            State.DONE: "Object found",
            State.NOT_FOUND: f"NOT FOUND: {self.target}",
            State.SCANNING: f"looking for {self.target}",
            State.IDLE: "press the wristband button and ask",
        }.get(self.guide.state, self.guide.state.value)

    def run(self) -> None:
        a = self.args
        if a.target:
            self.handle(a.target)
        elif a.no_voice:
            self.ask()
        source = a.video if a.video else camera_source(self.config)
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the newest frame, never a backlog
        if not cap.isOpened():
            raise SystemExit(f"Could not open {source!r}")
        frames = read_frames(cap)
        try:
            for frame in frames:
                if self.pending:
                    self.target, self.pending = self.pending, None
                    self.guide.start(self.target, announce=False)
                self.frame(frame)
        except (Quit, KeyboardInterrupt):
            pass
        finally:
            self.guide.stop()
            if hasattr(self.parts.tracker, "close"):
                self.parts.tracker.close()
            cap.release()
            cv2.destroyAllWindows()
            faulthandler.cancel_dump_traceback_later()
            self.speaker.close()

    def scan(self, frames) -> None:
        self.target, self.pending = self.pending, None
        view = ScanView(self.parts.detector, self.parts.tracker, self.viewer, self.target)
        found = self.guide.find(frames, self.target, detector=view)
        if found and isinstance(self.parts.tracker, FakeHandTracker):
            self.parts.tracker.goal = found.center

    def frame(self, frame) -> None:
        self.poll_button()
        g = self.guide
        if self.target:
            g.follow(frame)
            if self.narrator and g.state == State.GUIDING:
                self.narrator.update(g.cmd, turning=g.turning)
            elif self.narrator:
                self.narrator.reset()
        else:
            if self.narrator:
                self.narrator.reset()
            # Not guiding: keep showing the hand (and target, if any) live; no band output.
            if self.target:
                found = g.detector.detect(frame, self.target)
                g.det = found[0] if found else g.det
            g.hand = g.tracker.track(frame)
        landmarks = getattr(g.tracker, "landmarks", None)
        cmd = g.cmd if g.state == State.GUIDING else None
        hint = "continuous detection   wristband button: talk   q: quit"
        image = render(
            frame, self.status(), g.det, g.hand, landmarks, cmd, self.viewer.fps, hint,
            status_color=GREEN if g.state == State.DONE else RED,
        )
        self.viewer.show_frame(image)
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
    parser.add_argument(
        "--say-directions",
        choices=("auto", "on", "off"),
        default="off",
        help="speak directions only when explicitly set to on (default: off)",
    )
    App(parser.parse_args()).run()


if __name__ == "__main__":
    main()
