"""Local browser UI: python -m app.web [--mock-band] [--no-voice] [--demo]."""

import argparse
from collections import deque
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from app import voice_out
from app.fakes import FakeDetector, FakeHandTracker, PrintBand, fake_compute
from app.loop import State
from app.main import App, Viewer, camera_source

STATIC = Path(__file__).with_name("web_static")


class WebState:
    def __init__(self):
        self.condition = threading.Condition()
        self.actions = queue.Queue(maxsize=16)
        self.jpeg = None
        self.sequence = 0
        self.last_frame = 0.0
        self.last_client = time.monotonic()
        self.events = deque(maxlen=6)
        self.data = {"ready": False, "status": "Starting camera and guidance…", "error": None}
        self.shutdown = threading.Event()

    def update(self, **values):
        with self.condition:
            self.data.update(values)

    def snapshot(self):
        with self.condition:
            self.last_client = time.monotonic()
            return dict(self.data, events=list(self.events),
                        frame_age_s=round(time.monotonic() - self.last_frame, 1) if self.last_frame else None)

    def announce(self, text):
        with self.condition:
            self.events.appendleft(text)


class WebViewer(Viewer):
    def __init__(self, state):
        super().__init__(show=False)
        self.state = state

    def show_frame(self, frame):
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1 / dt
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.state.condition:
                self.state.jpeg = encoded.tobytes()
                self.state.sequence += 1
                self.state.last_frame = time.monotonic()
                self.state.condition.notify_all()
        return -1


def demo_parts(config, mock_band):
    """Standalone UI demo: no camera, models, microphone or Arduino needed."""
    detector = FakeDetector()
    tracker = FakeHandTracker(speed=12)
    return SimpleNamespace(detector=detector, tracker=tracker, compute=fake_compute,
                           turn=None, stabilizer=None, band=PrintBand(), real_band=False,
                           normalize=lambda word: "can" if word.strip().lower() == "can" else None,
                           supported=["can"], fakes=["demo camera and guidance"])


def stop_guidance(app):
    app.guide.cancel()
    app.target, app.pending = None, None
    app.guide.target, app.guide.det = None, None


class WebApp(App):
    def handle(self, text):
        if not getattr(self, "_ignore_transcript", False):
            super().handle(text)


def run_camera(state, args):
    app, cap = None, None
    try:
        # App creates all camera/serial/model resources on this worker thread.
        app = WebApp(args, build_parts=demo_parts if args.demo else None)
        original_say = app.speaker.say

        def say(text):
            state.announce(text)
            original_say(text)

        app.speaker.say = say
        app.guide.say = say
        app.viewer = WebViewer(state)
        if not args.demo:
            cap = cv2.VideoCapture(args.video or camera_source(app.config))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError("Cannot open the camera. Reconnect the phone camera, then restart the app.")
        state.update(ready=True, supported=app.parts.supported,
                     voice=app.listener is not None, arduino=app.parts.real_band, demo=args.demo)
        if args.target:
            app.handle(args.target)
        while not state.shutdown.is_set():
            # Only this thread ever touches the Arduino or guidance state.
            for _ in range(16):
                try:
                    action, value = state.actions.get_nowait()
                except queue.Empty:
                    break
                if action == "stop":
                    stop_guidance(app)
                    # Ignore an in-flight transcript after the user presses Stop.
                    app._ignore_transcript = True
                elif action == "listen":
                    if app.listener is not None:
                        app._ignore_transcript = False
                        app.ask()
                elif action == "target":
                    if app.listener is None or app.listener.state == "idle":
                        app._ignore_transcript = False
                        stop_guidance(app)
                        app.handle(value)
            if time.monotonic() - state.last_client > 3 and app.target:
                stop_guidance(app)
                state.announce("Guidance paused: browser disconnected.")
            if app.pending:
                app.target, app.pending = app.pending, None
                app.guide.start(app.target, announce=False)
            if args.demo:
                frame = np.full((720, 1280, 3), (35, 42, 38), np.uint8)
                cv2.putText(frame, "DEMO CAMERA", (440, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (190, 210, 190), 2)
                cv2.rectangle(frame, (768, 360), (896, 576), (100, 140, 200), -1)
                if app.guide.det:
                    app.parts.tracker.goal = app.guide.det.center
                state.shutdown.wait(1 / 20)
            else:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera disconnected. Reconnect it and restart the app.")
            # Consume and discard a cancelled recording before App.frame can use it.
            if getattr(app, "_ignore_transcript", False) and app.listener:
                app.listener.result()
            app.frame(frame)
            g = app.guide
            state.update(status=app.status(), phase=g.state.value, target=app.target,
                         hand=g.hand is not None, detected=g.det is not None,
                         direction=g.cmd.direction.name.replace("_", " ") if g.cmd and g.state == State.GUIDING else None,
                         fps=round(app.viewer.fps), listening=app.listener.state if app.listener else "disabled")
    except Exception as exc:
        state.update(ready=False, error=str(exc), status="Connection needs attention")
        print(f"[web] {exc}", flush=True)
    finally:
        if app:
            try:
                app.guide.stop()
            finally:
                if hasattr(app.parts.tracker, "close"):
                    app.parts.tracker.close()
                app.speaker.close()
        if cap is not None:
            cap.release()
        state.update(ready=False)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/status":
                self.send(200, json.dumps(state.snapshot()).encode())
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                sequence = -1
                try:
                    while not state.shutdown.is_set():
                        with state.condition:
                            state.condition.wait_for(
                                lambda: (state.jpeg is not None and state.sequence != sequence)
                                or state.shutdown.is_set(), timeout=1)
                            if state.jpeg is None or sequence == state.sequence:
                                continue
                            sequence, jpeg = state.sequence, state.jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            else:
                files = {"/": ("index.html", "text/html; charset=utf-8"),
                         "/style.css": ("style.css", "text/css"),
                         "/app.js": ("app.js", "text/javascript")}
                if self.path not in files:
                    self.send(404, b'{"error":"Not found"}')
                    return
                filename, mime = files[self.path]
                self.send(200, (STATIC / filename).read_bytes(), mime)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.send(403, b'{"error":"Use this app from its local browser page"}')
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send(415, b'{"error":"Expected JSON"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError()
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError()
            except (ValueError, json.JSONDecodeError):
                self.send(400, b'{"error":"Invalid request"}')
                return
            actions = {"/api/target": "target", "/api/listen": "listen", "/api/stop": "stop"}
            action = actions.get(self.path)
            if action is None:
                self.send(404, b'{"error":"Not found"}')
                return
            data = state.snapshot()
            if not data.get("ready"):
                self.send(409, b'{"error":"Camera is not ready yet"}')
                return
            if action == "listen" and not data.get("voice"):
                self.send(409, b'{"error":"Voice is disabled; use the target buttons"}')
                return
            if action == "target" and payload.get("target") not in data.get("supported", []):
                self.send(400, b'{"error":"Choose a supported target"}')
                return
            try:
                state.actions.put_nowait((action, payload.get("target")))
            except queue.Full:
                self.send(429, b'{"error":"Please wait for the previous command"}')
                return
            self.send(202, b'{"ok":true}')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mock-band", action="store_true")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--demo", action="store_true", help="synthetic camera and mock hardware")
    parser.add_argument("--video")
    parser.add_argument("--target")
    args = parser.parse_args()
    args.no_window, args.say_directions = True, "off"
    if args.demo:
        args.no_voice, args.mock_band = True, True
    state = WebState()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    worker = threading.Thread(target=run_camera, args=(state, args), daemon=True)
    worker.start()
    print(f"Octaby web UI: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        state.shutdown.set()
        with state.condition:
            state.condition.notify_all()
        server.server_close()
        worker.join(timeout=8)


if __name__ == "__main__":
    main()
