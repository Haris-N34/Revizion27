"""Analyze browser-supplied frames without touching the laptop camera or band."""

import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import re
import threading
import time

import cv2
import numpy as np

from shared.config import load_config

INFERENCE_LOCK = threading.Lock()


def decode_request(payload):
    session = payload.get("session", "")
    if not isinstance(session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,80}", session):
        raise ValueError("Invalid camera session")
    target = payload.get("target")
    if not isinstance(target, str) or len(target) > 80:
        raise ValueError("Choose a supported target")
    image = payload.get("image")
    if not isinstance(image, str) or len(image) > 700_000:
        raise ValueError("Camera frame is too large")
    try:
        raw = base64.b64decode(image, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid camera image") from exc
    if not raw.startswith(b"\xff\xd8"):
        raise ValueError("Send a JPEG camera frame")
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or max(frame.shape[:2]) > 960:
        raise ValueError("Camera frame must be at most 960 pixels per side")
    if type(payload.get("mirror", False)) is not bool:
        raise ValueError("Invalid mirror setting")
    return session, target, frame, payload.get("mirror", False)


class BrowserAnalyzer:
    """One detector; separate hand tracking for each of at most four active tabs."""

    def __init__(self):
        from guidance.compute import compute
        from vision.hands.tracker import HandTracker
        from vision.yolo.model.classes import normalize
        from vision.yolo.runtime.detector import ObjectDetector

        self.config = load_config()
        self.detector = ObjectDetector(self.config)
        self.tracker_factory, self.compute, self.normalize = HandTracker, compute, normalize
        self.sessions = {}

    def analyze(self, session, target, frame, mirror):
        target = self.normalize(target)
        if target is None:
            raise ValueError("Choose a supported target")
        now = time.monotonic()
        for key, (_, used, _) in list(self.sessions.items()):
            if now - used > 120:
                self.sessions.pop(key)[0].close()
        config = dict(self.config, mirror=mirror)
        if session in self.sessions and self.sessions[session][2] != mirror:
            self.sessions.pop(session)[0].close()
        if session not in self.sessions:
            if len(self.sessions) >= 4:
                oldest = min(self.sessions, key=lambda key: self.sessions[key][1])
                self.sessions.pop(oldest)[0].close()
            self.sessions[session] = (self.tracker_factory(config), now, mirror)
        tracker = self.sessions[session][0]
        self.sessions[session] = (tracker, now, mirror)
        dets = self.detector.detect(frame, target)
        hand = tracker.track(frame)
        command = self.compute(dets[0], hand, frame.shape, config) if dets else None
        return {"detections": [asdict(det) for det in dets],
                "hand": asdict(hand) if hand else None,
                "command": {"direction": command.direction.name.replace("_", " "),
                            "intensity": command.intensity} if command else None,
                "width": frame.shape[1], "height": frame.shape[0], "target": target}

    def close(self):
        for tracker, _, _ in self.sessions.values():
            tracker.close()
        self.sessions.clear()


class BrowserFrames:
    """Keep inference on one thread and reject excess frames instead of queuing."""

    def __init__(self, factory=BrowserAnalyzer):
        self.factory, self.analyzer = factory, None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-camera")
        self.slot = threading.BoundedSemaphore(1)

    def submit(self, payload):
        args = decode_request(payload)
        if not self.slot.acquire(blocking=False):
            raise RuntimeError("Detection is busy; retry in a moment")
        future = self.executor.submit(self._analyze, args)
        future.add_done_callback(lambda _: self.slot.release())
        return future.result(timeout=45)

    def _analyze(self, args):
        # Serialize model loading/inference with the laptop loop. Some native
        # Torch backends crash when separate models warm up concurrently.
        with INFERENCE_LOCK:
            if self.analyzer is None:
                self.analyzer = self.factory()
            return self.analyzer.analyze(*args)

    def close(self):
        def release():
            if self.analyzer:
                self.analyzer.close()
        self.executor.submit(release)
        self.executor.shutdown(wait=False)
