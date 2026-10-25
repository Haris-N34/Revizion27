import base64
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from app.browser_camera import BrowserAnalyzer, BrowserFrames, INFERENCE_LOCK, decode_request
from shared.contracts import Detection, Direction, GuidanceCommand, HandPosition


def payload(session="camera-session-1234"):
    _, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
    return {"session": session, "target": "can",
            "image": base64.b64encode(encoded).decode(), "mirror": False}


def test_browser_frame_decode_validates_sessions_and_images():
    session, target, frame, mirror = decode_request(payload())
    assert session == "camera-session-1234" and target == "can"
    assert frame.shape == (48, 64, 3) and not mirror
    for changes in ({"session": "../bad"}, {"target": None}, {"image": "broken"},
                    {"image": base64.b64encode(b"not jpeg").decode()}, {"mirror": "false"}):
        with pytest.raises(ValueError):
            decode_request(dict(payload(), **changes))


def test_browser_users_have_separate_hand_tracking_and_mirror_settings():
    analyzer = BrowserAnalyzer.__new__(BrowserAnalyzer)
    analyzer.config, analyzer.sessions = {}, {}
    analyzer.normalize = lambda target: "can" if target == "can" else None
    det = Detection("can", .9, 20, 20, 40, 40, 1)
    analyzer.detector = SimpleNamespace(detect=lambda frame, target: [det])
    trackers = []
    def tracker_factory(config):
        tracker = SimpleNamespace(track=Mock(return_value=HandPosition(10, 10, 5, 5, "Right", 1)),
                                  close=Mock(), mirror=config["mirror"])
        trackers.append(tracker)
        return tracker
    analyzer.tracker_factory = tracker_factory
    analyzer.compute = Mock(return_value=GuidanceCommand(Direction.RIGHT, 100, 20))
    frame = np.zeros((48, 64, 3), np.uint8)
    first = analyzer.analyze("user-a", "can", frame, True)
    analyzer.analyze("user-b", "can", frame, False)
    analyzer.analyze("user-a", "can", frame, True)
    assert len(trackers) == 2
    assert trackers[0].track.call_count == 2 and trackers[1].track.call_count == 1
    assert trackers[0].mirror and not trackers[1].mirror
    assert first["command"]["direction"] == "RIGHT"
    assert first["width"] == 64 and first["hand"]["x"] == 10
    with pytest.raises(ValueError):
        analyzer.analyze("user-a", "unsupported", frame, False)
    analyzer.analyze("user-a", "can", frame, False)
    assert trackers[0].close.call_count == 1
    assert len(trackers) == 3 and not trackers[2].mirror
    analyzer.close()
    assert all(tracker.close.call_count == 1 for tracker in trackers)


def test_browser_service_does_not_queue_unlimited_frames():
    analyzer = SimpleNamespace(analyze=Mock(return_value={"detections": []}), close=Mock())
    service = BrowserFrames(factory=lambda: analyzer)
    try:
        service.slot.acquire()
        with pytest.raises(RuntimeError, match="busy"):
            service.submit(payload())
        service.slot.release()
        assert service.submit(payload()) == {"detections": []}
        analyzer.analyze.assert_called_once()
    finally:
        service.close()


def test_browser_model_initialization_waits_for_laptop_inference():
    import threading
    entered = threading.Event()
    def factory():
        entered.set()
        return SimpleNamespace(analyze=lambda *args: {}, close=lambda: None)
    service = BrowserFrames(factory=factory)
    try:
        with INFERENCE_LOCK:
            job = service.executor.submit(service._analyze, decode_request(payload()))
            assert not entered.wait(.05)
        assert job.result(timeout=2) == {}
        assert entered.is_set()
    finally:
        service.close()
