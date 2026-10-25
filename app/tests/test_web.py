import json
import threading
import time
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from app.web import WebApp, WebState, WebViewer, make_handler, rotate_frame, run_camera


@pytest.fixture
def web_server():
    state = WebState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield state, f"http://127.0.0.1:{server.server_port}"
    state.shutdown.set()
    with state.condition:
        state.condition.notify_all()
    server.shutdown()
    server.server_close()
    state.browser_frames.close()
    thread.join(timeout=2)


def post(url, path, payload=None, **headers):
    request = Request(url + path, json.dumps(payload or {}).encode(),
                      {"Content-Type": "application/json", **headers}, method="POST")
    return urlopen(request, timeout=2)


def test_page_and_status_serve_without_hardware(web_server):
    state, url = web_server
    with urlopen(url, timeout=2) as response:
        assert b"Live view" in response.read()
    with urlopen(url + "/api/status", timeout=2) as response:
        data = json.load(response)
        assert not data["ready"] and data["error"] is None
    assert state.jpeg is None


def test_controls_wait_for_camera_and_validate_target(web_server):
    state, url = web_server
    with pytest.raises(HTTPError) as error:
        post(url, "/api/target", {"target": "can"})
    assert error.value.code == 409
    state.update(ready=True, supported=["can"], voice=True)
    with pytest.raises(HTTPError) as error:
        post(url, "/api/target", {"target": "unknown"})
    assert error.value.code == 400
    for path, payload, expected in [
        ("/api/target", {"target": "can"}, ("target", "can")),
        ("/api/listen", {}, ("listen", None)),
        ("/api/stop", {}, ("stop", None)),
    ]:
        with post(url, path, payload) as response:
            assert response.status == 202
        assert state.actions.get_nowait() == expected


def test_camera_backup_can_be_selected_while_disconnected(web_server):
    state, url = web_server
    for camera in ("webcam", "usb", "configured"):
        with post(url, "/api/camera", {"camera": camera}) as response:
            assert response.status == 202
        assert state.actions.get_nowait() == ("camera", camera)
    with pytest.raises(HTTPError) as error:
        post(url, "/api/camera", {"camera": "invalid"})
    assert error.value.code == 400


def test_cross_origin_commands_and_path_traversal_are_rejected(web_server):
    state, url = web_server
    state.update(ready=True, supported=["can"], voice=True)
    with pytest.raises(HTTPError) as error:
        post(url, "/api/target", {"target": "can"}, Origin="https://example.com")
    assert error.value.code == 403
    assert state.actions.empty()
    with pytest.raises(HTTPError) as error:
        urlopen(url + "/../config.yaml", timeout=2)
    assert error.value.code == 404


def test_https_tunnel_accepts_same_origin_controls(web_server):
    state, url = web_server
    state.update(ready=True, supported=["can"], voice=True)
    host = "revision.example.com"
    with post(url, "/api/stop", Host=host, Origin="https://" + host) as response:
        assert response.status == 202
    assert state.actions.get_nowait() == ("stop", None)


def test_browser_frames_are_independent_of_laptop_camera_readiness(web_server):
    state, url = web_server
    received = []
    state.browser_frames.submit = lambda payload: received.append(payload) or {"detections": []}
    with post(url, "/api/analyze", {"session": "browser-session-1234", "image": "x" * 2000}) as response:
        assert json.load(response) == {"detections": []}
    assert len(received[0]["image"]) == 2000
    assert not state.snapshot()["ready"] and state.actions.empty()


def test_stream_contains_encoded_camera_frame(web_server):
    state, url = web_server
    WebViewer(state).show_frame(np.zeros((100, 160, 3), dtype=np.uint8))
    with urlopen(url + "/stream", timeout=2) as response:
        assert response.headers["Content-Type"].startswith("multipart/")
        assert b"Content-Type: image/jpeg" in response.read(100)
    assert state.jpeg.startswith(b"\xff\xd8")


def test_stop_discards_transcript_that_finishes_later():
    app = WebApp.__new__(WebApp)
    app._ignore_transcript = True
    app.pending = None
    app.handle("find a can")
    assert app.pending is None


def test_stop_is_allowed_while_camera_reconnects(web_server):
    state, url = web_server
    state.update(ready=False, phase="RECONNECTING")
    with post(url, "/api/stop") as response:
        assert response.status == 202
    assert state.actions.get_nowait() == ("stop", None)


def test_camera_rotation_applies_to_frame_before_guidance():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = 255
    rotated = rotate_frame(frame, 90)
    assert rotated.shape == (3, 2, 3)
    assert (rotated[0, 1] == 255).all()
    assert rotate_frame(frame, 0) is frame


def test_rotation_api_validates_and_queues_camera_change(web_server):
    state, url = web_server
    state.update(ready=True, listening="idle")
    with post(url, "/api/rotate", {"rotation": 90}) as response:
        assert response.status == 202
    assert state.actions.get_nowait() == ("rotate", 90)
    for invalid in (45, "90", True):
        with pytest.raises(HTTPError) as error:
            post(url, "/api/rotate", {"rotation": invalid})
        assert error.value.code == 400
    state.update(listening="recording")
    with pytest.raises(HTTPError) as error:
        post(url, "/api/rotate", {"rotation": 180})
    assert error.value.code == 409


def test_demo_runs_and_controls_guidance_without_hardware(monkeypatch):
    from shared.config import DEFAULTS

    monkeypatch.setattr("app.main.load_config", lambda: dict(DEFAULTS))
    args = SimpleNamespace(demo=True, no_voice=True, mock_band=True, no_window=True,
                           say_directions="off", video=None, target=None)
    state = WebState()
    worker = threading.Thread(target=run_camera, args=(state, args), daemon=True)
    worker.start()

    def wait_until(predicate):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if predicate(state.snapshot()):
                return
            time.sleep(0.02)
        pytest.fail(f"Worker did not reach expected state: {state.snapshot()}")

    try:
        wait_until(lambda data: data.get("ready") and state.jpeg is not None)
        state.actions.put(("target", "can"))
        wait_until(lambda data: data.get("target") == "can")
        state.actions.put(("stop", None))
        wait_until(lambda data: data.get("target") is None and data.get("phase") == "IDLE")
        assert state.data["error"] is None
    finally:
        state.shutdown.set()
        worker.join(timeout=4)
    assert not worker.is_alive()
