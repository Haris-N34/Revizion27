import json
import threading
import time
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from app.web import WebApp, WebState, WebViewer, make_handler, run_camera


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
