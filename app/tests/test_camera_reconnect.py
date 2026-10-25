import threading
import time
from types import SimpleNamespace

from app import web
from app.camera import DroidCamCapture
from app.tests.test_camera import Connections, Phone
from shared.config import DEFAULTS
from shared.contracts import Direction


def wait_for(state, predicate):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if predicate(state.snapshot()):
            return
        time.sleep(0.01)
    assert predicate(state.snapshot()), state.snapshot()


def live_phone(connections):
    phone = Phone(100)
    connections.phones.put(phone)

    def feed():
        while not phone.closed:
            phone.feed(100)
            time.sleep(0.02)

    threading.Thread(target=feed, daemon=True).start()
    return phone


def test_web_survives_repeated_camera_loss_and_stop_cancels_resume(monkeypatch):
    connections = Connections()
    camera = DroidCamCapture("phone.test", connect=connections, retry_s=0.01)
    monkeypatch.setattr(web, "open_camera", lambda source: camera)
    monkeypatch.setattr("app.main.load_config", lambda: dict(DEFAULTS))
    app_factory, apps = web.WebApp, []

    def factory(args, build_parts=None):
        app = app_factory(args, build_parts=web.demo_parts)
        apps.append(app)
        return app

    monkeypatch.setattr(web, "WebApp", factory)
    args = SimpleNamespace(demo=False, no_voice=True, mock_band=True, no_window=True,
                           say_directions="off", video=None, target=None, rotation=0)
    state = web.WebState()
    worker = threading.Thread(target=web.run_camera, args=(state, args), daemon=True)
    worker.start()
    phones = []
    try:
        # The server can start with the phone absent and remain alive.
        wait_for(state, lambda data: data.get("phase") == "RECONNECTING")
        assert worker.is_alive() and state.jpeg is None
        phones.append(live_phone(connections))
        wait_for(state, lambda data: data.get("ready"))
        assert state.data["error"] is None
        state.actions.put(("target", "can"))
        wait_for(state, lambda data: data.get("target") == "can")

        phones[-1].close()
        wait_for(state, lambda data: data.get("phase") == "RECONNECTING")
        assert state.jpeg is None and state.last_frame == 0
        assert apps[0].guide.det is None
        assert apps[0].parts.band.sent[-1].direction == Direction.STOP
        phones.append(live_phone(connections))
        wait_for(state, lambda data: data.get("ready") and data.get("target") == "can")
        assert state.data["error"] is None

        phones[-1].close()
        wait_for(state, lambda data: data.get("phase") == "RECONNECTING")
        state.actions.put(("stop", None))
        wait_for(state, lambda data: data.get("target") is None)
        phones.append(live_phone(connections))
        wait_for(state, lambda data: data.get("ready"))
        assert state.data["target"] is None
        assert len(apps) == 1  # models and serial connection were never rebuilt
    finally:
        state.shutdown.set()
        worker.join(timeout=4)
        for phone in phones:
            phone.close()
        state.browser_frames.close()
    assert not worker.is_alive()
    assert not camera.worker.is_alive()
