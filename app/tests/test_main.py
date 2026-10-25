from app.main import camera_source


def test_camera_source_number_or_url():
    assert camera_source({"camera_index": 0}) == 0
    assert camera_source({"camera_index": "1"}) == 1
    assert camera_source({}) == 0
    url = "http://192.168.1.20:4747/video"
    assert camera_source({"camera_index": f" {url} "}) == url


def test_space_stops_old_guidance_before_listening():
    from types import SimpleNamespace
    from app.main import App

    events = []
    app = App.__new__(App)
    app.target, app.pending = "can", "can"
    app.guide = SimpleNamespace(cancel=lambda: events.append("stop"))
    app.listener = SimpleNamespace(state="idle", start=lambda: events.append("listen"))
    app.speaker = SimpleNamespace(
        say=lambda text: events.append(text), wait=lambda: events.append("wait")
    )
    app.ask()
    assert app.target is None and app.pending is None
    assert events == ["stop", "Listening.", "wait", "listen"]


def test_repeated_space_does_not_restart_busy_listener():
    from types import SimpleNamespace
    from app.main import App

    app = App.__new__(App)
    app.listener = SimpleNamespace(state="recording")
    app.ask()  # must return without cancelling or starting another recording


def test_recognized_request_is_acknowledged_before_detection():
    from types import SimpleNamespace
    from app.main import App

    said = []
    app = App.__new__(App)
    app.parts = SimpleNamespace(normalize=lambda word: "can" if word == "can" else None)
    app.speaker = SimpleNamespace(say=said.append)
    app.pending = None
    app.handle("find a can")
    assert said == ["Looking for a can."]
    assert app.pending == "can"
