from app.main import camera_source


def test_camera_source_number_or_url():
    assert camera_source({"camera_index": 0}) == 0
    assert camera_source({"camera_index": "1"}) == 1
    assert camera_source({}) == 0
    url = "http://192.168.1.20:4747/video"
    assert camera_source({"camera_index": f" {url} "}) == url
