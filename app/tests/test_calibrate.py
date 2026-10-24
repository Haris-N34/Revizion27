import pytest

from app.calibrate import REFERENCES, distance_m, hfov_from_sample, save_hfov


def test_hfov_round_trip():
    """The field of view found from a sample puts the can back at the measured distance."""
    hfov = hfov_from_sample(0.50, 200, 1280)
    assert distance_m(200, 1280, hfov) == pytest.approx(0.50)


def test_bigger_can_in_image_means_narrower_fov():
    # Same distance, can looks bigger -> more zoomed-in camera -> smaller field of view.
    assert hfov_from_sample(0.5, 300, 1280) < hfov_from_sample(0.5, 200, 1280)


def test_typical_webcam_values_are_sane():
    # ~70 deg webcam at 1280 px: f = 914 px; can at 50 cm front (53.3 cm) -> ~209 px tall.
    assert 65 < hfov_from_sample(0.50, 209, 1280) < 75


def test_save_keeps_other_lines(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("# my camera\ncamera_index: 1\ncamera_hfov_deg: 70\nmirror: false\n")
    save_hfov(63.25, cfg)
    assert cfg.read_text() == "# my camera\ncamera_index: 1\ncamera_hfov_deg: 63.2\nmirror: false\n"


def test_save_appends_when_missing(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("camera_index: 1\n")
    save_hfov(80, cfg)
    assert cfg.read_text().endswith("camera_hfov_deg: 80.0\n")
    assert "camera_index: 1" in cfg.read_text()


def test_phone_reference_round_trip():
    phone = REFERENCES["phone"]
    assert phone.label == "cell phone" and phone.height_m == pytest.approx(0.1634)
    hfov = hfov_from_sample(0.60, 240, 1920, phone)
    assert distance_m(240, 1920, hfov, phone) == pytest.approx(0.60)


def test_taller_reference_same_pixels_means_wider_fov():
    # Same px height and distance: a taller real object implies a wider field of view.
    can, phone = REFERENCES["can"], REFERENCES["phone"]
    assert hfov_from_sample(0.5, 200, 1280, phone) > hfov_from_sample(0.5, 200, 1280, can)
