from pathlib import Path

from shared.config import DEFAULTS, load_config
from shared.contracts import Detection, Direction

EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.yaml"


def test_detection_center():
    det = Detection("can", 0.9, 10, 20, 30, 60, frame_id=1)
    assert det.center == (20, 40)


def test_direction_values_match_serial_protocol():
    assert Direction.ARRIVED.value == "G"
    assert f"{Direction.LEFT.value},180" == "L,180"


def test_example_config_has_every_default_key():
    cfg = load_config(EXAMPLE)
    assert set(cfg) == set(DEFAULTS)
    assert set(cfg["yolo"]) == set(DEFAULTS["yolo"])
    assert set(cfg["guidance"]) == set(DEFAULTS["guidance"])


def test_missing_config_uses_defaults(tmp_path):
    assert load_config(tmp_path / "nope.yaml") == DEFAULTS


def test_depth_fields_are_optional():
    """Old call sites (no distance) keep working; distance is None by default."""
    from shared.contracts import HandPosition

    assert Detection("can", 0.9, 0, 0, 1, 1, 0).distance_m is None
    assert HandPosition(1, 2, 3, 4, "Right", 0).distance_m is None
    assert Detection("can", 0.9, 0, 0, 1, 1, 0, distance_m=0.5).distance_m == 0.5


def test_forward_back_serial_codes():
    assert (Direction.FORWARD.value, Direction.BACK.value) == ("F", "B")
    assert len({d.value for d in Direction}) == len(Direction)
