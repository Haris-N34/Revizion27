import pytest

from shared.config import DEFAULTS, REPO_ROOT
from vision.yolo.model.classes import ALIASES, PROMPTS, SUPPORTED, normalize
from vision.yolo.model.weights import weights_path


@pytest.mark.parametrize(
    "spoken, label",
    [
        ("can", "can"),
        ("Can", "can"),
        ("a can", "can"),
        ("take me to a can", "can"),
        ("coke", "can"),
        ("soda", "can"),
        ("pop can", "can"),
        ("can of coke", "can"),
        ("Coke cans!", "can"),
        ("water bottle", "bottle"),
        ("my water bottle", "bottle"),
        ("bottles", "bottle"),
        ("mug", "cup"),
        ("coffee cup", "cup"),
        ("cups", "cup"),
        ("phone", "cell phone"),
        ("my phone", "cell phone"),
        ("cell phone", "cell phone"),
        ("TV remote", "remote"),
    ],
)
def test_normalize(spoken, label):
    assert normalize(spoken) == label


@pytest.mark.parametrize("spoken", ["", "elephant", "banana", "the", "car keys"])
def test_unsupported_is_none(spoken):
    assert normalize(spoken) is None


def test_supported_is_lowercase_and_unique():
    assert SUPPORTED == [s.lower().strip() for s in SUPPORTED]
    assert len(set(SUPPORTED)) == len(SUPPORTED)


def test_aliases_point_at_supported_labels():
    assert set(ALIASES.values()) <= set(SUPPORTED)


def test_prompts_are_for_supported_labels():
    assert set(PROMPTS) <= set(SUPPORTED)


def test_weights_path_relative_to_repo_root():
    assert weights_path(DEFAULTS) == REPO_ROOT / "models" / "octaby.pt"


def test_weights_path_absolute(tmp_path):
    cfg = {"yolo": {"weights": str(tmp_path / "w.pt")}}
    assert weights_path(cfg) == tmp_path / "w.pt"


def test_weights_produce_supported_labels():
    """Skipped unless the weights file has been downloaded or exported."""
    path = weights_path(DEFAULTS)
    if not path.exists():
        pytest.skip(f"{path} not present")
    yolo = pytest.importorskip("ultralytics").YOLO
    assert list(yolo(str(path)).names.values()) == SUPPORTED
