"""Load config.yaml from the repo root, filling in defaults for missing keys."""

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Every key gets a default so older config.yaml files keep working.
DEFAULTS: dict = {
    "camera_index": 0,
    "mirror": True,
    "serial_port": None,
    "baud": 115200,
    "wrist": "right",
    "yolo": {
        "weights": "models/octaby.pt",
        "confidence": 0.5,
        "confirm_frames": 3,
    },
    "scan_timeout_s": 5.0,
    "guidance": {
        "arrive_px": 40,
        "dead_zone_px": 25,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None) -> dict:
    """Return config as a dict. Missing file or keys fall back to DEFAULTS."""
    path = Path(path) if path else CONFIG_PATH
    if not path.exists():
        print(f"[config] {path.name} not found, using defaults")
        return copy.deepcopy(DEFAULTS)
    with path.open(encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    return _merge(DEFAULTS, user)
