"""Where the agreed weights file lives locally (models/ is gitignored)."""

from pathlib import Path

from shared.config import REPO_ROOT


def weights_path(config: dict) -> Path:
    """config["yolo"]["weights"], resolved against the repo root if relative."""
    path = Path(config["yolo"]["weights"])
    return path if path.is_absolute() else REPO_ROOT / path
