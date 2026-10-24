"""python -m vision.yolo.model.export

Rebuilds the weights: YOLO-World (open-vocabulary, pretrained) with SUPPORTED
baked in as its class list, saved to weights_path(config). No training.
"""

import argparse
import hashlib
from pathlib import Path

from shared.config import load_config
from vision.yolo.model.classes import SUPPORTED
from vision.yolo.model.weights import weights_path

BASE_WEIGHTS = "yolov8s-worldv2.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE_WEIGHTS, help="YOLO-World base weights")
    args = parser.parse_args()

    from ultralytics import YOLO

    out = weights_path(load_config())
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.base)
    model.set_classes(SUPPORTED)
    model.save(str(out))

    names = list(YOLO(str(out)).names.values())
    assert names == SUPPORTED, f"saved classes {names} != SUPPORTED {SUPPORTED}"
    print(f"saved {out}")
    print(f"classes {names}")
    print(f"sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
