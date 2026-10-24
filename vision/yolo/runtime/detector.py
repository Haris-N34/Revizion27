"""ObjectDetector: run YOLO on one frame and return Detections for one target."""

import math
from pathlib import Path

import numpy as np

from shared.config import REPO_ROOT
from shared.contracts import Detection

# Used until Rizwan's model/ package is merged, or if the weights file is missing.
# YOLO-World takes custom class names, so it can find "can" (not a COCO class).
FALLBACK_WEIGHTS = "yolov8s-worldv2.pt"
FALLBACK_CLASSES = ["can", "bottle", "cup", "cell phone", "remote"]
# Search text per label; bare "can" misses real cans, "drink can" finds them.
FALLBACK_PROMPTS = {"can": "drink can"}
# Real size per label for distance estimates (model/classes.py SIZES_M when present).
FALLBACK_SIZES_M = {
    "can": (0.122, "height"),
    "bottle": (0.23, "height"),
    "cup": (0.095, "height"),
    "cell phone": (0.15, "longest"),
    "remote": (0.18, "longest"),
    "mouse": (0.11, "longest"),
}
DEFAULT_HFOV_DEG = 70.0
IMGSZ = 640


def _weights_path(config: dict) -> Path:
    try:
        from vision.yolo.model.weights import weights_path
    except ImportError:
        return REPO_ROOT / config["yolo"]["weights"]
    return Path(weights_path(config))


def _supported() -> tuple[list[str], dict[str, str]]:
    """(labels, label -> search text) from model/, or the fallback lists."""
    try:
        from vision.yolo.model.classes import SUPPORTED
    except ImportError:
        return list(FALLBACK_CLASSES), dict(FALLBACK_PROMPTS)
    try:
        from vision.yolo.model.classes import PROMPTS
    except ImportError:
        PROMPTS = FALLBACK_PROMPTS
    return list(SUPPORTED), dict(PROMPTS)


def _sizes() -> dict[str, tuple[float, str]]:
    try:
        from vision.yolo.model.classes import SIZES_M
    except ImportError:
        return dict(FALLBACK_SIZES_M)
    return dict(SIZES_M)


def focal_px(frame_width: int, hfov_deg: float) -> float:
    """Pinhole focal length in pixels from the camera's horizontal field of view."""
    return (frame_width / 2) / math.tan(math.radians(hfov_deg) / 2)


def object_distance(
    label: str, w_px: float, h_px: float, frame_width: int, hfov_deg: float, sizes: dict
) -> float | None:
    """Camera-to-object distance in metres from the box size, or None if unknown."""
    if label not in sizes:
        return None
    real_m, side = sizes[label]
    px = h_px if side == "height" else max(w_px, h_px)
    if px <= 0:
        return None
    return float(focal_px(frame_width, hfov_deg) * real_m / px)


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _np(x) -> np.ndarray:
    """Tensor or array -> numpy array."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def load_model(config: dict):
    """Load the agreed weights, or YOLO-World with the supported classes as a fallback."""
    from ultralytics import YOLO

    path = _weights_path(config)
    if path.exists():
        print(f"[detector] loading {path.name}")
        model = YOLO(str(path))
    else:
        print(f"[detector] {path} not found, falling back to {FALLBACK_WEIGHTS}")
        model = YOLO(FALLBACK_WEIGHTS)
        labels, prompts = _supported()
        model.set_classes([prompts.get(label, label) for label in labels])
        model.model.names = dict(enumerate(labels))
    return model


class ObjectDetector:
    def __init__(self, config: dict, model=None):
        self.config = config
        self.confidence = float(config["yolo"]["confidence"])
        self.confirm_frames = int(config["yolo"]["confirm_frames"])
        self.model = model if model is not None else load_model(config)
        self.device = None if model is not None else _device()
        self._frame_id = 0
        self.hfov_deg = float(config.get("camera_hfov_deg", DEFAULT_HFOV_DEG))
        self.sizes = _sizes()

    def detect(self, frame: np.ndarray, target: str) -> list[Detection]:
        """Boxes labelled `target` with confidence >= threshold, best first."""
        frame_id = self._frame_id
        self._frame_id += 1
        target = target.strip().lower()

        kwargs = {"conf": self.confidence, "imgsz": IMGSZ, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(frame, **kwargs)[0]

        names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy, confs, classes = _np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls)

        found = []
        frame_width = frame.shape[1]
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
            label = str(names[int(cls)]).lower()
            if label != target or conf < self.confidence:
                continue
            found.append(
                Detection(
                    label=label,
                    confidence=float(conf),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    frame_id=frame_id,
                    distance_m=object_distance(
                        label, x2 - x1, y2 - y1, frame_width, self.hfov_deg, self.sizes
                    ),
                )
            )
        found.sort(key=lambda d: d.confidence, reverse=True)
        return found
