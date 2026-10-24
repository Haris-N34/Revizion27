"""HandTracker: MediaPipe hand landmarks -> HandPosition (index fingertip + wrist)."""

import ssl
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from shared.config import REPO_ROOT
from shared.contracts import HandPosition

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL = "models/hand_landmarker.task"  # gitignored, downloaded on first use
MIN_CONFIDENCE = 0.5
INDEX_TIP, WRIST = 8, 0


def model_path(config: dict) -> Path:
    path = Path(config.get("hands", {}).get("model", DEFAULT_MODEL))
    return path if path.is_absolute() else REPO_ROOT / path


def _download(path: Path) -> None:
    print(f"[hands] downloading {path.name} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(MODEL_URL, context=ctx) as r, path.open("wb") as f:
        f.write(r.read())


def _create_landmarker(path: Path):
    from mediapipe.tasks.python import BaseOptions, vision

    if not path.exists():
        _download(path)
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=MIN_CONFIDENCE,
        min_hand_presence_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


class HandTracker:
    def __init__(self, config: dict, landmarker=None):
        self.mirror = bool(config.get("mirror", False))
        self._landmarker = landmarker or _create_landmarker(model_path(config))
        self._frame_id = 0
        self._last_ts = -1
        # All 21 landmarks of the last tracked hand, in pixels (for drawing).
        self.landmarks: list[tuple[float, float]] | None = None

    def track(self, frame: np.ndarray) -> HandPosition | None:
        import mediapipe as mp

        frame_id = self._frame_id
        self._frame_id += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode needs strictly increasing timestamps.
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        height, width = frame.shape[:2]
        return self._to_position(result, width, height, frame_id)

    def _to_position(self, result, width: int, height: int, frame_id: int):
        self.landmarks = None
        if not result.hand_landmarks or not result.handedness:
            return None
        side = result.handedness[0][0]
        if side.score < MIN_CONFIDENCE:
            return None

        points = [(lm.x * width, lm.y * height) for lm in result.hand_landmarks[0]]
        self.landmarks = points
        handedness = side.category_name
        if self.mirror:
            handedness = {"Left": "Right", "Right": "Left"}.get(handedness, handedness)
        return HandPosition(
            x=points[INDEX_TIP][0],
            y=points[INDEX_TIP][1],
            wrist_x=points[WRIST][0],
            wrist_y=points[WRIST][1],
            handedness=handedness,
            frame_id=frame_id,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
