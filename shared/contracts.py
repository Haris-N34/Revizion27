"""Types shared between modules. Change only via a feat/shared PR to main."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Detection:
    label: str  # normalized name, e.g. "can"
    confidence: float  # 0.0 to 1.0
    x1: float  # bounding box in pixels, origin top-left
    y1: float
    x2: float
    y2: float
    frame_id: int
    distance_m: float | None = None  # camera -> object, metres; None if unknown

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class HandPosition:
    x: float  # grab point, pixels: palm center (default) or index fingertip (config hand_point)
    y: float
    wrist_x: float  # landmark 0, pixels
    wrist_y: float
    handedness: str  # "Left" or "Right" as seen by the user
    frame_id: int
    distance_m: float | None = None  # camera -> hand, metres; None if unknown
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2 of all
    # hand landmarks (pixels): the hand's outline, for hand/object overlap; None if unknown


class Direction(str, Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"
    UP_LEFT = "UL"
    UP_RIGHT = "UR"
    DOWN_LEFT = "DL"
    DOWN_RIGHT = "DR"
    FORWARD = "F"  # aligned; move the hand deeper into the scene (away from the camera)
    BACK = "B"  # aligned; hand went past the object, pull back toward the camera
    ARRIVED = "G"  # grab now
    LOST = "X"  # hand not visible
    STOP = "S"  # all motors off


@dataclass(frozen=True)
class GuidanceCommand:
    direction: Direction
    intensity: int  # 0 to 255, stronger = farther from target
    distance_px: float
