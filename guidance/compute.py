"""compute(): one frame of hand -> target guidance. Pure function, no I/O, no state.

Camera-independent: alignment tolerances come from the target's box size and the
intensity scale from the frame size, so a webcam and an iPhone (1x or 0.5x, any
resolution) behave the same. Only `mirror` depends on where the camera faces.
"""

import math

from shared.contracts import Detection, Direction, GuidanceCommand, HandPosition

DEFAULT_ARRIVE_PX = 40
DEFAULT_DEAD_ZONE_PX = 25
DEFAULT_DEPTH_TOLERANCE = 0.15
# Depth gap (fraction of the object's distance) at which FORWARD/BACK intensity maxes out.
FULL_DEPTH_INTENSITY_AT = 0.5
MIN_INTENSITY, MAX_INTENSITY = 80, 255
# Distance (as a fraction of the frame diagonal) at which intensity maxes out.
FULL_INTENSITY_AT = 0.5

_DIRECTIONS = {
    (0, -1): Direction.UP,
    (0, 1): Direction.DOWN,
    (-1, 0): Direction.LEFT,
    (1, 0): Direction.RIGHT,
    (-1, -1): Direction.UP_LEFT,
    (1, -1): Direction.UP_RIGHT,
    (-1, 1): Direction.DOWN_LEFT,
    (1, 1): Direction.DOWN_RIGHT,
}


def _axis(offset: float, tolerance: float) -> int:
    """-1, 0 or +1: which way to move on one axis, 0 when already aligned."""
    if abs(offset) <= tolerance:
        return 0
    return 1 if offset > 0 else -1


def _depth_command(
    det: Detection, hand: HandPosition, distance: float, config: dict
) -> GuidanceCommand:
    """FORWARD / BACK until hand and object are at the same depth, then ARRIVED.

    Without both distances this is plain 2D arrival (ARRIVED), as before depth existed.
    """
    if not det.distance_m or hand.distance_m is None:
        return GuidanceCommand(Direction.ARRIVED, MAX_INTENSITY, distance)
    g = config.get("guidance") or {}
    tolerance = float(g.get("depth_tolerance", DEFAULT_DEPTH_TOLERANCE))
    # > 0: hand is nearer the camera than the object, so it has to go deeper.
    gap = (det.distance_m - hand.distance_m) / det.distance_m
    if abs(gap) <= tolerance:
        return GuidanceCommand(Direction.ARRIVED, MAX_INTENSITY, distance)
    deeper = gap > 0
    if config.get("mirror"):
        deeper = not deeper  # camera faces the user: deeper in the image is toward them
    scale = min(1.0, abs(gap) / FULL_DEPTH_INTENSITY_AT)
    intensity = round(MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * scale)
    return GuidanceCommand(Direction.FORWARD if deeper else Direction.BACK, intensity, distance)


def compute(
    det: Detection, hand: HandPosition | None, frame_shape, config: dict
) -> GuidanceCommand:
    """Direction and strength to move the hand (index fingertip) toward the target.

    - No hand: LOST.
    - Fingertip inside the target box, or within arrive_px of its center: lined up.
      Then, if both distances are known: FORWARD / BACK until the hand's depth is
      within depth_tolerance of the object's, then ARRIVED. Otherwise ARRIVED.
    - Each axis counts as aligned while the fingertip is within the box's span on
      that axis (never tighter than dead_zone_px), so a big/close object is easy to
      line up with and a small/far one needs more precision.
    - Intensity grows with distance, 80..255, so it is always strong enough to feel.
    """
    if hand is None:
        return GuidanceCommand(Direction.LOST, 0, 0.0)

    g = config.get("guidance") or {}
    arrive_px = float(g.get("arrive_px", DEFAULT_ARRIVE_PX))
    dead_zone = float(g.get("dead_zone_px", DEFAULT_DEAD_ZONE_PX))

    cx, cy = det.center
    dx, dy = cx - hand.x, cy - hand.y
    distance = math.hypot(dx, dy)

    inside = det.x1 <= hand.x <= det.x2 and det.y1 <= hand.y <= det.y2
    if inside or distance <= arrive_px:
        # Lined up in the image; with depth known, finish along the line of sight.
        return _depth_command(det, hand, distance, config)

    if config.get("mirror"):
        dx = -dx  # camera faces the user: image right is the user's left
    tol_x = max(dead_zone, (det.x2 - det.x1) / 2)
    tol_y = max(dead_zone, (det.y2 - det.y1) / 2)
    step = (_axis(dx, tol_x), _axis(dy, tol_y))
    if step == (0, 0):
        # Aligned on both axes but just outside the box corner: finish on the larger offset.
        step = (_axis(dx, 0), 0) if abs(dx) >= abs(dy) else (0, _axis(dy, 0))

    height, width = frame_shape[:2]
    full = FULL_INTENSITY_AT * math.hypot(width, height)
    scale = min(1.0, distance / full) if full else 1.0
    intensity = round(MIN_INTENSITY + (MAX_INTENSITY - MIN_INTENSITY) * scale)
    return GuidanceCommand(_DIRECTIONS[step], intensity, distance)
