import pytest

from guidance.compute import MAX_INTENSITY, MIN_INTENSITY, compute
from shared.contracts import Detection, Direction, HandPosition

SHAPE = (480, 640, 3)
# Can box centered at (320, 240), 60 wide x 100 tall.
CAN = Detection("can", 0.9, 290, 190, 350, 290, 0)
CFG = {"mirror": False, "guidance": {"arrive_px": 40, "dead_zone_px": 25}}


def hand(x, y):
    return HandPosition(x, y, x, y + 60, "Right", 0)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (100, 240, Direction.RIGHT),  # target to the right
        (550, 240, Direction.LEFT),
        (320, 450, Direction.UP),  # y grows down: target above
        (320, 30, Direction.DOWN),
        (100, 450, Direction.UP_RIGHT),
        (550, 450, Direction.UP_LEFT),
        (100, 30, Direction.DOWN_RIGHT),
        (550, 30, Direction.DOWN_LEFT),
    ],
)
def test_directions(x, y, expected):
    assert compute(CAN, hand(x, y), SHAPE, CFG).direction == expected


def test_aligned_axis_is_ignored_within_box_span():
    # Fingertip x is inside the box's x-span (310 in 290..350): only "up", no diagonal.
    assert compute(CAN, hand(345, 450), SHAPE, CFG).direction == Direction.UP
    # Just outside the span: diagonal.
    assert compute(CAN, hand(360, 450), SHAPE, CFG).direction == Direction.UP_LEFT


def test_dead_zone_is_minimum_tolerance_for_tiny_boxes():
    tiny = Detection("can", 0.9, 318, 238, 322, 242, 0)  # 4x4 box at (320, 240)
    cfg = {"guidance": {"arrive_px": 5, "dead_zone_px": 25}}
    # 20 px off in x (inside the 25 px dead zone), far below: just UP.
    assert compute(tiny, hand(340, 400), SHAPE, cfg).direction == Direction.UP


def test_arrived_inside_box():
    cmd = compute(CAN, hand(295, 195), SHAPE, CFG)  # near a corner, but inside
    assert cmd.direction == Direction.ARRIVED
    assert cmd.intensity == MAX_INTENSITY


def test_arrived_within_arrive_px_outside_box():
    small = Detection("can", 0.9, 315, 235, 325, 245, 0)
    assert compute(small, hand(320, 275), SHAPE, CFG).direction == Direction.ARRIVED  # 35 px


def test_mirror_flips_left_right_only():
    mirrored = dict(CFG, mirror=True)
    assert compute(CAN, hand(100, 240), SHAPE, mirrored).direction == Direction.LEFT
    assert compute(CAN, hand(100, 450), SHAPE, mirrored).direction == Direction.UP_LEFT
    assert compute(CAN, hand(320, 450), SHAPE, mirrored).direction == Direction.UP


def test_hand_missing_is_lost():
    cmd = compute(CAN, None, SHAPE, CFG)
    assert (cmd.direction, cmd.intensity) == (Direction.LOST, 0)


def test_intensity_grows_with_distance_and_stays_in_range():
    near = compute(CAN, hand(250, 240), SHAPE, CFG)
    far = compute(CAN, hand(0, 240), SHAPE, CFG)
    corner = compute(CAN, hand(0, 479), (480, 640, 3), CFG)
    assert MIN_INTENSITY <= near.intensity < far.intensity <= MAX_INTENSITY
    assert corner.intensity <= MAX_INTENSITY
    assert near.distance_px == pytest.approx(70)


def test_resolution_independent():
    """Same layout at 2x resolution (e.g. iPhone vs webcam) gives the same command."""
    big = Detection("can", 0.9, 580, 380, 700, 580, 0)
    small_cmd = compute(CAN, hand(100, 450), (480, 640, 3), CFG)
    big_cmd = compute(big, hand(200, 900), (960, 1280, 3), CFG)
    assert small_cmd.direction == big_cmd.direction
    assert small_cmd.intensity == big_cmd.intensity


def test_missing_config_keys_use_defaults():
    assert compute(CAN, hand(100, 240), SHAPE, {}).direction == Direction.RIGHT
