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


# --- depth -----------------------------------------------------------------
CAN_AT = Detection("can", 0.9, 290, 190, 350, 290, 0, distance_m=0.60)


def hand_at(x, y, metres):
    return HandPosition(x, y, x, y + 60, "Right", 0, distance_m=metres)


@pytest.mark.parametrize(
    "hand_m, expected",
    [
        (0.30, Direction.FORWARD),  # hand halfway: reach deeper
        (0.50, Direction.FORWARD),  # 17% short: still forward
        (0.55, Direction.ARRIVED),  # within 15%
        (0.60, Direction.ARRIVED),
        (0.68, Direction.ARRIVED),
        (0.80, Direction.BACK),  # past the object
    ],
)
def test_depth_when_lined_up(hand_m, expected):
    assert compute(CAN_AT, hand_at(320, 240, hand_m), SHAPE, CFG).direction == expected


def test_mirror_swaps_forward_back():
    mirrored = dict(CFG, mirror=True)
    assert compute(CAN_AT, hand_at(320, 240, 0.30), SHAPE, mirrored).direction == Direction.BACK
    assert compute(CAN_AT, hand_at(320, 240, 0.90), SHAPE, mirrored).direction == Direction.FORWARD


def test_line_up_in_2d_before_depth():
    # Far off to the side and much too close: fix the side first.
    assert compute(CAN_AT, hand_at(100, 240, 0.30), SHAPE, CFG).direction == Direction.RIGHT


def test_forward_intensity_grows_with_depth_gap():
    near = compute(CAN_AT, hand_at(320, 240, 0.48), SHAPE, CFG)
    far = compute(CAN_AT, hand_at(320, 240, 0.20), SHAPE, CFG)
    assert near.direction == far.direction == Direction.FORWARD
    assert MIN_INTENSITY <= near.intensity < far.intensity <= MAX_INTENSITY


def test_depth_tolerance_from_config():
    loose = {"guidance": {"depth_tolerance": 0.6}}
    assert compute(CAN_AT, hand_at(320, 240, 0.30), SHAPE, loose).direction == Direction.ARRIVED


def test_unknown_distance_falls_back_to_2d_arrival():
    assert compute(CAN, hand_at(320, 240, 0.30), SHAPE, CFG).direction == Direction.ARRIVED
    assert compute(CAN_AT, hand(320, 240), SHAPE, CFG).direction == Direction.ARRIVED


# --- out of view ---------------------------------------------------------------
def test_turn_toward_edge_it_left_from():
    from guidance.compute import turn_toward

    left = Detection("can", 0.9, 0, 200, 40, 300, 0)
    right = Detection("can", 0.9, 610, 200, 640, 300, 0)
    middle = Detection("can", 0.9, 290, 190, 350, 290, 0)
    assert turn_toward(left, SHAPE, CFG).direction == Direction.LEFT
    assert turn_toward(right, SHAPE, CFG).direction == Direction.RIGHT
    assert turn_toward(middle, SHAPE, CFG) is None
    mirrored = dict(CFG, mirror=True)
    assert turn_toward(left, SHAPE, mirrored).direction == Direction.RIGHT


# --- contact by overlap (holding / covering the object) -----------------------
def test_holding_phone_counts_as_arrived():
    """From a real frame: phone held in hand, YOLO sees only its top (camera bump),
    palm center sits below that box. The hand covers ~60% of the box: arrived."""
    phone = Detection("cell phone", 0.51, 1315, 460, 1690, 780, 0)
    palm = HandPosition(1405, 995, 1170, 1175, "Right", 0, box=(1150, 580, 1740, 1180))
    cmd = compute(phone, palm, (1080, 1920, 3), CFG)
    assert cmd.direction == Direction.ARRIVED


def test_little_overlap_is_not_contact():
    can = Detection("can", 0.9, 290, 190, 350, 290, 0)
    h = HandPosition(320, 450, 320, 480, "Right", 0, box=(300, 280, 400, 479))  # 10% overlap
    assert compute(can, h, SHAPE, CFG).direction == Direction.UP


def test_edge_margin_counts_as_lined_up():
    # 5 px outside the box's right edge (60 px wide box, 20% margin = 12 px): lined up.
    can = Detection("can", 0.9, 290, 190, 350, 290, 0)
    assert compute(can, hand(355, 240), SHAPE, CFG).direction == Direction.ARRIVED


def test_covered_fraction():
    from guidance.compute import covered_fraction

    box = Detection("can", 0.9, 0, 0, 100, 100, 0)
    assert covered_fraction(box, None) == 0
    assert covered_fraction(box, (50, 0, 150, 100)) == pytest.approx(0.5)
    assert covered_fraction(box, (200, 200, 300, 300)) == 0
