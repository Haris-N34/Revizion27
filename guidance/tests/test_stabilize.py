from guidance.stabilize import Stabilizer
from shared.contracts import Direction, GuidanceCommand

D = Direction


def cmd(direction, intensity=150, distance=100.0):
    return GuidanceCommand(direction, intensity, distance)


def run(stab, directions):
    return [stab.update(cmd(d)).direction for d in directions]


def test_first_direction_is_immediate():
    assert run(Stabilizer(), [D.RIGHT]) == [D.RIGHT]


def test_single_frame_flicker_is_ignored():
    out = run(Stabilizer(), [D.RIGHT, D.UP_RIGHT, D.RIGHT, D.UP_RIGHT, D.RIGHT])
    assert out == [D.RIGHT] * 5


def test_switch_after_three_frames():
    out = run(Stabilizer(), [D.RIGHT, D.UP, D.UP, D.UP, D.UP])
    assert out == [D.RIGHT, D.RIGHT, D.RIGHT, D.UP, D.UP]


def test_lost_needs_five_frames():
    out = run(Stabilizer(), [D.RIGHT] + [D.LOST] * 5)
    assert out == [D.RIGHT] * 5 + [D.LOST]


def test_arrived_needs_six_frames_even_at_start():
    out = run(Stabilizer(), [D.ARRIVED] * 6)
    assert out[:5] != [D.ARRIVED] * 5
    assert D.ARRIVED not in out[:5]
    assert out[5] == D.ARRIVED


def test_hand_sweeping_past_is_not_a_grab():
    out = run(Stabilizer(), [D.RIGHT, D.ARRIVED, D.ARRIVED, D.LEFT, D.LEFT, D.LEFT])
    assert D.ARRIVED not in out


def test_holding_keeps_last_intensity_not_arrived_255():
    stab = Stabilizer()
    stab.update(cmd(D.RIGHT, intensity=120))
    held = stab.update(cmd(D.ARRIVED, intensity=255, distance=10))
    assert (held.direction, held.intensity, held.distance_px) == (D.RIGHT, 120, 10)


def test_config_overrides_and_reset():
    stab = Stabilizer({"guidance": {"switch_frames": 1}})
    assert run(stab, [D.RIGHT, D.UP]) == [D.RIGHT, D.UP]
    stab.reset()
    assert stab.current is None
