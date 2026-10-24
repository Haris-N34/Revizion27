from app.narrator import Narrator
from shared.contracts import Direction, GuidanceCommand

D = Direction


class FakeSpeaker:
    def __init__(self):
        self.said = []
        self.talking = False

    def say(self, text):
        self.said.append(text)

    def busy(self):
        return self.talking


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def cmd(d):
    return GuidanceCommand(d, 150, 100.0)


def make():
    speaker, clock = FakeSpeaker(), Clock()
    return Narrator(speaker, min_gap_s=1.0, repeat_s=3.0, clock=clock), speaker, clock


def feed(narrator, clock, timeline):
    """timeline: [(time, direction)]"""
    for t, d in timeline:
        clock.t = t
        narrator.update(cmd(d))


def test_speaks_first_direction_immediately():
    n, speaker, clock = make()
    feed(n, clock, [(0.0, D.LEFT)])
    assert speaker.said == ["left"]


def test_same_direction_is_not_repeated_every_frame():
    n, speaker, clock = make()
    feed(n, clock, [(t / 10, D.LEFT) for t in range(25)])  # 2.5 s of "left"
    assert speaker.said == ["left"]


def test_repeats_after_repeat_interval():
    n, speaker, clock = make()
    feed(n, clock, [(0.0, D.LEFT), (2.9, D.LEFT), (3.0, D.LEFT), (5.0, D.LEFT), (6.0, D.LEFT)])
    assert speaker.said == ["left", "left", "left"]


def test_change_waits_for_min_gap():
    n, speaker, clock = make()
    feed(n, clock, [(0.0, D.LEFT), (0.5, D.UP), (0.9, D.UP), (1.0, D.UP)])
    assert speaker.said == ["left", "up"]


def test_never_talks_over_itself():
    n, speaker, clock = make()
    feed(n, clock, [(0.0, D.LEFT)])
    speaker.talking = True
    feed(n, clock, [(5.0, D.UP_RIGHT)])
    speaker.talking = False
    feed(n, clock, [(5.1, D.UP_RIGHT)])
    assert speaker.said == ["left", "up right"]


def test_lost_and_arrived_are_not_narrated_but_reset_direction():
    n, speaker, clock = make()
    feed(n, clock, [(0.0, D.LEFT), (1.5, D.LOST), (1.6, D.ARRIVED), (2.0, D.LEFT)])
    assert speaker.said == ["left", "left"]  # LEFT again after LOST counts as a change


def test_reset_and_none():
    n, speaker, clock = make()
    assert n.update(None) is None
    feed(n, clock, [(0.0, D.DOWN)])
    n.reset()
    feed(n, clock, [(0.1, D.DOWN)])
    assert speaker.said == ["down", "down"]


def test_forward_and_back_are_spoken():
    n, speaker, clock = make()
    feed(n, clock, [(0.0, D.FORWARD), (2.0, D.BACK)])
    assert speaker.said == ["forward", "back"]
