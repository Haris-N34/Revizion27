import numpy as np
import pytest

from app import voice_out
from app.voice_in import BLOCK_S, SAMPLE_RATE, end_of_speech, extract_target

# Stand-in for vision.yolo.model.classes.normalize (not on this branch).
LABELS = {
    "can": "can",
    "coke": "can",
    "bottle": "bottle",
    "water bottle": "bottle",
    "phone": "cell phone",
    "cell phone": "cell phone",
    "mug": "cup",
    "coffee mug": "cup",
}


def normalize(phrase):
    return LABELS.get(phrase)


@pytest.mark.parametrize(
    "text, target",
    [
        ("Where is the can?", "can"),
        ("where's my coke", "can"),
        ("Find my water bottle.", "bottle"),
        ("Can you find my phone?", "cell phone"),
        ("can I get the can", "can"),
        ("Where's my coffee mug?", "cup"),
        ("take me to a bottle please", "bottle"),
        ("hello there", None),
        ("", None),
    ],
)
def test_extract_target(text, target):
    assert extract_target(text, normalize) == target


def _blocks(levels):
    n = int(SAMPLE_RATE * BLOCK_S)
    rng = np.random.default_rng(0)
    return [rng.normal(0, lvl, n).astype(np.float32) for lvl in levels]


def test_end_of_speech_after_silence():
    quiet, loud = 0.002, 0.2
    speech = [quiet] * 3 + [loud] * 10
    assert not end_of_speech(_blocks(speech))
    assert not end_of_speech(_blocks(speech + [quiet] * 5))
    assert end_of_speech(_blocks(speech + [quiet] * 8))


def test_end_of_speech_gives_up_without_speech():
    assert not end_of_speech(_blocks([0.002] * 20))
    assert end_of_speech(_blocks([0.002] * 40))


def test_end_of_speech_max_length():
    assert end_of_speech(_blocks([0.2] * 60))


def test_disabled_speaker_prints(capsys):
    speaker = voice_out.Speaker(enabled=False)
    speaker.say(voice_out.NOT_FOUND.format("can"))
    speaker.wait()
    assert "I can't find a can" in capsys.readouterr().out
    assert not speaker.busy()
