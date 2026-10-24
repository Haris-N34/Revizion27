"""Labels the model returns, and mapping from what a user says to those labels."""

import re

# Lowercase labels the weights produce. export.py bakes exactly this list into
# the YOLO-World model, so model.names == SUPPORTED.
SUPPORTED: list[str] = ["can", "bottle", "cup", "cell phone", "remote"]

# Text YOLO-World actually searches for, when the bare label works poorly.
# Bare "can" scored 0 on a real Coke can (it said "bottle" 0.54); "drink can" 0.94.
PROMPTS: dict[str, str] = {"can": "drink can"}

# Spoken phrase -> label. Keys are already cleaned by _clean().
ALIASES: dict[str, str] = {
    # can
    "coke": "can",
    "coke can": "can",
    "can of coke": "can",
    "soda": "can",
    "soda can": "can",
    "can of soda": "can",
    "pop": "can",
    "pop can": "can",
    "can of pop": "can",
    "pepsi": "can",
    "sprite": "can",
    "beer": "can",
    "beer can": "can",
    "energy drink": "can",
    "red bull": "can",
    "tin": "can",
    "tin can": "can",
    # bottle
    "water bottle": "bottle",
    "bottle of water": "bottle",
    "water": "bottle",
    "plastic bottle": "bottle",
    # cup
    "mug": "cup",
    "coffee cup": "cup",
    "coffee mug": "cup",
    "coffee": "cup",
    "glass": "cup",
    "tea cup": "cup",
    "teacup": "cup",
    # cell phone
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "mobile phone": "cell phone",
    "smartphone": "cell phone",
    "iphone": "cell phone",
    # remote
    "remote control": "remote",
    "tv remote": "remote",
    "controller": "remote",
    "clicker": "remote",
}

_FILLER = {"a", "an", "the", "my", "some", "me", "to", "take", "find", "please", "get"}


def _clean(spoken: str) -> str:
    words = re.sub(r"[^a-z0-9 ]+", " ", spoken.lower()).split()
    words = [w for w in words if w not in _FILLER]
    return " ".join(words)


def _singular(phrase: str) -> str:
    if phrase.endswith("ies"):
        return phrase[:-3] + "y"
    if phrase.endswith("s") and not phrase.endswith("ss"):
        return phrase[:-1]
    return phrase


def normalize(spoken: str) -> str | None:
    """Map what the user said to a SUPPORTED label, or None if we can't look for it."""
    phrase = _clean(spoken)
    for candidate in (phrase, _singular(phrase)):
        if candidate in SUPPORTED:
            return candidate
        if candidate in ALIASES:
            return ALIASES[candidate]
    return None
