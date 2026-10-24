"""Spoken lines, kept short and consistent. speak() prints until TTS is wired up."""

LOOKING = "Looking for a {}."
NOT_FOUND = "I can't find a {}"
CANT_LOOK = "I can't look for that yet"
HAND_LOST = "I can't see your hand."
ARRIVED = "You're there"


def speak(text: str) -> None:
    print(f"[say] {text}")
