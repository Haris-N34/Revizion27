"""Spoken lines, kept short and consistent, and a non-blocking speaker.

macOS uses the built-in `say` command; Windows/Linux use pyttsx3. Speech runs on a
background thread so the video loop never waits for it. Every line is also printed.
"""

import queue
import shutil
import subprocess
import sys
import threading
import time

LISTENING = "Listening."
LOOKING = "Looking for a {}."
NOT_FOUND = "I can't find a {}"
CANT_LOOK = "I can't look for that yet"
DIDNT_HEAR = "Sorry, I didn't catch that."
HAND_LOST = "I can't see your hand."
LOST_TARGET = "I lost the {}. Looking again."
ARRIVED = "You're there"


class Speaker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._busy = threading.Event()
        self._thread = None
        if enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def say(self, text: str) -> None:
        print(f"[say] {text}")
        if self.enabled:
            self._busy.set()
            self._queue.put(text)

    def busy(self) -> bool:
        return self._busy.is_set()

    def wait(self, timeout: float = 3.0) -> None:
        """Block until queued speech is done (e.g. before listening), at most `timeout` s."""
        end = time.monotonic() + timeout
        while self.enabled and self._busy.is_set() and time.monotonic() < end:
            time.sleep(0.02)

    def close(self) -> None:
        if self._thread:
            self._queue.put(None)
            self._thread.join(timeout=5)

    def _run(self) -> None:
        speak_one = _backend()
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                speak_one(text)
            except Exception as e:  # noqa: BLE001 - never let TTS kill the app
                print(f"[say] speech failed: {e}")
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._busy.clear()


def _backend():
    if sys.platform == "darwin" and shutil.which("say"):
        return lambda text: subprocess.run(["say", text], check=False)
    import pyttsx3

    engine = pyttsx3.init()  # created on the speaker thread, used only there

    def speak_one(text: str) -> None:
        engine.say(text)
        engine.runAndWait()

    return speak_one


_default: Speaker | None = None


def speak(text: str) -> None:
    """Module-level helper; prints only until configure() turns speech on."""
    if _default is None:
        print(f"[say] {text}")
    else:
        _default.say(text)


def configure(enabled: bool) -> Speaker:
    global _default
    _default = Speaker(enabled)
    return _default
