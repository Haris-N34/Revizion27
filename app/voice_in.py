"""Speech in: press-to-talk recording, offline Whisper transcription, object extraction.

Listener.start() records on a background thread until the user stops talking
(energy-based end-of-speech, max MAX_S seconds), then transcribes with faster-whisper.
The video loop polls Listener.result() so it never blocks.
"""

import re
import threading
from collections.abc import Callable

import numpy as np

SAMPLE_RATE = 16000
BLOCK_S = 0.1
NOISE_S = 0.3  # first 0.3 s sets the noise floor
END_SILENCE_S = 0.8  # this much quiet after speech ends the recording
NO_SPEECH_S = 4.0  # give up if nobody talks
MAX_S = 6.0
MODEL_SIZE = "base.en"

# "can" is usually a verb in these, not the object.
_CAN_VERB_NEXT = {"you", "i", "we", "u", "someone", "it", "anyone"}


def extract_target(text: str, normalize: Callable[[str], str | None]) -> str | None:
    """Find the object in a sentence: longest matching phrase, the latest one on ties.

    "where is the can" -> can, "find my water bottle" -> bottle,
    "can you find my phone" -> cell phone.
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    best = None  # (length, start, label)
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            phrase = words[i : i + n]
            if phrase[0] == "can" and i + 1 < len(words) and words[i + 1] in _CAN_VERB_NEXT:
                continue
            label = normalize(" ".join(phrase))
            if label and (best is None or (n, i) > best[:2]):
                best = (n, i, label)
    return best[2] if best else None


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def end_of_speech(blocks: list[np.ndarray]) -> bool:
    """True when the recording so far should stop (pure; tested without a mic)."""
    n_noise = max(1, int(NOISE_S / BLOCK_S))
    if len(blocks) * BLOCK_S >= MAX_S:
        return True
    if len(blocks) <= n_noise:
        return False
    noise = np.median([_rms(b) for b in blocks[:n_noise]])
    threshold = max(noise * 3, 0.01)
    loud = [_rms(b) > threshold for b in blocks[n_noise:]]
    if not any(loud):
        return len(blocks) * BLOCK_S >= NO_SPEECH_S
    last_loud = len(loud) - 1 - loud[::-1].index(True)
    quiet_after = (len(loud) - 1 - last_loud) * BLOCK_S
    return quiet_after >= END_SILENCE_S


class Listener:
    def __init__(self, hint_words: list[str] | None = None, model_size: str = MODEL_SIZE):
        self.model_size = model_size
        words = ", ".join(hint_words or [])
        self.hint = f"Where is the can? Find my water bottle. {words}."
        self._model = None
        self._model_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: str | None = None
        self.state = "idle"  # idle | recording | transcribing

    def preload(self) -> None:
        """Load Whisper in the background so the first press is fast."""
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        with self._model_lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def start(self) -> bool:
        """Begin listening; False if already busy."""
        if self.state != "idle":
            return False
        self._result = None
        self.state = "recording"
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        return True

    def result(self) -> str | None:
        """The transcript once, after listening finishes ('' if nothing was heard)."""
        if self.state == "idle" and self._result is not None:
            text, self._result = self._result, None
            return text
        return None

    def _listen(self) -> None:
        try:
            audio = self._record()
            self.state = "transcribing"
            self._result = self.transcribe(audio)
        except Exception as e:  # noqa: BLE001 - mic/model errors must not kill the app
            print(f"[listen] failed: {e}")
            self._result = ""
        finally:
            self.state = "idle"

    def _record(self) -> np.ndarray:
        import sounddevice as sd

        blocks: list[np.ndarray] = []
        frames = int(SAMPLE_RATE * BLOCK_S)
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while not end_of_speech(blocks):
                block, _ = stream.read(frames)
                blocks.append(block[:, 0].copy())
        return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < SAMPLE_RATE * 0.3:
            return ""
        segments, _ = self._load().transcribe(
            audio, language="en", beam_size=1, initial_prompt=self.hint
        )
        text = " ".join(s.text for s in segments).strip()
        print(f"[heard] {text!r}")
        return text
