# app (Harim)

Voice-driven loop + live debug view: ask for an object, it scans, then guides your hand.

1. `python -m app.main --mock-band` (webcam; starts listening). `--target can` skips asking,
   `--no-voice` types instead of talking, `--video FILE`, `--no-window`.
2. Keys: SPACE ask ("where is the can?"), r rescan, q quit.
3. Tests (no camera, mic or internet): `pytest app`

Voice (offline, Windows + macOS): speech in = faster-whisper `base.en` (downloads ~150 MB once),
mic via sounddevice, end of speech by silence. Speech out = macOS `say`, pyttsx3 elsewhere.
Modules not merged yet are replaced by `app/fakes.py`; startup prints which fakes are active.
