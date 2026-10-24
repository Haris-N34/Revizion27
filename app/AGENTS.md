# AGENTS.md: app/

Owner: **Harim**. Branch: `feat/harim`. Read the root `AGENTS.md` first.

The app is the only place the pieces meet. It owns the camera, the voice
interface and the main loop. It must run end to end with mocks before any
other branch is merged.

```
app/
  main.py          # python -m app.main [--mock-band] [--video FILE] [--no-voice]
  voice_in.py      # speech to text -> target word
  voice_out.py     # text to speech
  loop.py          # the find-and-guide state machine
  debug_view.py    # optional OpenCV window with boxes and hand point
  fakes.py         # FakeDetector, FakeHandTracker, fake_compute for running alone
  tests/
  requirements.txt
  README.md
```

## Main loop (`loop.py`)

```
IDLE -> LISTENING -> SCANNING -> GUIDING -> DONE
                         |
                         +-> NOT_FOUND (say it, back to IDLE)
```

1. **LISTENING:** get a phrase like "take me to a can". Extract the object and
   pass it through `vision.yolo.model.classes.normalize()`. If it returns
   `None`, say "I can't look for that yet" and return to IDLE.
2. **SCANNING:** say "Looking for a can." Call `scan(...)` with
   `config["scan_timeout_s"]`. If it returns `None`, say
   **"I can't find a can"** and return to IDLE. Say it plainly; no hedging,
   no guidance toward anything else.
3. **GUIDING:** each frame: `detector.detect`, `HandTracker.track`,
   `guidance.compute`, `band.send`. If the can drops out for a few frames,
   keep using its last known box. If `compute` returns `ARRIVED`, go to DONE.
   If it returns `LOST` for more than 2 s, say "I can't see your hand."
4. **DONE:** say "You're there", send `STOP`, return to IDLE.
5. Any exception or Ctrl+C: send `STOP`, close the band, then exit.

## Voice

- Pick libraries that work offline on both Windows and macOS (for example
  `pyttsx3` for speech out; `vosk` or `faster-whisper` for speech in). Record
  the choice in `README.md`.
- `--no-voice` flag: read the target from the keyboard and print speech, so
  teammates can test without a mic.
- Keep spoken lines short and consistent; list them as constants in
  `voice_out.py`.

## Running before other branches exist

`fakes.py` provides stand-ins that satisfy the root contracts:
`FakeDetector` (returns a fixed box after N frames, or never, for the
not-found path), `FakeHandTracker` (moves a point toward the box) and
`fake_compute`. `main.py` uses them automatically when the real module can't
be imported, and prints which fakes are active. Delete nothing from `fakes.py`
after the merge; tests still use it.

## Tests

- State machine tests with fakes: found path ends in DONE, not-found path
  says the exact not-found line, hand-lost path triggers the warning, and
  `STOP` is always sent on exit.
- No camera, mic, Arduino or internet in tests.

## Don't

- Don't put YOLO, MediaPipe or guidance math in `app/`; call the modules.
- Don't edit any other folder, even when integration exposes a bug there.
  Write the bug down and tell its owner.
