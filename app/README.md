# app (Harim)

Main loop + live debug view: scan for the target, then guide the hand, window stays open.

1. `python -m app.main --target can --mock-band` (webcam; `--video FILE`, `--no-window`)
2. Window: target box, hand skeleton, fingertip -> target arrow, band command. `r` rescan, `q` quit.
3. Tests (fakes only, no camera): `pytest app`

Modules not merged yet are replaced by `app/fakes.py`; startup prints which fakes are active.
Voice in/out is not wired up yet: target comes from `--target` or the keyboard, speech is printed.
