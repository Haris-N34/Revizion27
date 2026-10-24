# AGENTS.md: vision/hands/

Owner: **UNASSIGNED** (assign at kickoff, then update the root table).
Branch: `feat/hands`. Read the root `AGENTS.md` first.

```
vision/hands/
  tracker.py       # HandTracker
  demo.py          # python -m vision.hands.demo  (draws landmarks on webcam)
  tests/
  requirements.txt # mediapipe, opencv-python
```

## Job

`HandTracker(config).track(frame) -> HandPosition | None`

- Use MediaPipe Hands, max 1 hand, video mode (reuse the same instance across
  frames, don't recreate it each call).
- MediaPipe wants RGB; frames arrive as OpenCV BGR. Convert inside `track`.
- Convert normalized landmarks to pixels using the frame's width and height.
- Fill `x`, `y` from landmark 8 (index fingertip) and `wrist_x`, `wrist_y`
  from landmark 0.
- If `config["mirror"]` is true, flip `handedness` so it is from the user's
  point of view.
- Return `None` when no hand is found or confidence is below 0.5.
- Add `close()` that releases MediaPipe resources.

## Target

15 or more frames per second alongside YOLO on the demo laptop.

## Tests

Run on still images in `tests/fixtures/` (one with a hand, one without). No
webcam in tests.

## Don't

- Don't open the camera outside `demo.py`.
- Don't compute directions here; that's `guidance/`.
