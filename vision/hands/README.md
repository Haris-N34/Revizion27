# vision/hands (Haris)

`HandTracker(config).track(frame) -> HandPosition | None` using MediaPipe Tasks HandLandmarker.
`tracker.landmarks` holds all 21 points for drawing; `draw.draw_hand(frame, landmarks)` draws them.

1. `pip install -r vision/hands/requirements.txt`
2. Demo: `python -m vision.hands.demo` (webcam; `--video FILE`; q to quit)
3. Tests: `pytest vision/hands`

The model (`models/hand_landmarker.task`, ~8 MB, gitignored) downloads on first use.
