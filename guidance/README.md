# guidance (Haris)

`compute(det, hand, frame_shape, config) -> GuidanceCommand`: pure, one frame.
`Stabilizer(config).update(cmd)`: stops flicker; LOST/ARRIVED must hold a few frames.

1. Demo (no camera, mouse = fingertip): `python -m guidance.demo`
2. Tests: `pytest guidance`

Camera-independent: alignment uses the target box's size, intensity the frame size, so a
webcam or iPhone (any lens/resolution) behaves the same. Set `mirror: true` only when
the camera faces the user (laptop webcam); a chest-mounted camera uses `mirror: false`.
2D only for now; depth is the next step.
