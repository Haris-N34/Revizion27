# guidance (Haris)

`compute(det, hand, frame_shape, config) -> GuidanceCommand`: pure, one frame.
`Stabilizer(config).update(cmd)`: stops flicker; LOST/ARRIVED must hold a few frames.
Hand distance (`HandPosition.distance_m`) comes from `vision/hands` palm size.

1. Demo (no camera, mouse = hand point): `python -m guidance.demo`
2. Tests: `pytest guidance`
`turn_toward(last_det, frame_shape, config)`: target left the view at an edge -> turn LEFT/RIGHT.

Camera-independent: alignment uses the target box's size, intensity the frame size, so a
webcam or iPhone (any lens/resolution) behaves the same. Set `mirror: true` only when
the camera faces the user (laptop webcam); a chest-mounted camera uses `mirror: false`.
Depth: with distances on both det and hand, a lined-up hand gets FORWARD/BACK until
within guidance.depth_tolerance, then ARRIVED (demo: w/s change the hand depth).
