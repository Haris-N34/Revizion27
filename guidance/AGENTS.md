# AGENTS.md: guidance/

Owner: **UNASSIGNED** (assign at kickoff, then update the root table).
Branch: `feat/guidance`. Read the root `AGENTS.md` first.

```
guidance/
  compute.py       # compute(det, hand, frame_shape, config) -> GuidanceCommand
  demo.py          # python -m guidance.demo  (click to place can and hand, prints command)
  tests/
  requirements.txt # none beyond the standard library, ideally
```

## Job

`compute(det, hand, frame_shape, config)` is a **pure function**: no camera,
no serial, no printing, no global state. That makes it trivial to test and
merge.

1. If `hand` is `None`: return `Direction.LOST`, intensity 0.
2. Target point = `det.center`. Hand point = `(hand.x, hand.y)`.
3. `dx = target_x - hand_x`, `dy = target_y - hand_y` (pixels; y grows down).
   If `config["mirror"]` is true, use `-dx`.
4. If the fingertip is inside the detection box, or
   `hypot(dx, dy) < guidance.arrive_px`: return `ARRIVED`, intensity 255.
5. Ignore any axis whose absolute value is under `guidance.dead_zone_px`.
6. Direction from the signs: `dx > 0` is RIGHT, `dy < 0` is UP; both axes
   active gives a diagonal (`UP_LEFT` and so on).
7. Intensity: scale distance to 80 to 255 (farther is stronger), so the
   motor is always strong enough to feel.
8. Return `GuidanceCommand(direction, intensity, distance_px)`.

Defaults if keys are missing: `arrive_px = 40`, `dead_zone_px = 25`.

## Limits to know

With one webcam this is 2D: it guides the hand across the image, not toward
or away from the camera. That is fine for objects on a table filmed from
above or in front. Depth is a stretch goal and would need a new contract
field, which means a `feat/shared` PR.

## Tests

Table-driven pytest cases: each direction, both diagonals, dead zone,
arrival inside the box, mirror flip, hand missing.
