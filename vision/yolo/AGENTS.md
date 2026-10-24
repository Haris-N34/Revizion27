# AGENTS.md: vision/yolo/

Owners: **Rizwan** (`model/`, branch `feat/yolo-rizwan`) and **Harim**
(`runtime/`, branch `feat/yolo-harim`). Read the root `AGENTS.md` first.

Two people share this area, so the split matters: each branch edits only its
own subfolder. If you're an agent, check the branch name and stay in the
matching subfolder.

```
vision/yolo/
  model/                 # Rizwan
    classes.py           # SUPPORTED list + normalize()
    weights.py           # weights_path(config)
    export.py            # how the weights were produced (training or YOLO-World)
    README.md            # where to download weights, how to rebuild them
    tests/
    requirements.txt     # ultralytics, etc.
  runtime/               # Harim
    detector.py          # ObjectDetector
    scan.py              # scan(... timeout_s) -> Detection | None
    demo.py              # python -m vision.yolo.runtime.demo --target can
    tests/
    requirements.txt
```

## The "can" problem (solve first)

Stock YOLO models use the 80 COCO classes: they include `bottle` and `cup`
but **not** `can`. Rizwan picks one approach in the first hour and records it
in `model/README.md`:

1. **YOLO-World** with custom classes, e.g.
   `model.set_classes(["can", "bottle", "cup", "cell phone", "remote"])`, then
   `model.save(...)`. Fastest; no training.
2. **Fine-tune** a small YOLO model on photos of cans (100 to 200 labelled
   images). Better accuracy, costs time.

Whichever is chosen, the weights file is **not committed**. Put it in a
shared drive or GitHub release, write the link and a SHA-256 checksum in
`model/README.md`, and have `weights_path()` point at `models/` (gitignored)
under the repo root.

## Rizwan: `model/`

- `SUPPORTED`: the labels the model can return, lowercase.
- `normalize(spoken)`: maps what a user might say to a supported label
  ("coke", "soda", "pop can", "can of coke" to `can`; "water bottle" to
  `bottle`). Returns `None` when unsupported, so the app can say "I can't look
  for that yet."
- Tests: `normalize` cases, and that every value in `SUPPORTED` is a label
  the weights actually produce.

## Harim: `runtime/`

- `ObjectDetector(config).detect(frame, target)`: run YOLO on one frame, keep
  only boxes whose label equals `target` and confidence is at or above
  `config["yolo"]["confidence"]`, return `Detection` objects sorted by
  confidence. Get weights via `model.weights.weights_path(config)` and labels
  via `model.classes`. Code against those signatures even before Rizwan's
  branch is merged; stub them locally in tests, not in `model/`.
- `scan(frames, detector, target, timeout_s)`: return the first detection
  seen in `confirm_frames` consecutive frames; return `None` if the timeout
  passes. "Not found" must be a clear `None`, never a low-confidence guess.
- `demo.py`: opens a webcam or `--video` file, draws boxes, prints FOUND or
  NOT FOUND.
- Tests run on the still images in `runtime/tests/fixtures/` with a fake
  detector; no webcam needed.

## Performance target

10 or more frames per second on the demo laptop. Start with the nano or
small model size and a 640 px input; only go bigger if accuracy fails.

## Don't

- Don't open the camera outside `demo.py`; the app passes frames in.
- Don't import from `app/`, `hardware/` or `guidance/`.
- Don't commit weights, datasets or training runs (`runs/` is gitignored).
