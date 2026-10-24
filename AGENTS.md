# AGENTS.md (repo root)

Read this file first, then the AGENTS.md inside the folder you are working in.
This file is the source of truth for how the pieces fit together. Folder files
add detail for one area only.

## What we are building

Octaby: a wearable that guides a blind user's hand to an object.

1. User puts on the wristband and says "take me to a can".
2. YOLO scans the webcam view for the object.
3. Not found after the scan timeout: the app says "I can't find a can" and stops.
4. Found: MediaPipe finds the user's hand, guidance computes direction and
   distance from hand to object.
5. The laptop sends a short command over USB serial to the Arduino, which
   buzzes the vibration motor for that direction.
6. Steps 4 and 5 repeat every frame until the hand reaches the object
   (Arduino plays the "grab" pattern).

YOLO and MediaPipe run on the laptop. The Arduino only drives motors.

## Team, folders, branches

One branch per person. Each branch edits only its owner's folders.

| Area | Owner | Folder | Branch |
| --- | --- | --- | --- |
| Hardware (Arduino firmware, motors, band, Python serial driver) | Azfar | `hardware/` | `feat/azfar` |
| YOLO model: classes, weights, training, export | Rizwan | `vision/yolo/model/` | `feat/rizwan` |
| YOLO runtime: detector + scan with timeout | Harim | `vision/yolo/runtime/` | `feat/harim` |
| UI app: voice in, speech out, camera, main loop, debug window | Harim | `app/` | `feat/harim` |
| MediaPipe hand tracking | Haris | `vision/hands/` | `feat/haris` |
| Guidance math | Haris | `guidance/` | `feat/haris` |
| Shared interfaces, config, repo setup | Whole team | `shared/` + root files | `main` only (see rule 3) |

The two YOLO owners are split by subfolder so their branches never edit the
same file. If the team changes this split, update this table on `main` first.

## Rules for agents (read before every task)

The team works on different computers (Windows and macOS), each in their own
branch, and merges everything at the end. Your job is to keep that merge
painless.

1. **Check where you are.** Run `git branch --show-current` before editing.
   Only edit files inside the folder(s) your branch's owner owns (table above).
2. **Never edit another area's files**, even to "fix" something. If you need a
   change elsewhere, stop and tell the human, who will ask the owner.
3. **Never edit `shared/` or root files from your personal branch.** For a
   contract/config/root change, make a short-lived branch off `main` (e.g.
   `shared-add-key`), open a tiny PR the whole team agrees on, and delete the
   branch after it merges. Then everyone runs `git pull origin main` into
   their own branch.
4. **Code against the contracts in `shared/contracts.py`, not against another
   person's implementation.** Their code may not exist yet on your machine.
5. **Every module must run without the others.** Provide a mock or fake for
   anything you depend on, and a standalone demo (`python -m <module>.demo`).
6. **No machine-specific values in code.** Camera index, serial port, file
   paths and model paths come from `config.yaml` (see Config). Use `pathlib`.
   Never write `C:\Users\...` or `/Users/...`.
7. **Do not commit:** `config.yaml`, `.venv/`, `__pycache__/`, `.DS_Store`,
   IDE folders, model weights (`*.pt`, `*.mlpackage`, `*.onnx`), or anything
   over 5 MB. The only videos in git are small test clips in
   `shared/fixtures/`. `.gitignore` covers the rest; don't override it.
8. **Do not reformat files you don't own** and don't run project-wide
   formatters. Format only files you changed (`ruff format <file>`).
9. **Do not change dependency versions** in another area's requirements file.
   Add your own deps only to your folder's `requirements.txt`.
10. **Small, frequent commits** with a prefix: `[hardware]`, `[yolo]`,
    `[app]`, `[hands]`, `[guidance]`, `[shared]` (by folder, not by person).
    Push your branch often so
    work is never only on one laptop.
11. **Keep `main` working.** Never commit directly to `main` except the agreed
    shared PRs from rule 3.

## Shared contracts (`shared/contracts.py`)

These types are the only thing modules share. Do not add fields from a
feature branch.

```python
from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class Detection:
    label: str            # normalized name, e.g. "can"
    confidence: float     # 0.0 to 1.0
    x1: float             # bounding box in pixels, origin top-left
    y1: float
    x2: float
    y2: float
    frame_id: int
    distance_m: float | None = None  # camera -> object, metres; None if unknown

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

@dataclass(frozen=True)
class HandPosition:
    x: float              # grab point, pixels: palm center (default) or
                          # index fingertip, set by config hand_point
    y: float
    wrist_x: float        # landmark 0, pixels
    wrist_y: float
    handedness: str       # "Left" or "Right" as seen by the user
    frame_id: int
    distance_m: float | None = None  # camera -> hand, metres; None if unknown
    box: tuple[float, float, float, float] | None = None  # hand outline x1,y1,x2,y2

class Direction(str, Enum):
    UP = "U"
    DOWN = "D"
    LEFT = "L"
    RIGHT = "R"
    UP_LEFT = "UL"
    UP_RIGHT = "UR"
    DOWN_LEFT = "DL"
    DOWN_RIGHT = "DR"
    FORWARD = "F"         # aligned; move hand deeper (away from the camera)
    BACK = "B"            # aligned; hand went past the object, pull back
    ARRIVED = "G"         # grab now
    LOST = "X"            # hand not visible
    STOP = "S"            # all motors off

@dataclass(frozen=True)
class GuidanceCommand:
    direction: Direction
    intensity: int        # 0 to 255, stronger = farther from target
    distance_px: float
```

Frames are OpenCV BGR `numpy.ndarray`, shape `(height, width, 3)`. Pixel y
grows downward. Directions are from the user's point of view; if the webcam
faces the user, set `mirror: true` in config and guidance flips left/right
(and forward/back).

Depth: `distance_m` is optional. The detector fills it from the object's known
real size, the hand tracker from MediaPipe's metric palm size, both using
`camera_hfov_deg`. When both are present, guidance aligns the hand in 2D first,
then says `FORWARD`/`BACK` until hand and object depths are within
`guidance.depth_tolerance`, then `ARRIVED`. When either is `None`, guidance is
2D as before. The hand/object distance ratio does not depend on the camera, so
`camera_hfov_deg` only needs calibrating for absolute centimetres.

## Module interfaces

Each owner implements exactly these entry points. Names and signatures are
fixed; internals are up to the owner.

| Module | Entry point | Owner |
| --- | --- | --- |
| `vision/yolo/runtime/detector.py` | `class ObjectDetector(config)` with `detect(frame, target: str) -> list[Detection]` | Harim |
| `vision/yolo/runtime/scan.py` | `scan(frames: Iterable[ndarray], detector, target: str, timeout_s: float) -> Detection \| None` | Harim |
| `vision/yolo/model/classes.py` | `normalize(spoken: str) -> str \| None` (maps "coke", "soda can" to "can"; None if unsupported) and `SUPPORTED: list[str]` | Rizwan |
| `vision/yolo/model/weights.py` | `weights_path(config) -> Path` (where the agreed weights live locally) | Rizwan |
| `vision/hands/tracker.py` | `class HandTracker(config)` with `track(frame) -> HandPosition \| None` | Haris |
| `guidance/compute.py` | `compute(det: Detection, hand: HandPosition \| None, frame_shape, config) -> GuidanceCommand` (pure function, no I/O) | Haris |
| `hardware/host/band.py` | `class Band(config)` with `send(cmd: GuidanceCommand)` and `close()`; plus `class MockBand` that prints commands | Azfar |
| `app/main.py` | `python -m app.main` runs the full loop; `--mock-band`, `--video <file>` flags | Harim |

Only `app/` opens the camera and owns the main loop. Other modules receive
frames as arguments; their own `demo.py` may open a camera for testing.

## Serial protocol (laptop to Arduino)

- 115200 baud, ASCII, one command per line ending in `\n`.
- Format: `<DIR>,<INTENSITY>\n`, e.g. `L,180\n`, `G,255\n`, `S,0\n`.
- `<DIR>` is a `Direction` value. `<INTENSITY>` is 0 to 255.
- `F` / `B` (forward / back) are distinct patterns, not a single motor: e.g.
  all four motors pulsing, faster as the hand gets closer (`F`), and one long
  buzz on all four (`B`). Firmware that doesn't know them yet replies `ERR`.
- Arduino replies `OK\n` to each valid line and `ERR\n` otherwise (host may
  ignore replies).
- Safety watchdog: if the Arduino gets no line for 500 ms it turns all motors
  off. The host should resend the current command at least every 200 ms.
- Host sends at most 30 commands per second.

## Config

- `config.example.yaml` is committed (in `shared/`) with every key and a
  comment. Each person copies it to `config.yaml` at the repo root and fills
  in their machine's values. `config.yaml` is gitignored.
- Load it only through `shared/config.py: load_config() -> dict`.
- Keys: `camera_index`, `mirror`, `serial_port` (e.g. `COM3` or
  `/dev/tty.usbmodem1101`), `baud`, `wrist` (`left`/`right`), `yolo.weights`,
  `yolo.confidence` (default 0.5), `yolo.confirm_frames` (default 3),
  `scan_timeout_s` (default 5), `camera_hfov_deg` (default 70),
  `hand_point` (`palm` default, or `index_tip`),
  `guidance.arrive_px`, `guidance.dead_zone_px`, `guidance.depth_tolerance`
  (default 0.15).
- New keys: add them to `config.example.yaml` via a shared PR (rule 3), and give
  every new key a default in code so older configs still run.

## Environment

- Python 3.11. Create `.venv` in the repo root; it is gitignored.
- Root `requirements.txt` only contains `-r` lines pointing at each folder's
  `requirements.txt`. Install with `pip install -r requirements.txt`.
- Arduino code builds with Arduino IDE 2 or `arduino-cli`; board and port are
  chosen locally, never committed.
- Line endings are normalized by `.gitattributes` (`* text=auto eol=lf`).

## Branching and the final merge

1. **Kickoff (first 30 min):** one person creates `main` with `shared/`,
   `.gitignore`, `.gitattributes`, `config.example.yaml`, empty folders with
   `__init__.py`, and these AGENTS.md files. Everyone clones and branches
   from that commit.
2. **During the build:** work only on your branch. Pull `main` into your
   branch whenever a contract PR lands. Push at least every hour.
3. **Final merge**, in this order, into an `integration` branch:
   `feat/azfar`, `feat/rizwan`, `feat/haris`, `feat/harim`. After each
   merge, run `pytest` and
   `python -m app.main --mock-band --video shared/fixtures/demo.mp4`.
4. When `integration` works with the real band, merge it into `main`.

Because each branch only touches its own folder, merges should have no
conflicts. A conflict means someone broke rule 2; resolve it with the owner of
that folder, not by picking a side blindly.

## Definition of done (every module)

- Entry point matches the table above exactly.
- `python -m <module>.demo` runs on its owner's laptop.
- `pytest <folder>` passes without a camera, Arduino, or internet.
- Folder `README.md` says how to run it in under 10 lines.
