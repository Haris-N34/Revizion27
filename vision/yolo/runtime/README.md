# vision/yolo/runtime (Harim)

`ObjectDetector(config).detect(frame, target)` and `scan(frames, detector, target, timeout_s)`.

1. `pip install -r vision/yolo/runtime/requirements.txt`
2. Demo: `python -m vision.yolo.runtime.demo --target can` (`--video FILE`, `--timeout 8`, `--no-window`)
3. Tests (no camera, no YOLO): `pytest vision/yolo/runtime`

Weights come from `model.weights.weights_path(config)`. Until that exists, or if the file
is missing, it falls back to YOLO-World (`yolov8s-worldv2.pt`, auto-downloaded) with
custom classes, since "can" is not a COCO class.
