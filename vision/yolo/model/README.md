# vision/yolo/model (Rizwan)

Approach: **YOLO-World** (pretrained, open-vocabulary) with `SUPPORTED` baked in as the
class list. No training. "can" is not a COCO class, so stock YOLO can't find it.

1. Get the weights: `python -m vision.yolo.model.export` (builds `models/octaby.pt`,
   ~26 MB, gitignored). Loads offline afterwards; no CLIP download at runtime.
2. Labels: `SUPPORTED` in `classes.py`; `normalize("coke") -> "can"`, unsupported -> `None`.
3. Tests: `pytest vision/yolo/model`
4. Add a class: append to `SUPPORTED` (+ aliases, + a `PROMPTS` phrase if the bare word
   scores low, e.g. "can" -> "drink can"), re-run export. Fine-tune only if
   YOLO-World misses real objects.

If export hits `CERTIFICATE_VERIFY_FAILED` (python.org Python on macOS), run
`/Applications/Python 3.11/Install Certificates.command` once.
The saved file embeds a timestamp, so its SHA-256 differs per export; that is expected.
