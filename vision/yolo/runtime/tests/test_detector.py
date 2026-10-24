from types import SimpleNamespace

import numpy as np

from shared.config import DEFAULTS
from vision.yolo.runtime.detector import ObjectDetector

NAMES = {0: "can", 1: "bottle", 2: "cup"}


class FakeModel:
    """Returns ultralytics-shaped results with fixed boxes."""

    def __init__(self, boxes):
        self.boxes = boxes  # list of (x1, y1, x2, y2, conf, cls)
        self.kwargs = None

    def predict(self, frame, **kwargs):
        self.kwargs = kwargs
        if not self.boxes:
            return [SimpleNamespace(names=NAMES, boxes=[])]
        arr = np.array(self.boxes, dtype=float)
        return [SimpleNamespace(names=NAMES, boxes=_Boxes(arr))]


class _Boxes:
    def __init__(self, arr):
        self.xyxy, self.conf, self.cls = arr[:, :4], arr[:, 4], arr[:, 5]

    def __len__(self):
        return len(self.xyxy)


def make(boxes):
    return ObjectDetector(DEFAULTS, model=FakeModel(boxes))


def test_keeps_only_target_sorted_by_confidence(can_frame):
    detector = make(
        [
            (0, 0, 10, 10, 0.6, 0),  # can
            (5, 5, 50, 50, 0.95, 1),  # bottle: wrong label
            (20, 20, 40, 60, 0.9, 0),  # can, best
        ]
    )
    dets = detector.detect(can_frame, "can")
    assert [d.confidence for d in dets] == [0.9, 0.6]
    assert dets[0].center == (30, 40)
    assert all(d.label == "can" for d in dets)


def test_drops_low_confidence(can_frame):
    detector = make([(0, 0, 10, 10, 0.3, 0)])
    assert detector.detect(can_frame, "can") == []


def test_no_boxes(empty_frame):
    assert make([]).detect(empty_frame, "can") == []


def test_target_is_case_insensitive(can_frame):
    assert len(make([(0, 0, 10, 10, 0.8, 0)]).detect(can_frame, " Can ")) == 1


def test_frame_id_increments(can_frame):
    detector = make([(0, 0, 10, 10, 0.8, 0)])
    ids = [detector.detect(can_frame, "can")[0].frame_id for _ in range(3)]
    assert ids == [0, 1, 2]


def test_passes_confidence_and_size_to_model(can_frame):
    model = FakeModel([])
    ObjectDetector(DEFAULTS, model=model).detect(can_frame, "can")
    assert model.kwargs["conf"] == DEFAULTS["yolo"]["confidence"]
    assert model.kwargs["imgsz"] == 640
