from types import SimpleNamespace

import numpy as np
import pytest

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


def test_distance_from_box_height():
    import math

    from vision.yolo.runtime.detector import focal_px, object_distance

    sizes = {"can": (0.122, "height"), "mouse": (0.11, "longest")}
    f = focal_px(640, 70)
    assert f == pytest.approx(320 / math.tan(math.radians(35)))
    # A can 100 px tall at 640 px wide, 70 deg: f * 0.122 / 100
    assert object_distance("can", 50, 100, 640, 70, sizes) == pytest.approx(f * 0.122 / 100)
    # Twice as tall -> half the distance
    assert object_distance("can", 50, 200, 640, 70, sizes) == pytest.approx(f * 0.122 / 200)
    # Flat objects: no distance (box size is unreliable when they lie, tilt, or are held)
    assert object_distance("mouse", 110, 40, 640, 70, sizes) is None
    assert object_distance("banana", 50, 100, 640, 70, sizes) is None
    assert object_distance("can", 50, 0, 640, 70, sizes) is None


def test_wider_fov_means_closer():
    from vision.yolo.runtime.detector import object_distance

    sizes = {"can": (0.122, "height")}
    assert object_distance("can", 0, 100, 640, 90, sizes) < object_distance(
        "can", 0, 100, 640, 60, sizes
    )


def test_detections_carry_distance(can_frame):
    dets = make([(20, 20, 40, 60, 0.9, 0)]).detect(can_frame, "can")
    assert dets[0].distance_m is not None and dets[0].distance_m > 0
