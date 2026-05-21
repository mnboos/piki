import numpy as np
import norfair
from django.test import SimpleTestCase

from .utils.interfaces import Box
from .utils.tracker_safety import is_bbox_safe_for_update, required_update_margin, sanitize_bbox_for_frame

_LARGE_MARGIN_EXPECTED = 8
_SMALL_MARGIN_EXPECTED = 2
_CONFIRM_DELAY = 3


class TrackerSafetyTests(SimpleTestCase):
    def test_sanitize_bbox_clamps_to_frame(self):
        bbox = sanitize_bbox_for_frame(bbox=(-10, 3, 20, 10), frame_shape=(100, 200))
        assert bbox == Box(0, 3, 10, 10)

    def test_sanitize_bbox_returns_none_for_invalid_sizes(self):
        assert sanitize_bbox_for_frame(bbox=(10, 10, 0, 5), frame_shape=(100, 200)) is None
        assert sanitize_bbox_for_frame(bbox=(10, 10, 5, -1), frame_shape=(100, 200)) is None

    def test_required_update_margin_scales_with_bbox(self):
        assert required_update_margin(bbox=(10, 10, 10, 8)) == _LARGE_MARGIN_EXPECTED
        assert required_update_margin(bbox=(10, 10, 2, 2)) == _SMALL_MARGIN_EXPECTED

    def test_bbox_safe_for_update_requires_edge_margin(self):
        assert not is_bbox_safe_for_update(bbox=(1, 1, 20, 20), frame_shape=(100, 200))
        assert is_bbox_safe_for_update(bbox=(40, 30, 20, 20), frame_shape=(100, 200))


def _box(x: float, label: str, conf: float = 0.9) -> norfair.Detection:
    """A normalized [0,1] bbox detection as stream.py builds it for Norfair."""
    return norfair.Detection(
        points=np.array([[x, 0.1], [x + 0.2, 0.3]], dtype=np.float32),
        scores=np.array([conf, conf], dtype=np.float32),
        label=label,
    )


def _make_tracker() -> norfair.Tracker:
    return norfair.Tracker(
        distance_function="iou",
        distance_threshold=1.0 - 0.3,  # iou_threshold 0.3 → distance 0.7
        hit_counter_max=10,
        initialization_delay=_CONFIRM_DELAY,
    )


class NorfairContractTests(SimpleTestCase):
    """Guard the Norfair behaviors stream.py's tracker block relies on."""

    def test_id_appears_after_delay_then_stays_stable(self):
        t = _make_tracker()
        ids = []
        for f in range(6):
            objs = t.update(detections=[_box(0.10 + f * 0.02, "cat")])
            ids.append(objs[0].id if objs else None)
        assert ids[:_CONFIRM_DELAY] == [None] * _CONFIRM_DELAY  # withheld during delay
        confirmed = [i for i in ids if i is not None]
        assert confirmed and len(set(confirmed)) == 1  # one stable id afterwards

    def test_estimate_is_two_corner_points(self):
        t = _make_tracker()
        for f in range(4):
            objs = t.update(detections=[_box(0.10 + f * 0.02, "cat")])
        est = objs[0].estimate
        assert est.shape == (2, 2)  # [[x1,y1],[x2,y2]] — stream.py rebuilds bbox from this

    def test_matched_this_frame_identity(self):
        # stream.py uses `id(obj.last_detection) in {id(d) for this-frame dets}`
        # to tell a real match from a Kalman-predicted gap fill (strict zones).
        t = _make_tracker()
        for f in range(4):
            d = _box(0.10 + f * 0.02, "cat")
            objs = t.update(detections=[d])
        assert objs[0].last_detection is d                 # matched this frame
        objs = t.update(detections=[])                     # gap → prediction only
        assert objs and objs[0].last_detection is not None
        assert objs[0].last_detection is d                 # stale instance, not this frame's

    def test_same_label_matching(self):
        # A confirmed cat track must not absorb an overlapping dog detection;
        # the dog must spawn its own id (Norfair matches within a label).
        t = _make_tracker()
        for _ in range(4):
            objs = t.update(detections=[_box(0.30, "cat")])
        cat_id = objs[0].id
        dog_id = None
        for _ in range(6):
            objs = t.update(detections=[_box(0.30, "dog")])
            for o in objs:
                if o.last_detection.label == "dog":
                    dog_id = o.id
        assert dog_id is not None
        assert dog_id != cat_id
