from django.test import SimpleTestCase

from .utils.interfaces import Box
from .utils.tracker_safety import is_bbox_safe_for_update, required_update_margin, sanitize_bbox_for_frame

_LARGE_MARGIN_EXPECTED = 8
_SMALL_MARGIN_EXPECTED = 2


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
