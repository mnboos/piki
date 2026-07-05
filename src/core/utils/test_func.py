"""Offline tests for cross-tile duplicate merging (no device deps).

Verifies core.utils.func.merge_cross_tile_duplicates against the exact detection
clusters captured on-device (each a single frame-filling person sliced across
overlapping tiles), plus the distinct-object guard. Runnable on any platform:
`uv run pytest src/core/utils/test_func.py`.
"""
from core.utils.func import merge_cross_tile_duplicates
from core.utils.shared import Detection

_FW, _FH = 1280.0, 640.0

# (label, conf, [x,y,w,h]) clusters straight from the device logs; all are ONE person.
_LOGGED_CLUSTERS = [
    [("person", 0.909, [560, 10, 621, 603]), ("person", 0.909, [658, 10, 621, 603]), ("person", 0.909, [214, 10, 621, 603])],
    [("person", 0.899, [344, 0, 640, 637]), ("person", 0.899, [640, 0, 640, 637])],
    [("person", 0.845, [602, 10, 637, 629]), ("person", 0.845, [642, 10, 637, 629])],
    [("person", 0.884, [558, 1, 608, 602]), ("person", 0.884, [666, 1, 608, 602]), ("person", 0.884, [236, 1, 608, 602])],
    [("person", 0.793, [671, 14, 607, 596]), ("person", 0.793, [161, 14, 607, 596])],   # barely overlap
    [("person", 0.789, [581, 17, 612, 608]), ("person", 0.789, [667, 17, 612, 608]), ("person", 0.789, [205, 17, 612, 608])],
    [("person", 0.74, [677, 5, 601, 610]), ("person", 0.74, [167, 5, 601, 610])],        # barely overlap
    [("person", 0.874, [662, 17, 617, 622]), ("person", 0.874, [224, 17, 617, 622])],
    [("person", 0.851, [334, 7, 639, 632]), ("person", 0.851, [640, 7, 639, 632])],
    [("person", 0.875, [254, 6, 639, 626]), ("person", 0.875, [640, 6, 639, 626])],
    [("person", 0.8, [585, 15, 625, 576]), ("person", 0.8, [653, 15, 625, 576]), ("person", 0.8, [159, 15, 625, 576])],
]


def _det(label: str, conf: float, xywh) -> Detection:
    x, y, w, h = xywh
    # normalized [ymin, xmin, ymax, xmax]
    bbox = [y / _FH, x / _FW, (y + h) / _FH, (x + w) / _FW]
    return Detection(label=label, confidence=conf, bbox=bbox,
                     mask_centroid=(0.5, 0.5), mask_polygon=None)


def test_logged_clusters_collapse_to_single_detection():
    for idx, cluster in enumerate(_LOGGED_CLUSTERS):
        dets = [_det(*c) for c in cluster]
        merged = merge_cross_tile_duplicates(dets)
        assert len(merged) == 1, f"cluster {idx}: {len(cluster)} slices -> {len(merged)} boxes (want 1)"
        d = merged[0]
        # Union spans the widest x extent of the input slices.
        want_xmin = min(c[2][0] for c in cluster) / _FW
        want_xmax = max(c[2][0] + c[2][2] for c in cluster) / _FW
        assert abs(d.bbox[1] - want_xmin) < 1e-6
        assert abs(d.bbox[3] - want_xmax) < 1e-6
        # Merged aim centre is the union-box centre; polygons dropped.
        assert d.mask_centroid == ((d.bbox[1] + d.bbox[3]) / 2.0, (d.bbox[0] + d.bbox[2]) / 2.0)
        assert d.mask_polygon is None
        assert d.confidence == max(c[1] for c in cluster)


def test_separated_objects_are_not_merged():
    # Two people with a real horizontal gap must stay distinct.
    dets = [_det("person", 0.9, [200, 100, 250, 450]), _det("person", 0.9, [800, 110, 240, 440])]
    assert len(merge_cross_tile_duplicates(dets)) == 2


def test_different_classes_never_merge():
    # Overlapping boxes of different classes stay separate.
    dets = [_det("person", 0.9, [300, 100, 600, 500]), _det("dog", 0.8, [320, 120, 560, 460])]
    assert len(merge_cross_tile_duplicates(dets)) == 2


def test_single_detection_is_passed_through_unchanged():
    d = _det("cat", 0.7, [400, 200, 300, 300])
    out = merge_cross_tile_duplicates([d])
    assert out == [d]
    # Singleton keeps its real mask centroid (precise aim preserved).
    assert out[0].mask_centroid == (0.5, 0.5)
