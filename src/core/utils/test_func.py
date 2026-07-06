"""Offline tests for cross-tile duplicate merging + single-frame NV12 tiling.

Verifies core.utils.func against on-device data, no device deps. Runnable on any
platform: `uv run pytest src/core/utils/test_func.py`.
"""
import numpy as np

from core.utils.func import build_full_frame_tile, merge_cross_tile_duplicates
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


# --- single-frame NV12 downscale (build_full_frame_tile) ---------------------

_MODEL = 640


def _make_nv12(frame_h: int, frame_w: int) -> np.ndarray:
    """Synthetic NV12 with a chroma pattern that catches U/V mixing.

    Y[y,x]      = x & 0xFF               (column gradient)
    U[r,c]      = (c * 3) & 0xFF         (encodes chroma column -> catches wrong
                                          column subsampling)
    V[r,c]      = 200                    (constant -> catches U/V swap)
    """
    y = np.tile((np.arange(frame_w, dtype=np.uint8) & 0xFF), (frame_h, 1))
    uv = np.zeros((frame_h // 2, frame_w // 2, 2), dtype=np.uint8)
    uv[:, :, 0] = (np.arange(frame_w // 2, dtype=np.int32) * 3 & 0xFF).astype(np.uint8)
    uv[:, :, 1] = 200
    return np.vstack([y, uv.reshape(frame_h // 2, frame_w)])


def test_build_full_frame_tile_downscales_y_uv_correctly():
    frame_h, frame_w, ds = 640, 1280, 2
    nv12 = _make_nv12(frame_h, frame_w)
    tile = build_full_frame_tile(nv12=nv12, frame_h=frame_h, tile_size=_MODEL, downscale=ds)

    t2d = np.asarray(tile).reshape(_MODEL * 3 // 2, _MODEL)
    y_plane = t2d[:_MODEL]              # 640 x 640
    uv_plane = t2d[_MODEL:]            # 320 x 640
    ch, cw = frame_h // ds, frame_w // ds   # content 320 x 640

    # Y: content = every 2nd source pixel; padding zeroed.
    assert np.array_equal(y_plane[:ch, :cw], nv12[0:frame_h:ds, 0:frame_w:ds])
    assert np.count_nonzero(y_plane[ch:]) == 0

    # UV: even cols carry U (the c*3 pattern from every 2nd chroma column),
    # odd cols carry V (=200). This fails loudly if U/V are swapped or columns
    # are subsampled wrong.
    uv_content = uv_plane[: ch // 2, :cw]          # 160 x 640 interleaved
    out_cols = cw // 2                              # 320 chroma pairs
    expected_u = ((np.arange(out_cols) * ds) * 3 & 0xFF).astype(np.uint8)
    assert np.array_equal(uv_content[:, 0::2], np.tile(expected_u, (ch // 2, 1)))
    assert np.all(uv_content[:, 1::2] == 200)
    assert np.count_nonzero(uv_plane[ch // 2:]) == 0


def test_build_full_frame_tile_survives_cv2_nv12_conversion():
    import cv2
    nv12 = _make_nv12(640, 1280)
    tile = build_full_frame_tile(nv12=nv12, frame_h=640, tile_size=_MODEL, downscale=2)
    t2d = np.asarray(tile).reshape(_MODEL * 3 // 2, _MODEL)
    bgr = cv2.cvtColor(t2d, cv2.COLOR_YUV2BGR_NV12)   # must not raise
    assert bgr.shape == (_MODEL, _MODEL, 3)
    # Content carries the Y gradient (varies across columns). Padding rows are a
    # single uniform colour — NV12 zero-pad is non-neutral chroma (green), not
    # black; the native tile slicer zero-pads identically.
    assert bgr[:320].std() > 0
    assert np.unique(bgr[400].reshape(-1, 3), axis=0).shape[0] == 1
