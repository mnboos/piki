import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from django.test import SimpleTestCase
from trackforge import OCSORT

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


def _make_ocsort() -> OCSORT:
    return OCSORT(max_age=30, min_hits=1, iou_threshold=0.3, delta_t=3, inertia=0.2)


class OCSortContractTests(SimpleTestCase):
    """Guard the OC-Sort behaviors stream.py's tracker block relies on."""

    def test_return_format_is_track_id_tlwh_score_class(self):
        t = _make_ocsort()
        for _ in range(3):
            tracks = t.update([([0.1, 0.1, 0.2, 0.2], 0.9, 0)])
        assert len(tracks) == 1
        tid, tlwh, score, cls_id = tracks[0]
        assert isinstance(tid, int) and tid > 0
        assert len(tlwh) == 4
        assert isinstance(score, float)
        assert isinstance(cls_id, int)

    def test_track_id_stays_stable(self):
        t = _make_ocsort()
        ids = []
        for f in range(5):
            tracks = t.update([([0.10 + f * 0.02, 0.1, 0.2, 0.2], 0.9, 0)])
            ids.append(tracks[0][0] if tracks else None)
        confirmed = [i for i in ids if i is not None]
        assert confirmed and len(set(confirmed)) == 1

    def test_different_classes_track_independently(self):
        t = _make_ocsort()
        for _ in range(4):
            tracks = t.update([
                ([0.30, 0.10, 0.20, 0.20], 0.9, 0),  # class 0 = cat
                ([0.30, 0.10, 0.20, 0.20], 0.9, 1),  # class 1 = dog (overlapping)
            ])
        # Both classes should have separate track IDs
        assert len(tracks) == 2
        ids = sorted(t[0] for t in tracks)
        assert ids[0] != ids[1]

    def test_track_expires_after_max_age(self):
        t = OCSORT(max_age=2, min_hits=1, iou_threshold=0.3, delta_t=3, inertia=0.2)
        for _ in range(3):
            tracks = t.update([([0.1, 0.1, 0.2, 0.2], 0.9, 0)])
        assert len(tracks) == 1
        tid = tracks[0][0]
        # Feed empty detections for max_age+1 frames
        for _ in range(3):
            tracks = t.update([])
        # Track should be gone
        remaining_ids = [t[0] for t in tracks]
        assert tid not in remaining_ids


# ---------------------------------------------------------------------------
# AI pipeline tests — guard against regressions in tile slicing, post-
# processing, and model input-type detection.
# ---------------------------------------------------------------------------

from .utils.func import _slice_nv12_tile, OLD_slice_nv12_tile, slice_roi_into_tiles
from .utils.interfaces import Box

_TILE = 640


def _make_nv12_frame(w: int = 1280, h: int = 640) -> np.ndarray:
    """Build a synthetic NV12 frame (Y plane + interleaved UV plane)."""
    frame = np.zeros((h * 3 // 2, w), dtype=np.uint8)
    # Fill Y with a gradient so tiles aren't blank
    frame[:h] = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    # Fill UV with a constant (grey = UV both 128)
    frame[h:] = 128
    return frame


class Nv12TileTests(SimpleTestCase):
    """Both tile slicers must produce exactly 640×640×3/2 bytes of NV12 data."""

    def test_new_slicer_output_size(self):
        frame = _make_nv12_frame()
        tile = _slice_nv12_tile(nv12=frame, buffer_h=640, tx=0, ty=0, tile_size=_TILE)
        assert tile.size == _TILE * _TILE * 3 // 2, f"got {tile.size}, want {_TILE * _TILE * 3 // 2}"
        assert tile.dtype == np.uint8

    def test_old_slicer_output_size(self):
        frame = _make_nv12_frame()
        tile = OLD_slice_nv12_tile(nv12=frame, buffer_h=640, tx=320, ty=0, tile_size=_TILE)
        assert tile.size == _TILE * _TILE * 3 // 2, f"got {tile.size}, want {_TILE * _TILE * 3 // 2}"

    def test_new_slicer_preserves_y_data(self):
        """The first _TILE×_TILE bytes must match the Y-plane slice."""
        frame = _make_nv12_frame()
        tile = _slice_nv12_tile(nv12=frame, buffer_h=640, tx=100, ty=50, tile_size=_TILE)
        y = tile[:_TILE * _TILE].reshape(_TILE, _TILE)
        expected = frame[50:50 + _TILE, 100:100 + _TILE]
        # Y-plane shape may be smaller than 640 — padded rows must be zero
        h_actual = min(_TILE, frame.shape[0] - 50)
        assert (y[:min(_TILE, 640 - 50), :640] == expected[:min(_TILE, 640 - 50)]).all()
        if h_actual < _TILE:
            assert (y[h_actual:] == 0).all()

    def test_old_slicer_preserves_y_data(self):
        frame = _make_nv12_frame()
        tile = OLD_slice_nv12_tile(nv12=frame, buffer_h=640, tx=0, ty=0, tile_size=_TILE)
        y = tile[:_TILE * _TILE].reshape(_TILE, _TILE)
        expected = frame[:min(_TILE, 640), :_TILE]
        assert (y[:expected.shape[0]] == expected).all()

    def test_new_slicer_zeroes_before_fill(self):
        """Rows beyond the NV12 frame height are zero-padded."""
        frame = _make_nv12_frame(h=352)  # NV12 shape: (528, 1280)
        tile = _slice_nv12_tile(nv12=frame, buffer_h=352, tx=0, ty=0, tile_size=_TILE)
        y = tile[:_TILE * _TILE].reshape(_TILE, _TILE)
        # NV12 frame has 528 total rows; Y slice captures all of them.
        # Only rows >= 528 (beyond NV12 height) are guaranteed zero.
        assert (y[528:] == 0).all(), (
            f"padding rows must be zero; non-zero at {(y[528:] != 0).sum()} pixels"
        )

    def test_both_slicers_produce_identical_output(self):
        """For a full-size tile, old and new must agree byte-for-byte."""
        frame = _make_nv12_frame(w=1920, h=1080)
        t_new = _slice_nv12_tile(nv12=frame, buffer_h=1080, tx=0, ty=0, tile_size=_TILE)
        t_old = OLD_slice_nv12_tile(nv12=frame, buffer_h=1080, tx=0, ty=0, tile_size=_TILE)
        assert t_new.size == t_old.size
        assert (t_new == t_old).all(), (
            f"mismatch at {(t_new != t_old).sum()} bytes"
        )


class SliceRoiIntoTilesTests(SimpleTestCase):
    """slice_roi_into_tiles must scale ROIs from lores→hires and produce valid tiles."""

    def test_single_roi_produces_one_tile(self):
        frame = _make_nv12_frame(w=1920, h=1080)
        # lores ROI (preview_downscale_factor=2 → hires (200, 100, 300, 200))
        tiles = slice_roi_into_tiles(
            frame=frame, rois=[Box(100, 50, 150, 100)],
            tile_size=_TILE, preview_downscale_factor=2, model_input_type="NV12",
        )
        assert len(tiles) == 1

    def test_roi_near_edge_clamps_to_frame(self):
        """ROI at (0,0) with any size must produce a tile starting at (0,0)."""
        frame = _make_nv12_frame(w=1280, h=640)
        tiles = slice_roi_into_tiles(
            frame=frame, rois=[Box(0, 0, 10, 10)],
            tile_size=_TILE, preview_downscale_factor=2, model_input_type="NV12",
        )
        assert len(tiles) == 1
        _, tx, ty = tiles[0]
        assert tx == 0 and ty == 0, f"edge ROI must clamp to origin, got ({tx},{ty})"

    def test_every_tile_is_correct_nv12_size(self):
        """Every tile returned must be 614400 bytes regardless of frame size."""
        for w, h in [(1920, 1080), (1280, 640), (640, 352)]:
            frame = _make_nv12_frame(w=w, h=h)
            tiles = slice_roi_into_tiles(
                frame=frame, rois=[Box(10, 10, 100, 80)],
                tile_size=_TILE, preview_downscale_factor=2, model_input_type="NV12",
            )
            for tile_img, _, _ in tiles:
                assert tile_img.size == _TILE * _TILE * 3 // 2, (
                    f"frame {w}x{h}: got {tile_img.size}"
                )
                assert tile_img.dtype == np.uint8


# ---------------------------------------------------------------------------
# Detection post-processing tests
# ---------------------------------------------------------------------------

from .utils.ai import _sigmoid, _filter_classification, _decode_ltrb_boxes, _nms_per_class

IMG_SIZE = 640  # matches ai.IMG_SIZE


def _make_cls_output(h: int, w: int, num_classes: int = 80, *,
                     positive_at: list[tuple[int, int, float]] = None,
                     ) -> np.ndarray:
    """Synthetic classification output with controlled logit values.

    By default all logits are -10 (conf ≈ 4.5e-5).  Pass ``positive_at`` as
    ``[(row, col, raw_logit), ...]`` to insert a peak at a specific grid cell.
    """
    cls = np.full((h, w, num_classes), -10.0, dtype=np.float32)
    if positive_at:
        for r, c, val in positive_at:
            cls[r, c, 0] = val  # class 0
    return cls


def _make_box_output(h: int, w: int) -> np.ndarray:
    """Synthetic LTRB output — all deltas = 1.0 in grid space."""
    return np.full((h, w, 4), 1.0, dtype=np.float32)


class PostProcessTests(SimpleTestCase):

    def test_sigmoid_extremes(self):
        assert abs(_sigmoid(np.array(0.0)) - 0.5) < 1e-6
        assert _sigmoid(np.array(-10.0)) < 0.001
        assert _sigmoid(np.array(10.0)) > 0.999

    def test_filter_classification_no_peaks_returns_empty(self):
        cls = _make_cls_output(20, 20)
        scores, ids, valid = _filter_classification(cls, conf_thres_raw=-1.1)
        assert scores.size == 0
        assert ids.size == 0
        assert valid.size == 0

    def test_filter_classification_peak_passes_threshold(self):
        """A single cell with logit 5.0 (sigmoid ≈ 0.993) must pass."""
        cls = _make_cls_output(20, 20, positive_at=[(10, 10, 5.0)])
        scores, ids, valid = _filter_classification(cls, conf_thres_raw=-1.1)
        assert scores.size == 1
        assert ids[0] == 0  # class 0
        assert valid[0] == 10 * 20 + 10  # flat index

    def test_filter_classification_peak_below_threshold_filtered(self):
        """logit -5 (sigmoid ≈ 0.007) must not pass conf_thres_raw=-1.1."""
        cls = _make_cls_output(20, 20, positive_at=[(5, 5, -5.0)])
        scores, ids, valid = _filter_classification(cls, conf_thres_raw=-1.1)
        assert scores.size == 0

    def test_decode_ltrb_boxes_basic(self):
        """At stride=8, grid (0,0): anchor=(4,4), ltrb=1.0 → box=(4-8,4-8,4+8,4+8)=(0,0,12,12)."""
        ltrb = np.full((20, 20, 4), 1.0, dtype=np.float32)
        boxes = _decode_ltrb_boxes(
            flat_indices=np.array([0], dtype=np.int32),
            ltrb=ltrb, stride=8, grid_h=20, grid_w=20,
        )
        assert boxes.shape == (1, 4)
        x1, y1, x2, y2 = boxes[0]
        # anchor at (0.5*8, 0.5*8) = (4, 4); delta=1*8=8
        assert x1 == 0, f"x1={x1}"
        assert y1 == 0, f"y1={y1}"
        assert x2 == 12, f"x2={x2}"
        assert y2 == 12, f"y2={y2}"

    def test_decode_ltrb_boxes_clamps_to_img_bounds(self):
        """Negative deltas must not produce coordinates outside [0, IMG_SIZE]."""
        ltrb = np.full((20, 20, 4), -2.0, dtype=np.float32)  # negative → x1,y1 would go past 0
        boxes = _decode_ltrb_boxes(
            flat_indices=np.array([0], dtype=np.int32),
            ltrb=ltrb, stride=8, grid_h=20, grid_w=20,
        )
        assert (boxes >= 0).all()
        assert (boxes <= IMG_SIZE).all()

    def test_nms_keeps_higher_score_detection(self):
        """Two highly overlapping boxes — only the higher-score one survives."""
        boxes = np.array([[10, 10, 50, 50], [12, 12, 48, 48]], dtype=np.float32)
        scores = np.array([0.6, 0.9], dtype=np.float32)
        cls_ids = np.array([0, 0], dtype=np.int32)
        kept = _nms_per_class(boxes, scores, cls_ids, iou_thres=0.45)
        assert kept == [1]  # index 1 has higher score

    def test_nms_keeps_separated_boxes(self):
        """Two non-overlapping boxes — both survive."""
        boxes = np.array([[0, 0, 50, 50], [200, 200, 250, 250]], dtype=np.float32)
        scores = np.array([0.7, 0.8], dtype=np.float32)
        cls_ids = np.array([0, 0], dtype=np.int32)
        kept = _nms_per_class(boxes, scores, cls_ids, iou_thres=0.45)
        assert sorted(kept) == [0, 1]

    def test_nms_different_classes_both_kept(self):
        """Same-location boxes of different classes must both survive."""
        boxes = np.array([[10, 10, 50, 50], [10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.8, 0.7], dtype=np.float32)
        cls_ids = np.array([0, 1], dtype=np.int32)
        kept = _nms_per_class(boxes, scores, cls_ids, iou_thres=0.45)
        assert sorted(kept) == [0, 1]


# ---------------------------------------------------------------------------
# End-to-end pipeline test (mocked HBM runtime).
# Only runs when rclpy + stream deps are available (typically on the device).
# ---------------------------------------------------------------------------

try:
    from .utils.stream import run_object_detection  # noqa: F811
    from .utils.shared import Detection, InferenceOutput
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False


@unittest.skipUnless(_PIPELINE_AVAILABLE, "rclpy / stream deps not available")
class PipelineIntegrationTests(SimpleTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.frame = _make_nv12_frame(w=1280, h=640)
        cls.frame[100:180, 200:300] = 220  # bright patch at (200,100)

    @staticmethod
    def _mock_detection(tile_img, **_kw):
        return 15, [("person", 0.85, np.array([300, 280, 340, 360], dtype=np.float32))]

    @patch("core.utils.stream.detect_objects")
    def test_pipeline_produces_normalized_coords(self, mock_detect):
        mock_detect.side_effect = self._mock_detection

        rois = [Box(0, 0, 640, 320)]
        result = run_object_detection(frame_hires=self.frame, rois=rois, timestamp=0)

        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.label == "person"
        assert abs(det.confidence - 0.85) < 1e-6
        # mock returns tile-pixel (300,280,340,360); tile at (0,0); frame=1280×640
        ymin, xmin, ymax, xmax = det.bbox
        assert abs(ymin - 300 / 640) < 0.01
        assert abs(xmin - 280 / 1280) < 0.01
        assert abs(ymax - 360 / 640) < 0.01
        assert abs(xmax - 340 / 1280) < 0.01

    @patch("core.utils.stream.detect_objects")
    def test_pipeline_handles_empty_detections(self, mock_detect):
        mock_detect.return_value = (10, [])
        result = run_object_detection(
            frame_hires=self.frame, rois=[Box(10, 10, 100, 80)], timestamp=0,
        )
        assert len(result.detections) == 0


# ---------------------------------------------------------------------------
# MODEL_INPUT_TYPE detection — the bug that broke everything.
# ---------------------------------------------------------------------------

class ModelInputTypeTests(SimpleTestCase):
    """Verify the NV12-vs-BGR fallback logic used at startup."""

    def _detect_input_type(self, input_type_attr: str, model_filename: str) -> str:
        """Replicate the startup detection logic from ai.py (module level)."""
        _input_type = str(input_type_attr) if input_type_attr else ""
        model_file_str = str(model_filename).lower()
        if "NV12" in _input_type.upper() or "YUV" in _input_type.upper() or "nv12" in model_file_str:
            return "NV12"
        return "BGR"

    def test_runtime_reports_nv12(self):
        assert "NV12" == self._detect_input_type("NV12_TENSOR", "model.bin")

    def test_runtime_reports_yuv(self):
        assert "NV12" == self._detect_input_type("YUV420", "model.bin")

    def test_runtime_silent_filename_has_nv12(self):
        assert "NV12" == self._detect_input_type("", "yolo26n_detect_bayese_640x640_nv12.bin")

    def test_runtime_silent_filename_lacks_nv12(self):
        assert "BGR" == self._detect_input_type("", "yolo26n_detect_bayese_640x640.bin")

    def test_runtime_reports_bgr(self):
        """If the runtime says BGR and filename has no nv12, we must get BGR."""
        assert "BGR" == self._detect_input_type("BGR_PLANAR", "model.bin")

    def test_filename_nv12_overrides_missing_runtime_attr(self):
        """The real-world scenario: HBM runtime doesn't expose input_type, but
        the model filename confirms NV12."""
        assert "NV12" == self._detect_input_type(
            "", "yolo26n_detect_bayese_640x640_nv12.bin"
        )


# ---------------------------------------------------------------------------
# Real-image smoke tests — require the HBM runtime (device only).
# ---------------------------------------------------------------------------

_IMG_DIR = Path(__file__).resolve().parent.parent.parent  # repo root

try:
    from .utils.ai import detect_objects  # noqa: F811
    _MODEL_AVAILABLE = True
except (ImportError, OSError):
    _MODEL_AVAILABLE = False


def _bgr_to_nv12_640(bgr: np.ndarray) -> np.ndarray:
    """Resize a BGR image to 640×640 and convert to flat NV12."""
    resized = cv2.resize(bgr, (640, 640))
    yuv = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
    # I420 → NV12: take Y plane + interleaved UV (drop the separate V,U planes)
    y = yuv[:640].flatten()
    u = yuv[640:960, :320].flatten()   # U samples (quarter res, first half-width)
    v = yuv[640:960, 320:].flatten()   # V samples (quarter res, second half-width)
    uv = np.empty(u.size + v.size, dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v
    return np.concatenate([y, uv])


@unittest.skipUnless(_MODEL_AVAILABLE, "HBM runtime / detect_objects not available")
class RealImageSmokeTests(SimpleTestCase):
    """Verify the live model detects expected classes in real images."""

    def _run_on_image(self, path: str, expected_label: str, min_conf: float = 0.3):
        bgr = cv2.imread(path)
        assert bgr is not None, f"failed to load {path}"
        nv12 = _bgr_to_nv12_640(bgr)
        assert nv12.size == 640 * 640 * 3 // 2

        _duration_ms, _bpu_ms, detections = detect_objects(nv12)

        labels_found = {d[0] for d in detections if d[1] >= min_conf}
        assert expected_label in labels_found, (
            f"{Path(path).name}: expected '{expected_label}' at conf≥{min_conf}, "
            f"got {[(d[0], round(float(d[1]), 2)) for d in detections]}"
        )

    def test_cat_detected(self):
        self._run_on_image(str(_IMG_DIR / "cat.jpg"), "cat")

    def test_dog_detected(self):
        self._run_on_image(str(_IMG_DIR / "dog.jpg"), "dog")
