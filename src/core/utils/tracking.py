"""SORT-style IoU + Kalman tracker.

Built on top of ``cv2.KalmanFilter`` (already part of the project's OpenCV
distribution) and operates entirely in normalized ``[0, 1]`` bbox space so
it's agnostic to camera resolution.

Each track holds a constant-velocity Kalman filter on the bbox centre and a
constant (zero-velocity) Kalman pass on the bbox dimensions.  Predictions
fill detection gaps so a brief YOLO miss doesn't drop the box on screen or
the servo aim.

Same-label greedy IoU matching: simpler than ByteTrack but tuned for the
single-camera, low-traffic deployment.  ByteTrack would need a second
post-NMS pass below the model's confidence floor, which is invasive — see
``docs/feature-plan.md``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .shared import cv2

logger = logging.getLogger(__name__)


def _iou(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """IoU between two normalized [ymin, xmin, ymax, xmax] bboxes."""
    ay1, ax1, ay2, ax2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    by1, bx1, by2, bx2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    inter_y1 = max(ay1, by1)
    inter_x1 = max(ax1, bx1)
    inter_y2 = min(ay2, by2)
    inter_x2 = min(ax2, bx2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0


def _bbox_to_meas(bbox: list[float]) -> np.ndarray:
    """Convert [ymin, xmin, ymax, xmax] → measurement [cx, cy, w, h]."""
    ymin, xmin, ymax, xmax = bbox
    return np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5,
                     max(1e-6, xmax - xmin), max(1e-6, ymax - ymin)], dtype=np.float32)


def _state_to_bbox(state: np.ndarray) -> list[float]:
    """Convert state [cx, cy, w, h, dcx, dcy] → bbox [ymin, xmin, ymax, xmax]."""
    cx, cy, w, h = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    w = max(1e-4, w)
    h = max(1e-4, h)
    return [
        max(0.0, cy - h * 0.5),
        max(0.0, cx - w * 0.5),
        min(1.0, cy + h * 0.5),
        min(1.0, cx + w * 0.5),
    ]


def _make_kalman(meas: np.ndarray) -> cv2.KalmanFilter:
    """Six-state (cx, cy, w, h, dcx, dcy), four-measurement (cx, cy, w, h)."""
    kf = cv2.KalmanFilter(6, 4, 0, cv2.CV_32F)
    # Δt = 1 frame; position += velocity each tick.
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ], dtype=np.float32)
    # Modest process noise on size; higher on centre velocity.
    kf.processNoiseCov = np.diag(np.array(
        [1e-3, 1e-3, 1e-4, 1e-4, 1e-2, 1e-2], dtype=np.float32))
    kf.measurementNoiseCov = np.diag(np.array(
        [5e-3, 5e-3, 5e-3, 5e-3], dtype=np.float32))
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
    kf.statePost = np.array([meas[0], meas[1], meas[2], meas[3], 0.0, 0.0],
                            dtype=np.float32).reshape(-1, 1)
    return kf


@dataclass
class Track:
    id: int
    label: str
    bbox_norm: list[float]
    kalman: cv2.KalmanFilter = field(repr=False)
    hits: int = 1
    misses: int = 0
    age_frames: int = 1
    last_confidence: float = 0.0
    # The Kalman-predicted bbox for the *current* frame.  Computed by
    # ``IouTracker.update`` before matching so external code can use it as
    # the "tentative" position even when no detection arrived.
    predicted_bbox: list[float] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.hits >= self._confirm_hits

    # Filled in by the tracker so .confirmed can check without holding state.
    _confirm_hits: int = 3


class IouTracker:
    """Single-camera multi-target tracker with Kalman smoothing.

    Designed to be called once per inference completion (``update``) and once
    per motion-gated frame (``tick_idle``) so tracks survive quiet stretches
    without accumulating ``misses``.

    Parameters are dialled in by the caller before each ``update`` so config
    changes take effect on the next frame.
    """

    def __init__(self, *, iou_threshold: float = 0.3,
                 max_misses: int = 10, confirm_hits: int = 3):
        self.iou_threshold = float(iou_threshold)
        self.max_misses = int(max_misses)
        self.confirm_hits = max(1, int(confirm_hits))
        self._tracks: list[Track] = []
        self._next_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.confirmed]

    def tick_idle(self) -> None:
        """Advance time without consuming a detection batch.

        Called on motion-gated frames so a stationary target's track survives
        quiescent periods.  We bump ``age_frames`` and let the Kalman predict,
        but we do *not* increment ``misses`` (no inference ran).
        """
        for t in list(self._tracks):
            t.age_frames += 1
            t.kalman.predict()

    def update(self, detections) -> list[Track]:
        """Match the given detections against current tracks and return them.

        ``detections`` is an iterable of ``(label, confidence, bbox_norm)`` —
        the same shape used elsewhere in the pipeline.  ``bbox_norm`` is
        ``[ymin, xmin, ymax, xmax]`` in [0, 1].  Returns the full ``Track``
        list (callers usually want ``confirmed_tracks``).
        """
        # 1. Predict each existing track's bbox via Kalman.
        for t in self._tracks:
            t._confirm_hits = self.confirm_hits  # keep .confirmed in sync
            predicted_state = t.kalman.predict()
            t.predicted_bbox = _state_to_bbox(predicted_state[:, 0])

        # 2. Greedy IoU matching by descending confidence, same label only.
        det_list = sorted(
            [(label, float(conf), list(bbox)) for label, conf, bbox in detections],
            key=lambda d: d[1], reverse=True,
        )
        unmatched_track_ids: set[int] = {t.id for t in self._tracks}
        matched_track_ids: set[int] = set()
        new_tracks: list[Track] = []

        for label, conf, bbox in det_list:
            best_iou = self.iou_threshold
            best_track: Track | None = None
            label_lc = label.strip().lower()
            for t in self._tracks:
                if t.id in matched_track_ids:
                    continue
                if t.label.strip().lower() != label_lc:
                    continue
                iou = _iou(t.predicted_bbox, bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track = t
            if best_track is not None:
                # Matched — update Kalman + bookkeeping.
                meas = _bbox_to_meas(bbox).reshape(-1, 1)
                best_track.kalman.correct(meas)
                # Use the corrected state for the on-screen bbox (smoother
                # than the raw measurement).
                state = best_track.kalman.statePost[:, 0]
                best_track.bbox_norm = _state_to_bbox(state)
                best_track.hits += 1
                best_track.misses = 0
                best_track.age_frames += 1
                best_track.last_confidence = conf
                matched_track_ids.add(best_track.id)
                unmatched_track_ids.discard(best_track.id)
            else:
                # Spawn a new tentative track.
                meas = _bbox_to_meas(bbox)
                kf = _make_kalman(meas)
                t = Track(
                    id=self._next_id,
                    label=label,
                    bbox_norm=list(bbox),
                    kalman=kf,
                    hits=1,
                    misses=0,
                    age_frames=1,
                    last_confidence=conf,
                    predicted_bbox=list(bbox),
                )
                t._confirm_hits = self.confirm_hits
                self._next_id += 1
                new_tracks.append(t)

        # 3. Bookkeeping for unmatched tracks: bump misses, drop if too stale.
        survivors: list[Track] = []
        for t in self._tracks:
            if t.id in matched_track_ids:
                survivors.append(t)
                continue
            t.misses += 1
            t.age_frames += 1
            # Use the prediction as the displayed bbox while we wait for the
            # next match — smoother than freezing the last observation.
            t.bbox_norm = t.predicted_bbox or t.bbox_norm
            if t.misses <= self.max_misses:
                survivors.append(t)
            else:
                logger.debug("Drop track id=%d label=%s after %d misses", t.id, t.label, t.misses)

        survivors.extend(new_tracks)
        self._tracks = survivors
        return list(self._tracks)
