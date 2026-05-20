"""Runtime cache for ExclusionZone polygons.

Zones are stored in the DB in normalized [0, 1] coordinates.  At inference
time we need three things, fast:

  1. A rasterised "allowed" mask matching the current motion-detection frame
     shape, used to zero out motion inside excluded polygons before ROIs are
     created.
  2. A point-in-any-polygon test in normalized space, used by the on_done()
     filter and the engine aim clamp.
  3. The raw polygon list for server-side drawing.

A monotonic ``generation`` counter is bumped on every CRUD mutation; the
inference loop checks it once per frame and rebuilds the rasterised mask
only when stale.  All public functions are safe to call from any thread.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from .shared import cv2

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class _ZoneSnapshot:
    """Immutable view of enabled zones at a given generation."""

    __slots__ = ("generation", "polygons_norm")

    def __init__(self, generation: int, polygons_norm: list[np.ndarray]):
        self.generation = generation
        # Each entry is an (N, 2) float32 array of normalized [x, y] points.
        self.polygons_norm: list[np.ndarray] = polygons_norm


_lock = threading.Lock()
_generation: int = 0
# Sentinel generation -1 forces the first call to _load_snapshot_if_stale() to
# read from the DB.  Without this, both _generation and _snapshot.generation
# start at 0, the staleness check passes immediately, and the inference loop
# never sees existing zones until the user triggers a CRUD (which is the only
# thing that calls bump_generation).  Restart the server with zones already in
# the DB → zones silently ignored.
_snapshot: _ZoneSnapshot = _ZoneSnapshot(generation=-1, polygons_norm=[])
# Cache of the most recently rasterised allowed-mask, keyed by frame shape.
_raster_cache: dict[tuple[int, int], tuple[int, "NDArray[np.uint8]"]] = {}


def bump_generation() -> None:
    """Mark the cached snapshot stale.  Call after any zone CRUD mutation."""
    global _generation
    with _lock:
        _generation += 1
        # Drop rasterised masks — they reference the old polygons.
        _raster_cache.clear()


def _load_snapshot_if_stale() -> _ZoneSnapshot:
    """Refresh the in-memory zone list from the DB if the generation moved."""
    global _snapshot
    with _lock:
        current_gen = _generation
        if _snapshot.generation == current_gen:
            return _snapshot

    # Reload outside the lock so the DB query doesn't block other readers.
    try:
        from ..models import ExclusionZone  # noqa: PLC0415
        rows = list(ExclusionZone.objects.filter(enabled=True).values_list("points", flat=True))
    except Exception:
        logger.exception("Failed to load ExclusionZone rows; treating as empty.")
        rows = []

    polygons: list[np.ndarray] = []
    for points in rows:
        if not isinstance(points, list) or len(points) < 3:
            continue
        try:
            arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        except (ValueError, TypeError):
            continue
        if arr.shape[0] >= 3:
            polygons.append(arr)

    with _lock:
        # Another thread may have bumped the generation while we were loading.
        # That's OK — they will re-trigger this refresh on their next call.
        _snapshot = _ZoneSnapshot(generation=current_gen, polygons_norm=polygons)
        return _snapshot


def polygons_norm() -> list[np.ndarray]:
    """Return the current list of enabled-zone polygons in normalized space.

    Each polygon is an (N, 2) float32 array.  Returns a fresh list (caller is
    free to iterate without locking).
    """
    snap = _load_snapshot_if_stale()
    return list(snap.polygons_norm)


def has_zones() -> bool:
    """Quick "is there anything to do?" check used to short-circuit hot paths."""
    snap = _load_snapshot_if_stale()
    return bool(snap.polygons_norm)


def allowed_mask_for(shape: tuple[int, int]) -> "NDArray[np.uint8] | None":
    """Return a 2-D uint8 mask where excluded pixels are 0 and the rest are 1.

    ``shape`` is ``(height, width)`` of the target frame (typically the lores
    motion frame).  Returns ``None`` when there are no active zones — callers
    should skip the AND in that case.  The mask is rasterised on first request
    per ``(shape, generation)`` and cached for subsequent frames.
    """
    snap = _load_snapshot_if_stale()
    if not snap.polygons_norm:
        return None

    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return None

    key = (h, w)
    with _lock:
        cached = _raster_cache.get(key)
        if cached is not None and cached[0] == snap.generation:
            return cached[1]

    # Rasterise outside the lock — cv2.fillPoly is the expensive bit.
    mask = np.ones((h, w), dtype=np.uint8)
    pts_int = [
        np.round(p * np.array([w, h], dtype=np.float32)).astype(np.int32)
        for p in snap.polygons_norm
    ]
    if pts_int:
        cv2.fillPoly(mask, pts_int, color=0)

    with _lock:
        _raster_cache[key] = (snap.generation, mask)
    return mask


def point_inside_any(point_norm: tuple[float, float]) -> bool:
    """True if the normalized (x, y) point falls inside any enabled zone."""
    snap = _load_snapshot_if_stale()
    if not snap.polygons_norm:
        return False
    px, py = float(point_norm[0]), float(point_norm[1])
    for poly in snap.polygons_norm:
        # cv2.pointPolygonTest wants float32 contour and a tuple.
        # measureDist=False returns +1 (inside), 0 (edge), -1 (outside).
        if cv2.pointPolygonTest(poly, (px, py), False) >= 0:
            return True
    return False


def bbox_centroid_inside_any(bbox_normalized) -> bool:
    """Convenience: check the bbox centre.  bbox = [ymin, xmin, ymax, xmax]."""
    ymin, xmin, ymax, xmax = bbox_normalized
    cx = (float(xmin) + float(xmax)) * 0.5
    cy = (float(ymin) + float(ymax)) * 0.5
    return point_inside_any((cx, cy))
