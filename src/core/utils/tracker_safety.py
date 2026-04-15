import math
from collections.abc import Sequence

from .interfaces import Box

_TRACKER_UPDATE_MARGIN_RATIO = 0.75
_TRACKER_MIN_EDGE_MARGIN_PX = 2
_BBOX_COMPONENTS = 4


def sanitize_bbox_for_frame(
    *,
    bbox: Sequence[float] | None,
    frame_shape: Sequence[int],
    edge_margin: int = 0,
) -> Box | None:
    """Clamp a bbox to frame bounds, returning None when it cannot be made valid."""
    result: Box | None = None

    if bbox is not None and len(bbox) == _BBOX_COMPONENTS:
        fh, fw = frame_shape[:2]
        if fh > 0 and fw > 0:
            x_f, y_f, w_f, h_f = bbox
            if all(math.isfinite(v) for v in (x_f, y_f, w_f, h_f)):
                x = int(x_f)
                y = int(y_f)
                w = int(w_f)
                h = int(h_f)
                if w > 0 and h > 0:
                    min_x = max(0, int(edge_margin))
                    min_y = max(0, int(edge_margin))
                    max_x = fw - min_x
                    max_y = fh - min_y
                    if max_x > min_x and max_y > min_y:
                        left = max(min_x, min(x, max_x - 1))
                        top = max(min_y, min(y, max_y - 1))
                        right = max(min_x + 1, min(x + w, max_x))
                        bottom = max(min_y + 1, min(y + h, max_y))
                        if right > left and bottom > top:
                            result = Box(left, top, right - left, bottom - top)

    return result


def required_update_margin(*, bbox: Sequence[int]) -> int:
    """Return the edge margin needed for safe tracker updates."""
    _, _, w, h = bbox
    return max(_TRACKER_MIN_EDGE_MARGIN_PX, round(max(w, h) * _TRACKER_UPDATE_MARGIN_RATIO))


def is_bbox_safe_for_update(*, bbox: Sequence[int] | None, frame_shape: Sequence[int]) -> bool:
    """Whether a bbox is fully in-frame with enough edge margin for tracker search windows."""
    if bbox is None or len(bbox) != _BBOX_COMPONENTS:
        return False

    fh, fw = frame_shape[:2]
    if fh <= 0 or fw <= 0:
        return False

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False

    margin = required_update_margin(bbox=(x, y, w, h))
    return x >= margin and y >= margin and (x + w) <= (fw - margin) and (y + h) <= (fh - margin)
