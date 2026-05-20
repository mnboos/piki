"""Per-event JSONL technical log.

For each event-triggered recording we create a `<stem>.log.jsonl` sidecar
next to the video file.  Lines are written in order:

    {"event": "event_started",  …, "config_snapshot": {…}}
    {"event": "frame", "frame_idx": 0, "prebuffer": true,  …}
    …
    {"event": "frame", "frame_idx": N, "prebuffer": false, …}
    {"event": "splash_fired", "duration_seconds": …}
    {"event": "event_ended", "reason": …, "frames_written": …, …}

The writer is line-buffered and thread-safe — pre-buffer flush happens on
the streaming thread, while later frames are written from `process_frame`
(same thread in practice, but the lock keeps us honest).
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class EventLogger:
    __slots__ = ("path", "event_id", "_fh", "_lock", "_closed")

    def __init__(self, log_path: str, event_id: str):
        self.path = log_path
        self.event_id = event_id
        self._lock = threading.Lock()
        self._closed = False
        # Line-buffered text mode: each `write()` line is flushed to the OS.
        # This survives crashes — the JSONL is always usable even mid-event.
        self._fh = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        # Stamp timestamp if the caller didn't provide one.
        payload.setdefault("ts", _now_iso())
        try:
            line = json.dumps(payload, separators=(",", ":"), default=_json_default)
        except (TypeError, ValueError):
            logger.exception("EventLogger: failed to serialize payload keys=%s", list(payload.keys()))
            return
        with self._lock:
            if self._closed:
                return
            try:
                self._fh.write(line + "\n")
            except OSError:
                logger.exception("EventLogger: write failed (path=%s)", self.path)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._fh.close()
            except OSError:
                logger.exception("EventLogger: close failed (path=%s)", self.path)


def _json_default(o):
    # numpy scalars / arrays etc.
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def summarize_log(log_path: str) -> dict[str, Any]:
    """Stream-parse the JSONL and return a compact summary for the UI.

    Computed on demand (no caching) since events are short and the file is
    small.  Returns counters + a per-label histogram of peak confidence.
    """
    summary: dict[str, Any] = {
        "event_id": None,
        "started_at": None,
        "ended_at": None,
        "duration_seconds": 0.0,
        "frames_total": 0,
        "frames_prebuffer": 0,
        "frames_live": 0,
        "detections_total": 0,
        "labels": {},   # label → {count, peak_confidence}
        "splashes": 0,
        "trigger": None,
        "config_snapshot": None,
        "unique_track_ids": [],
    }
    track_ids: set[int] = set()
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ev = rec.get("event")
                if ev == "event_started":
                    summary["event_id"] = rec.get("event_id")
                    summary["started_at"] = rec.get("ts")
                    summary["trigger"] = rec.get("trigger")
                    summary["config_snapshot"] = rec.get("config_snapshot")
                elif ev == "frame":
                    summary["frames_total"] += 1
                    if rec.get("prebuffer"):
                        summary["frames_prebuffer"] += 1
                    else:
                        summary["frames_live"] += 1
                    for det in rec.get("detections") or []:
                        summary["detections_total"] += 1
                        lbl = det.get("label", "?")
                        bucket = summary["labels"].setdefault(lbl, {"count": 0, "peak_confidence": 0.0})
                        bucket["count"] += 1
                        c = float(det.get("confidence") or 0.0)
                        if c > bucket["peak_confidence"]:
                            bucket["peak_confidence"] = c
                        tid = det.get("track_id")
                        if isinstance(tid, int):
                            track_ids.add(tid)
                elif ev == "splash_fired":
                    summary["splashes"] += 1
                elif ev == "event_ended":
                    summary["ended_at"] = rec.get("ts")
                    summary["duration_seconds"] = float(rec.get("duration_seconds") or 0.0)
    except FileNotFoundError:
        pass
    summary["unique_track_ids"] = sorted(track_ids)
    return summary
