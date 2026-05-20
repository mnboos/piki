# Plan: Exclusion Zones, Technical Log, and Detection Stability

> **Maintainer note:** This document is intended to be picked up by another
> agent/engineer with no prior session context. Read this file end-to-end
> before starting; each feature section is self-contained with file paths,
> line numbers, integration points, and verification steps.

---

## Context

The user is adding three features to **piki**, a Django + Vue cat-deterrent
system running on a D-Robotics RDK X5 SBC. The system runs YOLOv8 on the
Horizon BPU against MIPI camera frames and aims pan/tilt servos at detected
targets (see `README.md` for hardware/architecture overview).

Why these three features now:

1. **Exclusion zones** — the camera's field of view inevitably includes
   regions that should never trigger a detection (a neighbour's window, a
   busy street, parts of the garden where wildlife is welcome). Today the
   only way to filter detection geographically is through the motion-driven
   ROIs, which are generative (where motion *is*), not prescriptive (where
   detection *should not run*).
2. **Technical log** — when an event recording fires there is currently no
   record of *why*: which detections crossed the threshold, what the servo
   was doing, what the frame timestamp was, what the motion mask looked
   like. The user wants a structured per-event log paired with each
   recording so post-hoc analysis of false positives / missed events is
   possible.
3. **Detection jitter** — single-frame drop-outs cause visible flicker,
   premature recording stops, and servo "stuttering". The detection model
   itself is good; what is missing is temporal continuity across frames.

Repository: `/home/sunrise/src/piki` — Python is managed by **uv** (see
`CLAUDE.md`); always use `uv run ...`, never bare `python3` or `pip`.

---

## Architecture refresher (read before touching code)

### Frame pipeline (single source of truth)

```
ROS2 camera node (hobot_stereonet, 640×352 NV12)
  → listener_callback_hbm()                 [stream.py:198]
    → process_frame(nv12, frame_h)          [stream.py:919]
      → MotionDetector.is_moving(lores)     [shared.py:163]   # MOG2, 320×176
      → MotionDetector.create_rois(mask)    [shared.py:207]   # blob → rect
      → executor.submit(run_object_detection, …)              # YOLO on tiles
      → recording.write_frame(bgr)          [recording.py:124]  # if manual
      → _event_recorder.write_frame(bgr)    [stream.py:956]    # if event
      → recording.pre_buffer_append(bgr, …) [recording.py:23]
    on YOLO future complete:
      → on_done(future)                     [stream.py:621]
        → aim_at(target, …)                 [engine.py:347]
        → _start_event_recording(…)         [stream.py:827]
        → _finalize_event_recording(…)      [stream.py:861]
```

### Coordinate systems (gotcha)

| Stage | Resolution | Origin | Unit |
|---|---|---|---|
| Camera (NV12) | 640×352 | top-left | px |
| Motion / preview (`lores`) | 320×176 | top-left | px |
| Motion ROI tuples `(x, y, w, h)` | lores | top-left | px |
| YOLO output bbox | normalized | top-left | `[ymin, xmin, ymax, xmax]` ∈ [0, 1] |
| Servo angle | — | image centre | degrees |

**All persistent geometry in this plan is stored in normalized `[0, 1]`
coordinates** so the database survives resolution changes and applies
identically to lores/full/preview frames after a single multiply.

### Singleton Django models (all in `src/core/models.py`)

`DetectionConfig`, `EventRecordingConfig`, `SplashConfig`, `AimConfig`
each use `pk=1` and a `load()` classmethod. New singletons in this plan
follow the same pattern. The existing `Video` model is the only
non-singleton.

### Visualization

All overlays (boxes, mask, ROIs, crosshairs, segmentation) are rendered
**server-side** in OpenCV inside `stream_camera()` (`api.py:68-194`) before
JPEG encoding. The frontend `CameraFeed.vue` is a plain `<img>` consuming
the MJPEG stream — there is **no client-side canvas today**. This matters
for the exclusion-zone UX choice (see Feature 1).

---

## Feature 1 — Scan-region exclusion zones

### Goal

Allow the user to draw one or more "ignore me" regions on the camera view.
Anything inside an exclusion zone must be suppressed from:

- ROI creation (saves inference time)
- Detection results (defense in depth — catches any detection whose
  centroid lies inside a zone even if motion leaked through)
- Servo aiming (the aim target must never land in an exclusion zone)
- Event-recording triggers (no recording fires from inside a zone)

### Data model

Add to `src/core/models.py`:

```python
class ExclusionZone(models.Model):
    """Polygonal region in normalized [0,1] camera-frame coordinates
    that must be ignored by detection, aiming, and recording triggers."""

    name = models.CharField(max_length=64, default="zone")
    enabled = models.BooleanField(default=True)
    # JSON: [[x, y], [x, y], …] in normalized [0,1], min 3 points (polygon).
    # A rectangle is represented as 4 points.
    points = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return self.name
```

Generate migration:

```bash
uv run python src/manage.py makemigrations core
uv run python src/manage.py migrate
```

### Backend application points

There are three layers where zones must be applied. **All three are
required.** Order matters for performance: zero out motion first so YOLO
never runs on excluded pixels.

#### 1.1 Mask suppression — earliest filter

In `src/core/utils/shared.py`, modify `MotionDetector.is_moving()` (line
163) — or add a new step in `process_frame()` immediately after it (line
931) — to multiply the motion mask by an "allowed" mask:

```python
# pseudocode, placed after has_movement, mask = motion_detector.is_moving(...)
allowed_mask = _exclusion_cache.allowed_mask_for(lores_shape)
if allowed_mask is not None:
    mask = cv2.bitwise_and(mask, allowed_mask)
    has_movement = bool(mask.any())
```

The allowed-mask cache must be invalidated whenever any `ExclusionZone`
row changes. Implementation:

- Maintain a module-level `_exclusion_cache` keyed by `(lores_h, lores_w,
  generation)` where `generation` is bumped on every zone CRUD.
- `allowed_mask_for(shape)` rasterises all enabled zones once with
  `cv2.fillPoly` into a `uint8` array of `1`s with `0`s inside zones, then
  returns it.

#### 1.2 Detection filter — defense in depth

In `src/core/utils/stream.py` `on_done()` (around line 705 where
detections are denormalized for aiming and recording), drop any detection
whose **bbox centroid in normalized coordinates** is inside an enabled
zone. Use `cv2.pointPolygonTest` or a numpy ray-casting helper.

Concretely, factor a helper:

```python
def _detection_in_exclusion(bbox_normalized) -> bool:
    # bbox = [ymin, xmin, ymax, xmax] normalized
    cx = (bbox_normalized[1] + bbox_normalized[3]) / 2
    cy = (bbox_normalized[0] + bbox_normalized[2]) / 2
    return _exclusion_cache.point_inside_any((cx, cy))
```

Apply it before the lock-update loop and before the event-trigger check
(stream.py:808-818). Filtered detections must also be excluded from the
displayed bounding boxes (so they don't flash on screen).

#### 1.3 Servo aim clamp (cheap belt-and-braces)

In `engine.py` `aim_at()` (line 347), before issuing PWM, convert the
target pixel → normalized and reject if inside an exclusion zone. This is
nearly redundant with 1.2 but guards against stale `_locked_target_bbox`
state surviving a zone change.

### Backend API

Add to `src/core/api.py` — endpoints under the existing `/api` ninja
router:

| Verb | Path | Body | Returns |
|---|---|---|---|
| GET | `/exclusion_zones` | — | `list[ExclusionZoneSchema]` |
| POST | `/exclusion_zones` | `ExclusionZoneSchema (no id)` | created row |
| PATCH | `/exclusion_zones/{id}` | partial `ExclusionZoneSchema` | updated row |
| DELETE | `/exclusion_zones/{id}` | — | `{ok: true}` |

Schema:

```python
class ExclusionZoneSchema(Schema):
    id: int | None = None
    name: str = "zone"
    enabled: bool = True
    points: list[tuple[float, float]]  # normalized [0,1], ≥ 3 points
```

Mutations bump `_exclusion_cache.generation` so the inference loop
rebuilds the rasterised mask on its next frame. The cache module should
expose `bump_generation()` and the API handlers call it.

### Server-side visualization

In `src/core/api.py` `stream_camera()` (line 68), after the existing ROI
drawing block, add a render pass for exclusion zones whenever any zone
is enabled (always show — they are safety-critical, not optional):

```python
for zone in _exclusion_cache.zones():
    pts = (zone.points * np.array([lores_w, lores_h])).astype(np.int32)
    cv2.polylines(draw_frame, [pts], isClosed=True, color=(0, 0, 200), thickness=2)
    # semi-transparent red fill
    overlay = draw_frame.copy()
    cv2.fillPoly(overlay, [pts], color=(0, 0, 200))
    cv2.addWeighted(overlay, 0.25, draw_frame, 0.75, 0, draw_frame)
```

### Frontend UI

The codebase has **no client-side drawing canvas** today. We add one as an
SVG overlay positioned absolutely on top of `CameraFeed.vue`. SVG (not
canvas) because we need hit-testing on vertices to drag/edit them.

**Recommended UX (rectangle-only v1, polygon-ready v2):**

- `HomeView.vue` gains a "Zones" mode toggle next to the existing Boxes /
  Mask / ROIs / Seg buttons.
- When Zones mode is on, a `<svg>` overlays the `<img>` and:
  - Click-drag draws a new rectangle (release commits — `POST`).
  - Existing zones render as draggable rectangles with corner handles.
  - Each zone shows its name, an enable/disable toggle, and a delete
    button (a sidebar panel `ExclusionZonesPanel.vue` lists them
    alongside the existing detection controls).
- All coordinates the SVG manipulates are in normalized `[0, 1]`,
  converted to pixels for rendering via the SVG `viewBox`. The MJPEG
  `<img>` natural resolution is the source of truth.

**Files to add:**

- `frontend/src/components/ExclusionZoneOverlay.vue` — SVG overlay
- `frontend/src/components/ExclusionZonesPanel.vue` — list/edit sidebar
- `frontend/src/queries/exclusionZones.ts` — TanStack Query wrappers
  for the four endpoints, following the same pattern as existing
  queries in `frontend/src/queries/`.

**Files to modify:**

- `frontend/src/views/HomeView.vue` — embed overlay + panel, add toggle.
- `frontend/src/components/CameraFeed.vue` — accept slot or sibling for
  overlays; expose its natural dimensions via ref.

### Critical files

- `src/core/models.py` — add `ExclusionZone`
- `src/core/api.py` — endpoints + stream draw
- `src/core/utils/shared.py` — mask suppression + cache
- `src/core/utils/stream.py:705-820` — detection filter
- `src/core/utils/engine.py:347-480` — aim clamp
- `frontend/src/components/ExclusionZoneOverlay.vue` (new)
- `frontend/src/components/ExclusionZonesPanel.vue` (new)
- `frontend/src/queries/exclusionZones.ts` (new)
- `frontend/src/views/HomeView.vue`
- `frontend/src/components/CameraFeed.vue`

### Verification

1. `uv run python src/manage.py makemigrations && migrate` — migration applies.
2. `uv run python src/manage.py shell -c "from core.models import ExclusionZone; ExclusionZone.objects.create(name='test', points=[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]])"` — create a zone covering most of the frame.
3. Start the system (`./run.sh`), hold the cat.jpg toy in view → no
   detection should fire because everything is excluded.
4. Delete the zone via the UI → detection should resume immediately
   (cache generation bumped).
5. Draw a smaller zone over an irrelevant part of the frame; verify a
   detection inside the zone is rendered invisible and does not fire the
   event recorder; verify the same detection outside the zone behaves
   normally.

---

## Feature 2 — Technical log paired with recordings

### Goal

For every event recording, persist a structured log file that contains
every fact needed to reconstruct *what the system saw and decided*
during that event. The log must be **easily linkable to its recording**
both on disk and through the API.

### Format choice: JSONL sidecar + global rotating text log

Two complementary outputs:

**A. Per-event JSONL sidecar** — one line per event-relevant moment.
Located next to the video file, identical stem, `.log.jsonl` suffix:

```
/home/sunrise/src/piki/src/media/videos/event_20260520_143012_<uuid>.avi
/home/sunrise/src/piki/src/media/videos/event_20260520_143012_<uuid>.log.jsonl
```

The `<uuid>` is a short (8-char) random hex appended to the timestamp to
guarantee uniqueness — solves the existing second-granularity collision
risk.

Event log entry schemas (use one `event` field as a discriminator):

```json
{"ts": "2026-05-20T14:30:12.345+02:00", "frame_ns": 1234567890123, "event": "event_started",
 "event_id": "20260520_143012_a1b2c3d4",
 "trigger": {"label": "cat", "confidence": 0.78, "bbox_norm": [0.41,0.22,0.69,0.55]},
 "config_snapshot": {"conf_threshold": 0.4, "pre_buffer_seconds": 5, "post_trigger_seconds": 10, "cooldown_seconds": 30, "trigger_classes": ["cat","dog"]},
 "frame": {"w": 640, "h": 352, "fps_estimate": 28.4}}

{"ts": "...", "frame_ns": ..., "event": "frame", "frame_idx": 137,
 "motion": {"pixel_count": 1840, "rois": [[12,20,160,140]]},
 "detections": [{"label":"cat","confidence":0.81,"bbox_norm":[0.41,0.22,0.69,0.55],"inside_exclusion":false,"track_id":7,"track_age_frames":12}],
 "servo": {"pan_deg": -12.4, "tilt_deg": 3.1, "locked_label": "cat", "locked_lost_since_ms": null},
 "splash": {"armed": false, "fired": false}}

{"ts": "...", "event": "splash_fired", "duration_seconds": 1.0}

{"ts": "...", "event": "event_ended", "reason": "post_trigger_elapsed",
 "frames_written": 432, "duration_seconds": 15.2,
 "detections_total": 287, "unique_track_ids": [7,9]}
```

The `event: "frame"` line is written for **every frame written to the
recording** (pre-buffer flushed frames + live frames). Pre-buffer frames
are tagged with `prebuffer: true` so it's clear they predate the trigger.

**B. Global rotating log** — a single file `logs/piki.log` (rotated daily,
14-day retention) capturing everything the Python `logging` module already
emits. This is the "tail this when something is broken" log, distinct
from the per-event structured record.

### Backend implementation

#### 2.1 Event ID and filename

In `src/core/utils/stream.py:847` change the filename pattern to include
a short UUID:

```python
event_uuid = uuid.uuid4().hex[:8]
event_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event_uuid}"
filename = f"event_{event_id}.mp4"
log_filename = f"event_{event_id}.log.jsonl"
```

#### 2.2 EventLogger class

New file: `src/core/utils/event_log.py`.

```python
class EventLogger:
    """Thread-safe JSONL writer for a single event recording.

    Lifecycle is bound to EventClipRecorder — created when an event starts,
    closed in _finalize_event_recording.
    """
    def __init__(self, log_path: str, event_id: str):
        self._path = log_path
        self._fh = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
        self._lock = threading.Lock()
        self.event_id = event_id

    def write(self, payload: dict) -> None:
        payload.setdefault("ts", datetime.now(timezone.utc).astimezone().isoformat())
        with self._lock:
            self._fh.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()
```

#### 2.3 Wire-in points

- **`_start_event_recording()` (stream.py:827)** — instantiate `EventLogger`,
  write `event_started` entry with config snapshot. Store reference on
  the module-level globals next to `_event_recorder`.
- **`process_frame()` (stream.py:919)** — after `_event_recorder.write_frame`,
  call `_event_logger.write({"event":"frame", ...})` with motion stats
  and the latest detection list. The frame index counter increments per
  written frame.
- **`on_done()` (stream.py:707-748)** — when detections complete,
  cache them in a module-level `_latest_detections` (a small ring of the
  last few inference results) so `process_frame()` can attach the most
  recent detection batch to each frame log entry without blocking.
- **`_finalize_event_recording()` (stream.py:861)** — write
  `event_ended` entry, close logger, store `log_path` and `event_id` on
  the `Video` row.

#### 2.4 Video model changes

```python
class Video(models.Model):
    # ... existing fields ...
    event_id = models.CharField(max_length=64, blank=True, default="")
    log_file = models.FileField(upload_to="videos/", blank=True, null=True)
```

Migration via `makemigrations`/`migrate`.

#### 2.5 Global rotating log

Modify `src/piki/settings.py:149` `LOGGING` dict to add a file handler:

```python
"handlers": {
    "console": {...},  # unchanged
    "file": {
        "class": "logging.handlers.TimedRotatingFileHandler",
        "filename": str(BASE_DIR.parent / "logs" / "piki.log"),
        "when": "midnight",
        "backupCount": 14,
        "formatter": "verbose",
        "filters": ["add_relative_path"],
    },
},
"loggers": {
    "core": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": True},
},
```

Ensure `logs/` directory is created at startup (or in `apps.py:ready()`).

#### 2.6 API surface

- `GET /api/videos/{id}/log` → returns JSONL file as a download
  (`Content-Type: application/x-ndjson`). 404 if no log file linked.
- `GET /api/videos/{id}/log?format=summary` → returns a compact JSON
  summary parsed from the JSONL (counts, peak confidence, list of
  distinct labels, track IDs, total duration). This is cheap to compute
  on demand by streaming the file.

### Frontend changes

`RecordingsPanel.vue` (lines 228-307) — for each event clip row, add:

- A "Log" button next to Play/Download that opens a modal showing the
  structured summary (labels seen, peak confidence, frame count,
  duration) and a "Download .jsonl" link.

### Critical files

- `src/core/models.py` — `Video.event_id`, `Video.log_file`
- `src/core/utils/event_log.py` (new)
- `src/core/utils/stream.py:827, 861, 919, 707-748` — wiring
- `src/core/api.py` — log download / summary endpoints; filename change
- `src/piki/settings.py:149` — file handler
- `src/core/apps.py` — ensure logs dir exists at startup
- `frontend/src/components/RecordingsPanel.vue` — log button + modal

### Verification

1. Trigger an event recording (hold cat.jpg in view).
2. Confirm two files appear in `src/media/videos/` with matching stem:
   the `.avi`/`.mp4` and the `.log.jsonl`.
3. `head -3` the JSONL: first line is `event_started`, second line a
   pre-buffer `frame` entry, later lines are live frames, final line is
   `event_ended` with summary counters.
4. `wc -l <log>.jsonl` should equal `event_started + frames_written + event_ended ± optional events`.
5. `GET /api/videos/<id>/log?format=summary` returns plausible counters.
6. Restart the server; confirm `logs/piki.log` has new entries and
   rotates at midnight (verify by setting `when="S"` temporarily).
7. Confirm filename collision impossible by simulating two events within
   the same second in a unit test (each gets a different UUID suffix).

---

## Feature 3 — Detection stability (anti-jitter)

### Diagnosis

Today there is no tracker (one was removed in commit `4048df8`). Each
frame's YOLO output is consumed independently. The result is:

- **Box flicker** in the UI when YOLO randomly misses a detection.
- **Recording cutoff** when a single-frame gap drops below threshold
  during the post-trigger window (not strictly true today because the
  trigger latches for the configured 10s, but cooldown / future
  refinements depend on continuity).
- **Servo stutter** because `_locked_target_bbox` is rewritten in place
  on every inference completion with the raw new bbox.

Three quick wins and one structural change. Phase A lands first because
its results inform tracker tuning, but **both phases ship in this work
slice** (per user decision).

### Phase A — Quick wins (1-2 days, low risk)

#### A.1 Confidence hysteresis

Replace the single `conf_threshold` with two:

- `conf_threshold_enter` — what a detection needs to *start* mattering
  (servo lock, recording trigger).
- `conf_threshold_keep` — what a detection needs to *stay* mattering
  (continued servo aim, no end-of-recording).

In `DetectionConfig`: rename `conf_threshold` to `conf_threshold_enter`
(keep a property alias for the old name) and add `conf_threshold_keep`.
Defaults: enter=0.4 (existing), keep=0.25.

Apply in `on_done()` and the event-trigger check: a detection meets
"start" only if confidence ≥ enter; a detection meets "keep" if it
matches a previously confirmed track and confidence ≥ keep.

#### A.2 Min-streak gate

Add `min_consecutive_frames` (default 2) to `DetectionConfig`. A class
must be detected on N consecutive *inference completions* (not raw
frames, since inference is motion-gated) before it counts as
"confirmed". Apply by maintaining a small per-label counter in
`on_done()` that increments when a matching detection arrives and
decays on misses.

This is **not** a real tracker — it's a same-class consecutive-hit
counter. It catches isolated single-frame false positives without the
complexity of identity-tracking.

#### A.3 Bounding-box EMA

In the `_locked_target_bbox` update path (around `stream.py:660-705`),
replace the direct assignment with an exponential moving average over
the normalized bbox corners:

```python
α = 0.4  # tunable; lower = smoother, higher = more responsive
if _locked_target_bbox is not None:
    new = (1 - α) * np.array(_locked_target_bbox) + α * np.array(new_bbox)
    _locked_target_bbox = new.tolist()
else:
    _locked_target_bbox = new_bbox
```

This visibly stabilises the on-screen box and reduces the magnitude of
each servo correction without any tracking infrastructure.

#### A.4 Ghost frames for visualization

When the most recent inference yields no detection but the previous one
did within the last `ghost_frames_ms` (default 300 ms), keep drawing the
last bbox **on the MJPEG stream only** (visual stability) — do **not**
let it influence servo or recording state (functional correctness). Add
`ghost_frames_ms` to `DetectionConfig`.

### Phase B — Structural fix: lightweight IoU + Kalman tracker

Ships alongside Phase A so the technical log emits real `track_id` /
`track_age_frames` values from day one. Default `tracker_enabled=true`
once integration tests pass; ship behind a config flag so the user can
disable it if it misbehaves in the field. Recommended approach: a
**SORT-style tracker** entirely in numpy — no extra dependencies
(opencv-contrib already provides `cv2.KalmanFilter`).

#### Design

New file: `src/core/utils/tracking.py`.

```python
@dataclass
class Track:
    id: int
    label: str
    bbox_norm: np.ndarray   # current (smoothed) bbox
    kalman: cv2.KalmanFilter  # state = [cx, cy, w, h, dcx, dcy], measurement = [cx, cy, w, h]
    hits: int            # total detection hits
    misses: int          # consecutive misses since last hit
    age_frames: int      # total frames the track has existed
    last_confidence: float

class IouTracker:
    def __init__(self, *, iou_threshold=0.3, max_misses=10, confirm_hits=3):
        ...

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        - Predict each existing track's bbox via Kalman.
        - Greedy IoU match predictions ↔ detections, same-label only.
        - Matched: update Kalman with measurement, hits += 1, misses = 0.
        - Unmatched detections → new tracks (initial state = bbox).
        - Unmatched tracks → misses += 1; drop if misses > max_misses.
        - Returns the full list of *confirmed* tracks (hits ≥ confirm_hits).
        """
```

**Why SORT-style and not ByteTrack:** ByteTrack uses low-confidence
detections to maintain tracks, but our YOLO post-processing already
applies the confidence threshold inside `yolov8_post_process`
(`ai.py:264`). To benefit from ByteTrack we'd need a second pre-NMS pass
exposed by the model wrapper, which is invasive. SORT-with-Kalman is
the right complexity for a single-board single-camera deployment.

#### Wire-in

`on_done()` becomes:

```python
tracks = _tracker.update(detections)
confirmed = [t for t in tracks if t.hits >= confirm_hits]
# downstream consumes `confirmed`, not raw `detections`:
#   - aim_at picks the longest-lived confirmed track matching aim classes
#   - event-recording trigger uses confirmed tracks
#   - drawing iterates confirmed tracks (so boxes show track_id labels)
```

The tracker becomes the source of identity for the technical log
(`track_id`, `track_age_frames` fields in Feature 2).

#### Inference-gap mitigation

A subtle issue: the motion gate (`stream.py:986`) skips inference when
nothing moves, so tracks would mark misses every quiet frame. Fix by
calling `_tracker.tick_idle()` on motion-gated frames — increments age
but not misses, so a stationary target's track survives quiescence.

### Configuration knobs to expose

Add to `DetectionConfig` (with sensible defaults):

| Field | Default | Purpose |
|---|---|---|
| `conf_threshold_enter` (rename) | 0.4 | A.1 |
| `conf_threshold_keep` | 0.25 | A.1 |
| `min_consecutive_frames` | 2 | A.2 |
| `bbox_ema_alpha` | 0.4 | A.3 |
| `ghost_frames_ms` | 300 | A.4 |
| `tracker_enabled` | true | B (ships on; user can disable) |
| `tracker_iou_threshold` | 0.3 | B |
| `tracker_max_misses` | 10 | B |
| `tracker_confirm_hits` | 3 | B |

### Critical files

- `src/core/models.py` — config field additions / rename
- `src/core/utils/stream.py:621-820, 986` — hysteresis, min-streak, EMA,
  ghost frames, tracker wire-in
- `src/core/utils/tracking.py` (new, Phase B)
- `src/core/api.py` — surface the new config fields in `PikiOptions`
- `frontend/src/components/DetectionControls.vue` — sliders for the new
  thresholds

### Verification

1. **A.1 / A.2 unit-test stub:** write a pytest case
   (`src/core/tests.py`) that feeds a synthetic detection stream
   (cat, cat, ø, cat, ø, ø, cat, cat) into a function that wraps
   the gating logic. Assert min-streak=2 suppresses isolated hits and
   keep-threshold maintains lock across single misses.
2. **A.3 visual check:** with the running system, hold a printed cat
   image still and toggle `bbox_ema_alpha` between 0.05 (extremely
   smooth — laggy) and 1.0 (no smoothing — jittery). Pick a value where
   the displayed box looks stable but follows hand motion.
3. **A.4 visual check:** with `ghost_frames_ms=300`, briefly occlude
   the target with a hand. The box should persist for ~10 frames before
   fading; servo should NOT chase the ghost (verify by watching servo
   angles in `/api/tracker_status`).
4. **Phase B integration check:** enable tracker, walk the printed cat
   across the frame slowly. Track ID stays constant; track ID changes
   on a fast off-screen / on-screen jump (expected). Compare the
   `.log.jsonl` (Feature 2) `track_id`/`track_age_frames` fields against
   visual expectation.
5. **No-regression spot check:** with `tracker_enabled=false`, the system
   behaves exactly as before Phase A's changes for users who don't opt
   in to the new defaults (defaults should be conservative: enter=0.4,
   keep=0.25, min_streak=2, EMA=0.4, ghost=300 — these are mild changes
   that should only stabilise, never destabilise).

---

## Cross-feature integration

The three features lean on each other:

- The **technical log** (Feature 2) is the primary diagnostic tool for
  tuning **detection stability** (Feature 3). Land Feature 2 first so
  Phase A of Feature 3 is measurable.
- **Exclusion zones** (Feature 1) emit an `inside_exclusion: bool` per
  detection in the per-frame log entries (Feature 2). This lets the
  user see *why* a detection was suppressed.
- The tracker (Feature 3 Phase B) is the source of `track_id` /
  `track_age_frames` in the log (Feature 2). Until Phase B ships,
  those fields remain `null`.

Implementation order (all five slices ship in this work):

1. Feature 2 part A: rotating global log (10 minutes — pure config).
2. Feature 3 Phase A.1–A.4 (1-2 days — biggest user-facing win).
3. Feature 1: exclusion zones (rectangles only in v1) (2-3 days —
   independent slice).
4. Feature 2 part B: per-event JSONL + Video model fields + UI (1-2 days).
5. Feature 3 Phase B: SORT-style tracker (2-3 days).

---

## Out of scope

These were considered and deliberately deferred:

- **Per-user / per-camera scoping** of any of these features. Piki is a
  single-camera, single-user appliance today; the singleton model
  pattern is fine.
- **Multi-zone categories** (e.g., "ignore", "high-priority"). The
  initial design is binary inclusion.
- **Disk-space management** for recordings and logs. Already missing; not
  worsened by this work. Worth a separate ticket.
- **Cross-event correlation** in the technical log (e.g., "this is the
  fourth cat visit today"). Each event is self-contained for v1.
- **Replacing OpenCV server-side rendering with client-side WebGL**.
  Bigger architectural change; the SVG overlay for zones coexists with
  the existing MJPEG model.

---

## Decisions (resolved with user)

1. **Exclusion zone shape:** rectangles only in v1. Data model still
   stores normalized polygon points (`list[list[float]]`) so polygon
   support is a UI-only addition later — no migration needed.
2. **Technical log granularity:** every frame written to the recording
   gets a JSONL line, including pre-buffer flushed frames (tagged
   `prebuffer: true`).
3. **Jitter scope:** Phase A and Phase B ship together. Tracker defaults
   to enabled; can be turned off via `tracker_enabled=false`.
4. **Naming:** `ExclusionZone` model, "Exclusion zones" UI label.

---

## Post-ship fix: exclusion zones bypassed by tracker predictions

### Context

After all five slices shipped, the user reported "I can create the
exclusion zones but they seem to be ignored." A diagnostic question
confirmed: the **red translucent overlay drawn by the backend appears
at the right place** on the camera feed — the zones are persisted,
the cache invalidates correctly, and the server-side draw is correct.
What still fails: **detection boxes (and aim, and the event-recording
trigger) continue to fire inside the zone**.

### Root cause

`src/core/utils/stream.py:on_done()` filters YOLO detections by
exclusion-zone centroid (lines 685-704) **before** feeding them to the
SORT tracker. The tracker is fine — it correctly receives an empty
list when all detections are in zones. But:

1. Existing tracks (created before the zone was drawn, or moved
   *into* the zone from outside) keep predicting via Kalman for up to
   `tracker_max_misses` frames (~10 inference completions) inside the
   zone.
2. `if t.confirmed` at line 754 admits those predicted tracks back into
   `detections` (the replacement list that downstream consumes).
3. Downstream consumers — display (`detections_denormalized` →
   `latest_ai_detections` → MJPEG draw), aim (via the locked-target
   resolver), and the event-recording trigger — all see the predicted
   ghost track at its last-known position inside the zone.

The `aim_at()` belt-and-braces clamp in `engine.py:347` still suppresses
servo motion, but the on-screen box and the event trigger are not
guarded by that path.

Net effect: every time the user draws a zone over a detection, the
detection's bounding box, label, and confidence keep painting on the
MJPEG (and can fire recordings) for ~1-2 seconds before `max_misses`
drops the predicted track.

### Fix

One small change in `src/core/utils/stream.py:on_done()`, immediately
after `tracks = _tracker.update(detections)` (currently line 726):

```python
tracks = _tracker.update(detections)

# Hide tracks whose predicted bbox center is inside an exclusion zone
# from downstream consumers (display, aim, recording, log annotation).
# The track stays alive in the tracker so it can be re-acquired when
# the target leaves the zone — we just suppress its output while it's
# still predicting inside the carve-out.
if zones_active:
    visible_tracks = [
        t for t in tracks
        if not _exclusion.bbox_centroid_inside_any(t.bbox_norm)
    ]
else:
    visible_tracks = tracks
```

Then everywhere below that currently iterates `tracks`, iterate
`visible_tracks` instead:

- The log-entry annotation loop (currently `for t in tracks` around
  line 737) — use `visible_tracks` so log entries don't get track ids
  from suppressed tracks.
- The detections replacement (currently `[... for t in tracks if
  t.confirmed]` at line 752-755) — use `visible_tracks`.

No new files, no new config, no migration. Single function-local
change. The tracker module itself is unchanged.

### Why hide rather than delete

Two approaches were considered:

- **Hide (chosen)**: track stays alive in the tracker; gets `misses++`
  each frame it's inside a zone; if the target exits the zone the
  same Kalman state matches its first real re-detection and the same
  track id is preserved. The track is dropped naturally after
  `max_misses` frames if it never exits.
- **Delete**: drop the track from the tracker outright. Simpler, but
  if the user draws a zone over a transient passage and the target
  exits the other side, a fresh track id is assigned, breaking
  identity in the technical log.

Hide is no more code and preserves identity across zone passages.

### Edge case (not covered by this fix)

A bbox whose centroid sits *outside* a zone but whose body mostly
overlaps the zone still passes the centroid check. For the user's
real-world case (small/medium animals where the centroid is near the
visual centre) this is a non-issue. If false positives appear later,
the fix is to switch the check to "any-overlap > X%" against the
zone polygon — easy, but not needed today.

### Critical files

- `src/core/utils/stream.py` (~lines 726-755) — the only file
  touched by the fix.

### Verification

1. Reproduce the original failure:
   - Start the system (`./run.sh`).
   - Aim a cat picture / toy at the camera so a confirmed track exists.
   - Switch to the Detection tab, toggle "Zones", drag a zone over the
     detection.
   - **Before fix:** the green detection box keeps painting inside the
     red zone for ~10 frames. With aim enabled, servo doesn't move
     (the belt-and-braces clamp is working) but the recording trigger
     can still fire if event recording is enabled.
   - **After fix:** the green box disappears within one inference
     completion of the zone being drawn.
2. Track-re-emergence test: draw a zone in the upper-left quarter,
   walk the target diagonally through the zone toward the lower-right.
   Track id should *not* change when the target emerges from the
   other side (verify in the JSONL via Feature 2's log download or
   simply by glancing at the per-event summary's `unique_track_ids`).
3. No-zones regression: with no exclusion zones present, tracker
   output is unchanged (`visible_tracks is tracks`). Quick spot-check
   that detection behaviour matches what shipped before this fix.
4. Recording-trigger regression: with a zone covering the *whole*
   frame and `event_recording_enabled=true`, no event recording fires
   even when the model emits high-confidence detections. (Before
   fix: a recording could fire on a predicted track's last frame
   inside the zone before `max_misses` cleared it.)

---

## Post-ship fix #2: servo actively follows inside the zone

### Context

After fix #1 (hide tracks whose centroid is inside a zone) the user
still reports: **"servo actively follows me while I'm in the zone"**
and wants the servo to **move to a fixed home position (0°, 0°)**
when no valid target exists.

### Root cause

Three things compound to let the servo keep moving even when the
target is fully inside a zone:

1. **Kalman extrapolation past the zone boundary.**
   When the tracker has no detection for a track this frame (because
   the exclusion filter dropped it), it sets
   `t.bbox_norm = t.predicted_bbox` (`tracking.py` "Bookkeeping for
   unmatched tracks" block).  The Kalman state still carries the
   *velocity* learned from pre-zone frames.  If the user was walking
   into the zone, the predicted bbox keeps moving forward along that
   velocity vector — and a single extrapolation step is often enough
   to land the predicted centroid *outside* the zone polygon.
   Fix #1 only checks the centroid against the zone, so a track that
   extrapolates out of the zone becomes "visible" again and is
   emitted into `detections` for downstream consumers.  `aim_at`
   then drives the servo to that extrapolated position.

2. **Locked-target bbox is not re-checked against zones each frame.**
   `_locked_target_bbox` (the EMA-smoothed lock used for aim) is only
   cleared when the `target_lock_duration` (default 3 s) timer
   expires.  Even after the user is in the zone, the lock survives
   for up to 3 seconds — and during that window the EMA combines the
   pre-zone position with whatever the tracker now emits, including
   the extrapolations from (1).

3. **`aim_at` "suppression" doesn't move the servo home.**
   The belt-and-braces `aim_at` clamp returns `(0.0, 0.0)` when the
   aim point is inside a zone, but `return` doesn't update the PWM —
   it just declines to update.  The servo therefore *holds* at the
   last commanded position.  Combined with (1) and (2), that last
   position is on the user, so the gun stays pointed at them and
   appears to track them through small movements.

### Fix — three coordinated changes

All three changes are in `src/core/utils/stream.py:on_done()` (and one
small helper call into `engine.py`).  No model changes; no migration;
no new file.

#### 1. Strict track visibility when zones are active

Replace the current `visible_tracks` filter with a stricter version
that ignores **predicted-only tracks** as long as any zone exists:

```python
# In on_done, after `tracks = _tracker.update(detections)`:
if zones_active:
    # A track is only safe to expose when (a) it was matched against a
    # real detection this frame (misses == 0) AND (b) its current bbox
    # centroid is outside every zone.  Predicted-only tracks (misses>0)
    # are excluded because the Kalman state carries pre-zone velocity
    # and can extrapolate the bbox past the zone boundary, defeating
    # the centroid check on its own.
    visible_tracks = [
        t for t in tracks
        if t.misses == 0
        and not _exclusion.bbox_centroid_inside_any(t.bbox_norm)
    ]
else:
    visible_tracks = tracks
```

**Trade-off:** with zones active, the tracker no longer fills
single-frame YOLO misses for any track.  The user explicitly chose
"zones over jitter resistance" by drawing a zone, so this is the
right priority.  No regression when no zones exist.

#### 2. Release the lock immediately when its bbox enters a zone

Just before the existing lock-update block in `on_done()`:

```python
# If the lock has wandered into (or been covered by) a zone, release
# it now — don't wait `target_lock_duration` seconds for the natural
# lost-timer.  Otherwise the EMA keeps blending the pre-zone position
# into the lock and aim_at keeps firing at the smoothed bbox.
if (_locked_target_bbox is not None
        and _exclusion.has_zones()
        and _exclusion.bbox_centroid_inside_any(_locked_target_bbox)):
    logger.info("Locked target entered exclusion zone — releasing lock.")
    _locked_target_bbox = None
    _locked_target_label = None
    _locked_target_lost_since = None
    _just_released_lock = True   # flag consumed by step 3
else:
    _just_released_lock = False
```

#### 3. Move the servo to (0°, 0°) on any lock-release

The user's chosen idle behaviour is "fixed home position (0°, 0°)".
Implement it as a **one-shot** move on the lock-release transition
(not every frame, which would override manual debug control):

```python
def _go_home_servo():
    """One-shot move to (0°, 0°). Called on lock-release transitions."""
    from .engine import move_to  # already exists for ServoDebugPanel
    move_to(0.0, 0.0)
```

Call it from three places:
- The zone-triggered release in step 2 (when `_just_released_lock` is
  True and aim was enabled).
- The existing `target_lock_duration` timeout branch
  (`if elapsed >= target_lock_duration:` lines 717/731 in current
  stream.py, both branches).
- Wrap each with `if aim_enabled:` so we don't move the servo when
  servo aiming is globally disabled.

```python
if elapsed >= target_lock_duration:
    logger.info("Target lock expired after %.1fs.", elapsed)
    _locked_target_bbox = None
    _locked_target_label = None
    _locked_target_lost_since = None
    if aim_enabled and target_classes:
        _go_home_servo()
```

`engine.move_to(0, 0)` writes directly to PWM and updates
`servo_pan` / `servo_tilt` — exactly the path the ServoDebugPanel
"Move" button already uses, so we know it works and respects servo
limits.  It bypasses the dead-zone (manual moves should always take
effect).

### Critical files

- `src/core/utils/stream.py` — three small edits inside `on_done()`:
  - replace the `visible_tracks` filter with the strict version
  - add the locked-target-in-zone release block before the
    existing lock-update logic
  - call a `_go_home_servo()` helper from both lock-release paths
- `src/core/utils/engine.py` — **no changes**; uses existing
  `move_to()`.

### Verification

1. **Active-follow regression test:**
   - With `tracker_enabled=true`, `aim_enabled=true`, walk slowly
     into the camera's view from the left.  The servo locks on and
     follows you.
   - Drag a zone over your current position.
   - **Before fix #2:** the servo continues to drift to the right
     (Kalman extrapolation along your previous velocity) for ~1-2 s
     before stopping; the gun stays pointed at the extrapolated
     position.
   - **After fix #2:** within one inference completion of the zone
     being drawn, the servo snaps to `(0°, 0°)` and stays there
     while you remain in the zone.
2. **Stationary-in-zone test:** stand still inside a zone for 10 s.
   Servo stays at home.  No bounding boxes drawn over you.
3. **Walk-through test:** create a zone in the centre of the frame.
   Walk through it left to right.  Servo should aim at you on the
   left side, snap home as you enter the zone, then re-acquire and
   aim at you when you exit on the right.  A new `track_id` is
   expected (since fix #1's "hide-only" pass-through behaviour is
   relaxed by fix #2's strict filter — predicted tracks are dropped,
   so the tracker has no continuity through the zone).
4. **No-zones regression:** with no zones, behaviour is identical to
   before fix #2.  `visible_tracks` is `tracks` and predicted tracks
   keep filling single-frame gaps as designed in Slice 5.
5. **Manual-control regression:** with no detections at all,
   manually move the servo via the Debug tab.  The servo stays where
   you put it — `_go_home_servo` is only called on lock-release
   transitions, not on every empty frame.
