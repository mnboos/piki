<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from "vue";
import { useDetections, useMask, useRois, useTrackerStatus } from "@/composables/useEventStream";

const props = withDefaults(defineProps<{
    /** Draw detection bounding boxes. */
    showBoxes?: boolean;
    /** Draw the orange servo crosshair + cyan Kalman lead. */
    showCrosshair?: boolean;
    /** Draw motion-mask polygons (purple translucent). */
    showMask?: boolean;
    /** Draw motion ROI tiles (dashed orange). */
    showRois?: boolean;
}>(), {
    showBoxes: true,
    showCrosshair: true,
    showMask: false,
    showRois: false,
});

// Servo field-of-view (matches `SERVO_HFOV`/`SERVO_VFOV` env defaults in engine.py).
// If those env vars are changed on the server, this would need to be exposed
// via a settings endpoint — for now they're hardcoded to the defaults.
const SERVO_HFOV = 160.0;
const SERVO_VFOV = 100.0;

const canvas = ref<HTMLCanvasElement | null>(null);

const detections = useDetections();
const trackerStatus = useTrackerStatus();
const rois = useRois();
const mask = useMask();

// Cycle through a small palette by track id so each track has a stable colour.
const PALETTE = [
    "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
    "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#7cb305",
];
function colorForTrack(tid: number | null, label: string): string {
    if (tid !== null) return PALETTE[tid % PALETTE.length];
    let h = 0;
    for (const ch of label) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return PALETTE[h % PALETTE.length];
}

// --- Track trails -----------------------------------------------------------
// Each track stores its last N center-point positions (normalized [0,1]).
// Tracks not seen for STALE_TRACK_GRACE frames are removed.
interface TrailPoint { x: number; y: number; }
const MAX_TRAIL_LENGTH = 30;
const STALE_TRACK_GRACE = 60;
const TRAIL_ALPHA = 0.35;
const trailMap = new Map<number, TrailPoint[]>();
const trackMissCounter = new Map<number, number>();

function hexToRgba(hex: string, alpha: number): string {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

// ResizeObserver keeps the canvas backing store in sync with the displayed
// size (CSS pixels × devicePixelRatio).
let ro: ResizeObserver | null = null;
let raf: number | null = null;

function fitCanvas() {
    const c = canvas.value;
    if (!c) return;
    const rect = c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
    }
}

function draw() {
    raf = null;
    const c = canvas.value;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    const w = c.width;
    const h = c.height;
    ctx.clearRect(0, 0, w, h);

    // Mask polygons (bottom layer — purple fill matching the old server overlay).
    if (props.showMask && mask.value) {
        const polys = mask.value.polygons ?? [];
        if (polys.length) {
            ctx.fillStyle = "rgba(147, 20, 255, 0.35)";
            for (const poly of polys) {
                if (poly.length < 4) continue;
                ctx.beginPath();
                ctx.moveTo(poly[0] * w, poly[1] * h);
                for (let i = 2; i < poly.length; i += 2) {
                    ctx.lineTo(poly[i] * w, poly[i + 1] * h);
                }
                ctx.closePath();
                ctx.fill();
            }
        }
    }

    // ROI tiles (dashed orange, drawn over the mask but under the boxes).
    if (props.showRois && rois.value) {
        const items = rois.value.rois ?? [];
        if (items.length) {
            ctx.strokeStyle = "#FFA500";
            ctx.lineWidth = Math.max(1, Math.round(w / 800));
            ctx.setLineDash([6, 4]);
            for (const [rx, ry, rw, rh] of items) {
                ctx.strokeRect(rx * w, ry * h, rw * w, rh * h);
            }
            ctx.setLineDash([]);
        }
    }

    // Track trails (drawn behind boxes so trails don't obscure the current bbox).
    if (props.showBoxes && detections.value) {
        const dets = detections.value.detections ?? [];
        const trailWidth = Math.max(1, Math.round(w / 1000));

        for (const [tid, points] of trailMap) {
            if (points.length < 2) continue;
            // Use the same palette colour as the box — find a detection for this track
            // to get its colour, or fall back to the palette by id alone.
            const detForTid = dets.find(d => d.tid === tid);
            const color = detForTid
                ? colorForTrack(tid, detForTid.label)
                : PALETTE[tid % PALETTE.length];

            // Draw trail as segments with fading opacity (older → more transparent).
            const n = points.length;
            for (let i = 1; i < n; i++) {
                const alpha = TRAIL_ALPHA * ((i + 1) / n); // newer segments = more opaque
                ctx.strokeStyle = hexToRgba(color, alpha);
                ctx.lineWidth = trailWidth;
                ctx.beginPath();
                ctx.moveTo(points[i - 1].x * w, points[i - 1].y * h);
                ctx.lineTo(points[i].x * w, points[i].y * h);
                ctx.stroke();
            }
        }
    }

    // Boxes
    if (props.showBoxes && detections.value) {
        const dets = detections.value.detections ?? [];
        ctx.lineWidth = Math.max(1.5, Math.round(w / 600));
        ctx.font = `${Math.max(11, Math.round(w / 80))}px system-ui, sans-serif`;
        ctx.textBaseline = "alphabetic";
        for (const d of dets) {
            const [xmin, ymin, xmax, ymax] = d.bbox;
            const x = xmin * w;
            const y = ymin * h;
            const bw = (xmax - xmin) * w;
            const bh = (ymax - ymin) * h;
            const color = colorForTrack(d.tid, d.label);
            ctx.strokeStyle = color;
            ctx.strokeRect(x, y, bw, bh);

            const text = d.tid !== null
                ? `#${d.tid} ${d.label} ${(d.score * 100).toFixed(0)}%`
                : `${d.label} ${(d.score * 100).toFixed(0)}%`;
            const m = ctx.measureText(text);
            const padX = 4;
            const padY = 2;
            const fh = parseInt(ctx.font, 10) || 12;
            const labelW = m.width + padX * 2;
            const labelH = fh + padY * 2;
            const labelY = y > labelH ? y - labelH : y;
            ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
            ctx.fillRect(x, labelY, labelW, labelH);
            ctx.fillStyle = "#fff";
            ctx.fillText(text, x + padX, labelY + fh + padY - 2);
        }

        // Update trail history from this frame's detections.
        const seenTids = new Set<number>();
        for (const d of dets) {
            if (d.tid == null) continue;
            const tid = d.tid;
            seenTids.add(tid);
            const [xmin, ymin, xmax, ymax] = d.bbox;
            const cx = d.center ? d.center[0] : (xmin + xmax) / 2;
            const cy = d.center ? d.center[1] : (ymin + ymax) / 2;
            let points = trailMap.get(tid);
            if (!points) {
                points = [];
                trailMap.set(tid, points);
            }
            points.push({ x: cx, y: cy });
            if (points.length > MAX_TRAIL_LENGTH) points.shift();
            trackMissCounter.set(tid, 0);
        }
        // Age unseen tracks.
        for (const [tid, misses] of trackMissCounter) {
            if (seenTids.has(tid)) continue;
            const next = misses + 1;
            if (next >= STALE_TRACK_GRACE) {
                trailMap.delete(tid);
                trackMissCounter.delete(tid);
            } else {
                trackMissCounter.set(tid, next);
            }
        }
    }

    // Crosshairs (orange current + cyan Kalman lead).
    if (props.showCrosshair && trackerStatus.value?.servo) {
        const s = trackerStatus.value.servo;
        const cur = angleToPx(s.pan, s.tilt, w, h);
        const kal = angleToPx(s.kalmanPan, s.kalmanTilt, w, h);

        const lead = Math.hypot(cur.x - kal.x, cur.y - kal.y);
        if (lead > 3) {
            ctx.strokeStyle = "rgba(180,180,180,0.7)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cur.x, cur.y);
            ctx.lineTo(kal.x, kal.y);
            ctx.stroke();
        }

        drawCrosshair(ctx, kal.x, kal.y, "#00DCFF", 10, false);
        drawCrosshair(ctx, cur.x, cur.y, "#FFC800", 14, true);
    }
}

function angleToPx(pan: number, tilt: number, w: number, h: number) {
    const cxN = pan / SERVO_HFOV + 0.5;
    const cyN = tilt / SERVO_VFOV + 0.5;
    return {
        x: Math.max(0, Math.min(w - 1, cxN * w)),
        y: Math.max(0, Math.min(h - 1, cyN * h)),
    };
}

function drawCrosshair(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
    color: string, radius: number, withCenterCircle: boolean,
) {
    const gap = 4;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y - gap); ctx.lineTo(x, y - radius);
    ctx.moveTo(x, y + gap); ctx.lineTo(x, y + radius);
    ctx.moveTo(x - gap, y); ctx.lineTo(x - radius, y);
    ctx.moveTo(x + gap, y); ctx.lineTo(x + radius, y);
    ctx.stroke();
    if (withCenterCircle) {
        ctx.beginPath();
        ctx.arc(x, y, gap, 0, Math.PI * 2);
        ctx.stroke();
    }
}

function schedule() {
    if (raf !== null) return;
    raf = requestAnimationFrame(draw);
}

// Redraw whenever state changes.
watch(
    [
        detections, trackerStatus, rois, mask,
        () => props.showBoxes, () => props.showCrosshair,
        () => props.showRois, () => props.showMask,
    ],
    schedule,
    { deep: true },
);

onMounted(() => {
    fitCanvas();
    ro = new ResizeObserver(() => {
        fitCanvas();
        schedule();
    });
    if (canvas.value) ro.observe(canvas.value);
    schedule();
});

onBeforeUnmount(() => {
    if (ro) { ro.disconnect(); ro = null; }
    if (raf !== null) { cancelAnimationFrame(raf); raf = null; }
});
</script>

<template>
    <canvas ref="canvas" class="det-overlay" />
</template>

<style scoped>
.det-overlay {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
}
</style>
