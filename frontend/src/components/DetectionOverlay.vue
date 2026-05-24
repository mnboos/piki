<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from "vue";
import { useDetections, useTrackerStatus } from "@/composables/useEventStream";

const props = withDefaults(defineProps<{
    /** Draw detection bounding boxes. */
    showBoxes?: boolean;
    /** Draw the orange servo crosshair + cyan Kalman lead. */
    showCrosshair?: boolean;
}>(), {
    showBoxes: true,
    showCrosshair: true,
});

// Servo field-of-view (matches `SERVO_HFOV`/`SERVO_VFOV` env defaults in engine.py).
// If those env vars are changed on the server, this would need to be exposed
// via a settings endpoint — for now they're hardcoded to the defaults.
const SERVO_HFOV = 160.0;
const SERVO_VFOV = 100.0;

const canvas = ref<HTMLCanvasElement | null>(null);

const detections = useDetections();
const trackerStatus = useTrackerStatus();

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
watch([detections, trackerStatus, () => props.showBoxes, () => props.showCrosshair], schedule, { deep: true });

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
