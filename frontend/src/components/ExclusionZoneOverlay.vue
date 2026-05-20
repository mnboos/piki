<script setup lang="ts">
import { computed, ref } from "vue";
import type { ExclusionZoneSchema } from "@/api";
import {
    useCreateExclusionZoneMutation,
    useDeleteExclusionZoneMutation,
    useExclusionZonesQuery,
    useUpdateExclusionZoneMutation,
} from "@/queries/exclusionZones";

// All coordinates this component handles are in normalized [0, 1] space
// against the camera feed.  The <svg> uses a 0..1 viewBox so we never need
// to know the pixel size of the underlying MJPEG image.

const props = defineProps<{ enabled: boolean }>();

const zonesQuery = useExclusionZonesQuery();
const createMutation = useCreateExclusionZoneMutation();
const updateMutation = useUpdateExclusionZoneMutation();
const deleteMutation = useDeleteExclusionZoneMutation();

interface Rect {
    x: number; // min-x in [0, 1]
    y: number;
    w: number;
    h: number;
}

function pointsToRect(pts: ReadonlyArray<readonly [number, number]> | number[][]): Rect | null {
    if (!pts || pts.length === 0) return null;
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const p of pts) {
        const px = Array.isArray(p) ? p[0] : (p as { 0: number })[0];
        const py = Array.isArray(p) ? p[1] : (p as { 1: number })[1];
        if (typeof px !== "number" || typeof py !== "number") continue;
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
    }
    if (minX >= maxX || minY >= maxY) return null;
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function rectToPoints(r: Rect): [number, number][] {
    const x0 = Math.max(0, Math.min(1, r.x));
    const y0 = Math.max(0, Math.min(1, r.y));
    const x1 = Math.max(0, Math.min(1, r.x + r.w));
    const y1 = Math.max(0, Math.min(1, r.y + r.h));
    return [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ];
}

// Convert a pointer event into normalized [0, 1] coordinates relative to the SVG.
function svgPoint(ev: PointerEvent, svg: SVGSVGElement): { x: number; y: number } {
    const rect = svg.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / rect.width;
    const y = (ev.clientY - rect.top) / rect.height;
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
}

// Active drag state — exactly one of the two is non-null at a time.
type DragNew = { kind: "new"; start: { x: number; y: number }; current: { x: number; y: number } };
type DragMove = { kind: "move"; zoneId: number; startRect: Rect; startPt: { x: number; y: number } };
type DragResize = {
    kind: "resize";
    zoneId: number;
    startRect: Rect;
    startPt: { x: number; y: number };
    handle: "nw" | "ne" | "sw" | "se";
};
type Drag = DragNew | DragMove | DragResize;
const drag = ref<Drag | null>(null);

const svgRef = ref<SVGSVGElement | null>(null);

const zones = computed<ExclusionZoneSchema[]>(() => zonesQuery.data.value ?? []);

const previewRect = computed<Rect | null>(() => {
    const d = drag.value;
    if (!d || d.kind !== "new") return null;
    const x0 = Math.min(d.start.x, d.current.x);
    const y0 = Math.min(d.start.y, d.current.y);
    const x1 = Math.max(d.start.x, d.current.x);
    const y1 = Math.max(d.start.y, d.current.y);
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
});

// Per-zone displayed rect: while a move/resize is in flight, show the live
// preview (avoids the round-trip lag); otherwise show the persisted points.
function displayedRect(z: ExclusionZoneSchema): Rect | null {
    const d = drag.value;
    if (d && d.kind !== "new" && d.zoneId === z.id) {
        return livePreview(d);
    }
    return pointsToRect(z.points as unknown as number[][]);
}

function livePreview(d: DragMove | DragResize): Rect | null {
    if (d.kind === "move") {
        // Use a current pointer position captured during pointermove via drag.value.
        // For simplicity, we shadow the moving zone rect during pointermove inline.
        // (Updated by the pointermove handler below.)
        const moved = (d as DragMove & { current?: { x: number; y: number } }).current;
        if (!moved) return d.startRect;
        const dx = moved.x - d.startPt.x;
        const dy = moved.y - d.startPt.y;
        return {
            x: Math.max(0, Math.min(1 - d.startRect.w, d.startRect.x + dx)),
            y: Math.max(0, Math.min(1 - d.startRect.h, d.startRect.y + dy)),
            w: d.startRect.w,
            h: d.startRect.h,
        };
    }
    // resize
    const cur = (d as DragResize & { current?: { x: number; y: number } }).current ?? d.startPt;
    let x0 = d.startRect.x;
    let y0 = d.startRect.y;
    let x1 = d.startRect.x + d.startRect.w;
    let y1 = d.startRect.y + d.startRect.h;
    if (d.handle.includes("w")) x0 = cur.x;
    if (d.handle.includes("n")) y0 = cur.y;
    if (d.handle.includes("e")) x1 = cur.x;
    if (d.handle.includes("s")) y1 = cur.y;
    const minX = Math.min(x0, x1);
    const minY = Math.min(y0, y1);
    const maxX = Math.max(x0, x1);
    const maxY = Math.max(y0, y1);
    return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function onSvgPointerDown(ev: PointerEvent) {
    if (!props.enabled || ev.button !== 0) return;
    const svg = svgRef.value;
    if (!svg) return;
    // Only start a fresh-rectangle drag when the click landed on the SVG
    // background itself (not on an existing rect/handle).
    if (ev.target !== svg) return;
    const p = svgPoint(ev, svg);
    drag.value = { kind: "new", start: p, current: p };
    svg.setPointerCapture(ev.pointerId);
    ev.preventDefault();
}

function onSvgPointerMove(ev: PointerEvent) {
    const d = drag.value;
    if (!d) return;
    const svg = svgRef.value;
    if (!svg) return;
    const p = svgPoint(ev, svg);
    if (d.kind === "new") {
        d.current = p;
    } else {
        (d as DragMove & { current?: { x: number; y: number } }).current = p;
    }
    // Trigger reactivity.
    drag.value = { ...(d as object) } as Drag;
}

function onSvgPointerUp(ev: PointerEvent) {
    const d = drag.value;
    if (!d) return;
    const svg = svgRef.value;
    svg?.releasePointerCapture(ev.pointerId);
    if (d.kind === "new") {
        const r = previewRect.value;
        drag.value = null;
        if (r && r.w >= 0.02 && r.h >= 0.02) {
            createMutation.mutate({
                name: defaultZoneName(),
                enabled: true,
                points: rectToPoints(r),
            });
        }
        return;
    }
    if (d.kind === "move" || d.kind === "resize") {
        const r = livePreview(d);
        drag.value = null;
        if (r && r.w >= 0.02 && r.h >= 0.02) {
            updateMutation.mutate({
                zoneId: d.zoneId,
                patch: { points: rectToPoints(r) },
            });
        }
    }
}

function defaultZoneName(): string {
    const existing = zones.value.length;
    return `Zone ${existing + 1}`;
}

function startMove(ev: PointerEvent, z: ExclusionZoneSchema) {
    if (!props.enabled || !z.id) return;
    const svg = svgRef.value;
    if (!svg) return;
    const r = pointsToRect(z.points as unknown as number[][]);
    if (!r) return;
    const p = svgPoint(ev, svg);
    drag.value = { kind: "move", zoneId: z.id, startRect: r, startPt: p };
    svg.setPointerCapture(ev.pointerId);
    ev.stopPropagation();
    ev.preventDefault();
}

function startResize(ev: PointerEvent, z: ExclusionZoneSchema, handle: DragResize["handle"]) {
    if (!props.enabled || !z.id) return;
    const svg = svgRef.value;
    if (!svg) return;
    const r = pointsToRect(z.points as unknown as number[][]);
    if (!r) return;
    const p = svgPoint(ev, svg);
    drag.value = { kind: "resize", zoneId: z.id, startRect: r, startPt: p, handle };
    svg.setPointerCapture(ev.pointerId);
    ev.stopPropagation();
    ev.preventDefault();
}

defineExpose({
    createMutation,
    updateMutation,
    deleteMutation,
});
</script>

<template>
    <svg
        ref="svgRef"
        class="ez-svg"
        :class="{ 'ez-svg--interactive': enabled }"
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        @pointerdown="onSvgPointerDown"
        @pointermove="onSvgPointerMove"
        @pointerup="onSvgPointerUp"
        @pointercancel="onSvgPointerUp"
    >
        <!-- Persisted zones -->
        <g v-for="z in zones" :key="z.id ?? -1">
            <template v-if="displayedRect(z) as Rect | null">
                <rect
                    :x="displayedRect(z)!.x"
                    :y="displayedRect(z)!.y"
                    :width="displayedRect(z)!.w"
                    :height="displayedRect(z)!.h"
                    :class="['ez-rect', { 'ez-rect--disabled': !z.enabled }]"
                    @pointerdown="(e) => startMove(e, z)"
                />
                <template v-if="enabled && z.enabled !== false">
                    <rect
                        :x="displayedRect(z)!.x - 0.012"
                        :y="displayedRect(z)!.y - 0.012"
                        width="0.024" height="0.024"
                        class="ez-handle"
                        @pointerdown="(e) => startResize(e, z, 'nw')"
                    />
                    <rect
                        :x="displayedRect(z)!.x + displayedRect(z)!.w - 0.012"
                        :y="displayedRect(z)!.y - 0.012"
                        width="0.024" height="0.024"
                        class="ez-handle"
                        @pointerdown="(e) => startResize(e, z, 'ne')"
                    />
                    <rect
                        :x="displayedRect(z)!.x - 0.012"
                        :y="displayedRect(z)!.y + displayedRect(z)!.h - 0.012"
                        width="0.024" height="0.024"
                        class="ez-handle"
                        @pointerdown="(e) => startResize(e, z, 'sw')"
                    />
                    <rect
                        :x="displayedRect(z)!.x + displayedRect(z)!.w - 0.012"
                        :y="displayedRect(z)!.y + displayedRect(z)!.h - 0.012"
                        width="0.024" height="0.024"
                        class="ez-handle"
                        @pointerdown="(e) => startResize(e, z, 'se')"
                    />
                </template>
            </template>
        </g>

        <!-- In-flight new-rectangle preview -->
        <rect
            v-if="previewRect"
            :x="previewRect.x"
            :y="previewRect.y"
            :width="previewRect.w"
            :height="previewRect.h"
            class="ez-rect ez-rect--preview"
        />
    </svg>
</template>

<style scoped>
.ez-svg {
    width: 100%;
    height: 100%;
    pointer-events: none;
    touch-action: none;
}
.ez-svg--interactive {
    pointer-events: auto;
    cursor: crosshair;
}
.ez-rect {
    fill: rgba(220, 60, 60, 0.25);
    stroke: rgba(220, 60, 60, 0.95);
    stroke-width: 0.004;
    vector-effect: non-scaling-stroke;
    pointer-events: auto;
    cursor: move;
}
.ez-rect--disabled {
    fill: rgba(120, 120, 120, 0.15);
    stroke: rgba(160, 160, 160, 0.7);
    stroke-dasharray: 0.01;
}
.ez-rect--preview {
    fill: rgba(220, 60, 60, 0.15);
    stroke: rgba(220, 60, 60, 0.8);
    stroke-dasharray: 0.012;
    pointer-events: none;
}
.ez-handle {
    fill: white;
    stroke: rgba(220, 60, 60, 0.95);
    stroke-width: 0.003;
    vector-effect: non-scaling-stroke;
    pointer-events: auto;
}
.ez-svg--interactive .ez-handle:hover {
    fill: rgba(255, 200, 200, 1);
}
.ez-svg--interactive .ez-handle {
    cursor: nwse-resize;
}
</style>
