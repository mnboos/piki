import { reactive, ref, computed, watch } from "vue";
import { useWebSocket } from "@vueuse/core";
import { useBackendHost } from "@/utils";
import type {
    SystemStatus,
    SplashStatus,
    RecordingStatus,
    ReplayStatus,
    EventClipSchema,
} from "@/api";

export interface DetectionEntry {
    tid: number | null;
    label: string;
    score: number;
    /** Normalized [xmin, ymin, xmax, ymax]. */
    bbox: [number, number, number, number];
    /** Normalized [cx, cy] mask centroid. */
    center?: [number, number];
    /** Per-detection instance mask contours: each inner array is [x1,y1,x2,y2,...] normalized. */
    maskPolygon?: number[][];
}

export interface DetectionsPayload {
    frameTsNs: number;
    detections: DetectionEntry[];
}

export interface RoisPayload {
    /** Each entry: [x, y, w, h] normalized to the lores frame [0, 1]. */
    rois: [number, number, number, number][];
}

export interface MaskPayload {
    /** Each polygon: flat [x0, y0, x1, y1, ...] in normalized [0, 1] coords. */
    polygons: number[][];
}

export interface PipelineFpsPayload {
    fps: number;
}

export interface GamepadStatus {
    connected: boolean;
    enabled: boolean;
    pan: number;
    tilt: number;
}

interface State {
    tracker_status: SystemStatus | null;
    pipeline_fps: PipelineFpsPayload;
    splash_status: SplashStatus | null;
    recording_status: RecordingStatus | null;
    replay_status: ReplayStatus | null;
    event_clips: EventClipSchema[];
    detections: DetectionsPayload;
    rois: RoisPayload;
    mask: MaskPayload;
    gamepad_status: GamepadStatus | null;
}

const state = reactive<State>({
    tracker_status: null,
    pipeline_fps: { fps: 0 },
    splash_status: null,
    recording_status: null,
    replay_status: null,
    event_clips: [],
    detections: { frameTsNs: 0, detections: [] },
    rois: { rois: [] },
    mask: { polygons: [] },
    gamepad_status: null,
});

const seenClipKeys = new Set<string>();
const lastNewClip = ref<EventClipSchema | null>(null);

function snakeToCamel<T = unknown>(value: unknown): T {
    if (Array.isArray(value)) return value.map(snakeToCamel) as unknown as T;
    if (value === null || typeof value !== "object") return value as T;
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        const camelKey = k.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
        out[camelKey] = snakeToCamel(v);
    }
    return out as T;
}

// ---------------------------------------------------------------------------
// Local splash state machine — backend publishes only on transitions;
// the frontend counts down locally and auto-advances through the chain
// (armed → firing → cooldown → idle).  A new event from the backend
// cancels the local chain and restarts from the new state.
// ---------------------------------------------------------------------------

let splashAdvanceTimeout: ReturnType<typeof setTimeout> | null = null;
let splashCountdownInterval: ReturnType<typeof setInterval> | null = null;
const splashReceivedAt = ref(0);
const splashTotalDuration = ref(0);
const splashDisplayRemaining = ref(0);

function clearSplashTimers() {
    if (splashAdvanceTimeout !== null) {
        clearTimeout(splashAdvanceTimeout);
        splashAdvanceTimeout = null;
    }
    if (splashCountdownInterval !== null) {
        clearInterval(splashCountdownInterval);
        splashCountdownInterval = null;
    }
}

function startSplashCountdown(duration: number) {
    splashReceivedAt.value = Date.now();
    splashTotalDuration.value = duration;
    splashDisplayRemaining.value = Math.round(duration * 10) / 10;

    if (splashCountdownInterval) clearInterval(splashCountdownInterval);
    splashCountdownInterval = setInterval(() => {
        const elapsed = (Date.now() - splashReceivedAt.value) / 1000;
        const remaining = Math.max(0, splashTotalDuration.value - elapsed);
        splashDisplayRemaining.value = Math.round(remaining * 10) / 10;
    }, 100);
}

function advanceSplashChain(current: Record<string, unknown>) {
    const st = current.state as string;

    if (st === "armed") {
        const fd = (current.firingDuration as number) || 1.0;
        const cd = (current.cooldownDuration as number) || 10.0;
        applySplashPayload({
            state: "firing",
            delayRemaining: 0,
            firingRemaining: fd,
            cooldownRemaining: 0,
            firingDuration: 0,
            cooldownDuration: cd,
            enabled: current.enabled,
        });
    } else if (st === "firing") {
        const cd = (current.cooldownDuration as number) || 10.0;
        applySplashPayload({
            state: "cooldown",
            delayRemaining: 0,
            firingRemaining: 0,
            cooldownRemaining: cd,
            firingDuration: 0,
            cooldownDuration: 0,
            enabled: current.enabled,
        });
    } else if (st === "cooldown") {
        applySplashPayload({
            state: "idle",
            delayRemaining: 0,
            firingRemaining: 0,
            cooldownRemaining: 0,
            firingDuration: 0,
            cooldownDuration: 0,
            enabled: current.enabled,
        });
    }
    // idle: terminal — nothing to auto-advance to
}

function applySplashPayload(payload: Record<string, unknown>) {
    clearSplashTimers();
    state.splash_status = payload as unknown as SplashStatus;

    const st = payload.state as string;
    let duration = 0;
    if (st === "armed") duration = (payload.delayRemaining as number) || 0;
    else if (st === "firing") duration = (payload.firingRemaining as number) || 0;
    else if (st === "cooldown") duration = (payload.cooldownRemaining as number) || 0;

    if (duration > 0) {
        startSplashCountdown(duration);
        splashAdvanceTimeout = setTimeout(() => {
            splashAdvanceTimeout = null;
            advanceSplashChain(payload);
        }, duration * 1000);
    }
}

// ---------------------------------------------------------------------------
// WebSocket connection
// ---------------------------------------------------------------------------

const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
const wsUrl = useBackendHost(wsProtocol) + "/ws/events";

const { data, status } = useWebSocket(wsUrl, {
    autoReconnect: { retries: -1, delay: 1000 },
    heartbeat: { interval: 30_000, message: "ping", responseMessage: "pong" },
    immediate: true,
});

watch(data, raw => {
    if (typeof raw !== "string" || !raw) return;
    let env: { topic: string; payload: unknown; ts?: number };
    try {
        env = JSON.parse(raw);
    } catch {
        return;
    }
    const payload = snakeToCamel<Record<string, unknown>>(env.payload);

    switch (env.topic) {
        case "tracker_status":
            state.tracker_status = payload as unknown as SystemStatus;
            break;
        case "pipeline_fps":
            state.pipeline_fps = payload as unknown as PipelineFpsPayload;
            break;
        case "splash_status":
            applySplashPayload(payload as unknown as Record<string, unknown>);
            break;
        case "recording_status":
            state.recording_status = payload as unknown as RecordingStatus;
            break;
        case "replay_status":
            state.replay_status = payload as unknown as ReplayStatus;
            break;
        case "event_clips": {
            const p = payload as { clips?: EventClipSchema[]; clip?: EventClipSchema };
            if (p.clips) {
                state.event_clips = p.clips;
                seenClipKeys.clear();
                for (const c of p.clips) seenClipKeys.add(c.file!);
            } else if (p.clip && !seenClipKeys.has(p.clip.file!)) {
                seenClipKeys.add(p.clip.file!);
                state.event_clips.unshift(p.clip);
                lastNewClip.value = p.clip;
            }
            break;
        }
        case "detections":
            state.detections = payload as unknown as DetectionsPayload;
            break;
        case "rois":
            state.rois = payload as unknown as RoisPayload;
            break;
        case "mask":
            state.mask = payload as unknown as MaskPayload;
            break;
        case "gamepad_status":
            state.gamepad_status = payload as unknown as GamepadStatus;
            break;
    }
});

export const useTrackerStatus = () => computed(() => state.tracker_status);
export const usePipelineFps = () => computed(() => state.pipeline_fps.fps);
export const useDetections = () => computed(() => state.detections);
export const useRois = () => computed(() => state.rois);
export const useMask = () => computed(() => state.mask);
export const useSplashStatus = () => computed(() => state.splash_status);
export const useSplashDisplayRemaining = () => computed(() => splashDisplayRemaining.value);
export const useRecordingStatus = () => computed(() => state.recording_status);
export const useReplayStatus = () => computed(() => state.replay_status);
export const useEventClips = () => computed(() => state.event_clips);
export const useLastNewEventClip = () => computed(() => lastNewClip.value);
export const useGamepadStatus = () => computed(() => state.gamepad_status);
export const useWsStatus = () => status;
