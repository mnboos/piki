<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { DefaultApi, type AimConfigSchema, type AimConfigSchemaPatch, type PikiOptions, type PikiOptionsPatch, type SplashConfigSchema, type SplashConfigSchemaPatch } from "@/api";
import { useYoloClassesQuery } from "@/queries/recordings";
import { useLastNewEventClip } from "@/composables/useEventStream";
import DetectionControls from "@/components/DetectionControls.vue";
import ServoAimPanel from "@/components/ServoAimPanel.vue";
import RecordingsPanel from "@/components/RecordingsPanel.vue";
import ServoDebugPanel from "@/components/ServoDebugPanel.vue";
import SplashPanel from "@/components/SplashPanel.vue";
import SplashStatusIndicator from "@/components/SplashStatusIndicator.vue";
import GamepadStatusIndicator from "@/components/GamepadStatusIndicator.vue";
import CameraFeed from "@/components/CameraFeed.vue";
import DetectionOverlay from "@/components/DetectionOverlay.vue";
import ExclusionZoneOverlay from "@/components/ExclusionZoneOverlay.vue";
import ExclusionZonesPanel from "@/components/ExclusionZonesPanel.vue";
import MetricsPanel from "@/components/MetricsPanel.vue";
import Select from "primevue/select";
import Tab from "primevue/tab";
import TabList from "primevue/tablist";
import TabPanel from "primevue/tabpanel";
import TabPanels from "primevue/tabpanels";
import Tabs from "primevue/tabs";
import Toast from "primevue/toast";
import { useToast } from "primevue/usetoast";

const api = new DefaultApi();

const options = ref<PikiOptions>({
    showBoxes: true,
    confThreshold: 0.4,
    pixelcountThreshold: 500,
    minArea: 500,
    mog2History: 500,
    mog2VarThreshold: 16,
    denoiseKernelsize: 7,
    servoPidKp: 1.0,
    servoPidKi: 0.0,
    servoPidKd: 0.0,
    servoDeadZone: 1.5,
    servoKalmanProcessNoise: 10.0,
    servoKalmanMeasNoise: 5.0,
});

// Client-only overlay toggles — these are not persisted server-side
// because the mask + ROI overlays are pure visualizations.
const showMask = ref(false);
const showRois = ref(false);
const showSegMasks = ref(false);

const aimConfig = ref<AimConfigSchema>({
    targetClasses: [],
    servoEnabled: false,
    targetLockDuration: 3.0,
    aimConfidence: 0.4,
    verticalAngleOffset: 0.0,
    panInvert: false,
    tiltInvert: false,
});

const splashConfig = ref<SplashConfigSchema>({
    enabled: false,
    delaySeconds: 0.5,
    durationSeconds: 1.0,
    cooldownSeconds: 10.0,
    pumpDuty: 100.0,
});

const debugPanel = ref<InstanceType<typeof ServoDebugPanel> | null>(null);
const toast = useToast();
const editingZones = ref(false);

// WebRTC stream FPS — read/write the backend setting; in-memory only there.
const FPS_OPTIONS = [5, 10, 15, 20, 30];
const targetFps = ref<number>(30);
const { mutate: setTargetFps } = useMutation({
    mutationFn: (fps: number) => api.coreApiUpdateWebrtcConfig({
        webRtcConfigSchemaPatch: { targetFps: fps },
    }),
    onSuccess: (data) => { targetFps.value = data.targetFps; },
});

// Actual streaming FPS measured on the client via requestVideoFrameCallback.
// Null when the video element hasn't painted yet (first ~1s after connect).
const streamingFps = ref<number | null>(null);

function onStreamingFpsUpdate(fps: number) {
    streamingFps.value = fps;
}

// ── Live state (WebSocket) ────────────────────────────────────────────────

const lastNewClip = useLastNewEventClip();
watch(lastNewClip, c => {
  if (!c) return;
  toast.add({
    severity: "info",
    summary: "Event clip saved",
    detail: `${c.filename} (${c.frameCount} frames)`,
    life: 8000,
  });
});

const { data: allClasses } = useYoloClassesQuery();

// ── Mutations ─────────────────────────────────────────────────────────────

const { mutate: updateOptions } = useMutation({
    mutationFn: (opts: PikiOptionsPatch) => api.coreApiUpdateOptions({ pikiOptionsPatch: opts }),
});

const { mutate: resetBackground } = useMutation({
    mutationFn: () => api.coreApiResetBackground(),
});

const { mutate: updateAimConfig } = useMutation({
    mutationFn: (payload: AimConfigSchemaPatch) => api.coreApiUpdateAimConfig({ aimConfigSchemaPatch: payload }),
});

const { mutate: updateSplashConfig } = useMutation({
    mutationFn: (payload: SplashConfigSchemaPatch) => api.coreApiUpdateSplashConfig({ splashConfigSchemaPatch: payload }),
});

const { mutate: firePump, isPending: firePumpPending } = useMutation({
    mutationFn: () => api.coreApiActivateSplash(),
});

const { mutate: servoMove } = useMutation({
    mutationFn: (payload: { panAngle: number; tiltAngle: number }) =>
        api.coreApiServoMove({ servoMoveSchema: { panAngle: payload.panAngle, tiltAngle: payload.tiltAngle } }),
    onSuccess: data => debugPanel.value?.setResult(data.panAngle, data.tiltAngle),
});

// ── Initial data load into mutable refs (needed for v-model) ──────────────

onMounted(async () => {
    try {
        const cfg = await api.coreApiGetWebrtcConfig();
        targetFps.value = cfg.targetFps;
    } catch { /* ignore — endpoint may be unavailable during reload */ }

    const current = await api.coreApiGetOptions();
    options.value = {
        showBoxes: current.showBoxes ?? true,
        confThreshold: current.confThreshold ?? 0.4,
        pixelcountThreshold: current.pixelcountThreshold ?? 500,
        minArea: current.minArea ?? 500,
        mog2History: current.mog2History ?? 500,
        mog2VarThreshold: current.mog2VarThreshold ?? 16,
        denoiseKernelsize: current.denoiseKernelsize ?? 7,
        servoPidKp: current.servoPidKp,
        servoPidKi: current.servoPidKi,
        servoPidKd: current.servoPidKd,
        servoDeadZone: current.servoDeadZone,
        servoKalmanProcessNoise: current.servoKalmanProcessNoise ?? 10.0,
        servoKalmanMeasNoise: current.servoKalmanMeasNoise ?? 5.0,
    };

    const aim = await api.coreApiGetAimConfig();
    aimConfig.value = {
        targetClasses: aim.targetClasses ?? [],
        servoEnabled: aim.servoEnabled ?? false,
        targetLockDuration: aim.targetLockDuration ?? 3.0,
        aimConfidence: aim.aimConfidence ?? 0.4,
        verticalAngleOffset: aim.verticalAngleOffset ?? 0.0,
        panInvert: aim.panInvert ?? false,
        tiltInvert: aim.tiltInvert ?? false,
    };

    const splash = await api.coreApiGetSplashConfig();
    splashConfig.value = {
        enabled: splash.enabled ?? false,
        delaySeconds: splash.delaySeconds ?? 0.5,
        durationSeconds: splash.durationSeconds ?? 1.0,
        cooldownSeconds: splash.cooldownSeconds ?? 10.0,
        pumpDuty: splash.pumpDuty ?? 100.0,
    };
});

watch(options, opts => updateOptions(opts), { deep: true });
watch(aimConfig, cfg => updateAimConfig(cfg), { deep: true });
watch(splashConfig, cfg => updateSplashConfig(cfg), { deep: true });
</script>

<template>
    <div class="page">
        <Toast position="top-right" />
        <div class="top-bar">
            <SplashStatusIndicator />
            <GamepadStatusIndicator />
        </div>
        <Tabs value="camera" lazy>
            <TabList>
                <Tab value="camera">Camera</Tab>
                <Tab value="detection">Detection</Tab>
                <Tab value="servo">Servo</Tab>
                <Tab value="metrics">Metrics</Tab>
                <Tab value="recordings">Recordings</Tab>
            </TabList>
            <TabPanels>
                <TabPanel value="camera">
                    <div class="feed-wrapper">
                        <CameraFeed alt="camera feed" @fps-update="onStreamingFpsUpdate">
                            <template #overlay>
                                <DetectionOverlay
                                    :show-boxes="options.showBoxes"
                                    :show-mask="showMask"
                                    :show-seg-masks="showSegMasks"
                                    :show-rois="showRois"
                                />
                            </template>
                        </CameraFeed>
                        <div class="overlay-toggles">
                            <button :class="['ot-btn', { active: options.showBoxes }]"
                                @click="options.showBoxes = !options.showBoxes">Boxes</button>
                            <button :class="['ot-btn', { active: showSegMasks }]"
                                @click="showSegMasks = !showSegMasks">Seg</button>
                            <button :class="['ot-btn', { active: showMask }]"
                                @click="showMask = !showMask">Mask</button>
                            <button :class="['ot-btn', { active: showRois }]"
                                @click="showRois = !showRois">ROIs</button>
                        </div>
                    </div>
                    <div class="stream-controls">
                        <label class="stream-fps-label">
                            Stream FPS
                            <Select
                                :model-value="targetFps"
                                :options="FPS_OPTIONS"
                                @update:model-value="(v: number) => setTargetFps(v)"
                                class="stream-fps-select"
                            />
                        </label>
                        <span class="stream-fps-actual">
                            actual: {{ streamingFps !== null && streamingFps > 0 ? streamingFps.toFixed(1) : '--' }} fps
                        </span>
                    </div>
                </TabPanel>

                <TabPanel value="detection">
                    <div class="feed-wrapper">
                        <CameraFeed alt="camera feed">
                            <template #overlay>
                                <DetectionOverlay
                                    :show-boxes="options.showBoxes"
                                    :show-mask="showMask"
                                    :show-seg-masks="showSegMasks"
                                    :show-rois="showRois"
                                />
                                <ExclusionZoneOverlay :enabled="editingZones" />
                            </template>
                        </CameraFeed>
                        <div class="overlay-toggles">
                            <button :class="['ot-btn', { active: options.showBoxes }]"
                                @click="options.showBoxes = !options.showBoxes">Boxes</button>
                            <button :class="['ot-btn', { active: showSegMasks }]"
                                @click="showSegMasks = !showSegMasks">Seg</button>
                            <button :class="['ot-btn', { active: showMask }]"
                                @click="showMask = !showMask">Mask</button>
                            <button :class="['ot-btn', { active: showRois }]"
                                @click="showRois = !showRois">ROIs</button>
                            <button :class="['ot-btn', { active: editingZones }]"
                                @click="editingZones = !editingZones">Zones</button>
                        </div>
                    </div>
                    <ExclusionZonesPanel v-model="editingZones" />
                    <DetectionControls v-model="options" @reset-background="resetBackground()" />
                </TabPanel>

                <TabPanel value="servo">
                    <ServoAimPanel v-model="aimConfig" v-model:options="options" :classes="allClasses ?? []" />
                    <SplashPanel v-model="splashConfig" :fire-pump="firePump" :fire-pump-pending="firePumpPending" />
                    <ServoDebugPanel ref="debugPanel" @move="(pan, tilt) => servoMove({ panAngle: pan, tiltAngle: tilt })" />
                </TabPanel>

                <TabPanel value="metrics">
                    <MetricsPanel />
                </TabPanel>

                <TabPanel value="recordings">
                    <RecordingsPanel />
                </TabPanel>
            </TabPanels>
        </Tabs>
    </div>
</template>

<style scoped>
.page {
    padding: 1rem;
    max-width: 900px;
    margin: 0 auto;
}
.top-bar {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.5rem;
}
.feed-wrapper {
    position: relative;
    border: 1px solid #ff000044;
    border-radius: 5px;
    overflow: hidden;
    line-height: 0;
}
.stream-controls {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0.5rem;
    font-family: monospace;
}
.stream-fps-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: var(--p-text-color, #ddd);
}
.stream-fps-select {
    min-width: 80px;
    font-size: 0.8rem;
}
.stream-fps-select :deep(.p-select-label) {
    padding: 0.25rem 0.5rem;
    line-height: 1.2;
}
.stream-fps-actual {
    font-size: 0.75rem;
    color: var(--p-text-muted-color, #888);
    letter-spacing: 0.03em;
}
.overlay-toggles {
    position: absolute;
    bottom: 0.5rem;
    left: 0.5rem;
    display: flex;
    gap: 0.35rem;
}
.ot-btn {
    padding: 0.25rem 0.55rem;
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.5);
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.7rem;
    font-family: monospace;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    line-height: 1.3;
}
.ot-btn:hover {
    background: rgba(0, 0, 0, 0.7);
    color: rgba(255, 255, 255, 0.9);
}
.ot-btn.active {
    background: rgba(0, 200, 255, 0.25);
    border-color: rgba(0, 200, 255, 0.6);
    color: #fff;
}
</style>
