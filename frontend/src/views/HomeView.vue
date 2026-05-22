<script setup lang="ts">
import { ref, watch, watchEffect, onMounted } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { DefaultApi, type AimConfigSchema, type AimConfigSchemaPatch, type PikiOptions, type PikiOptionsPatch, type SplashConfigSchema, type SplashConfigSchemaPatch } from "@/api";
import { useTrackerStatusQuery, useEventClipsQuery, useYoloClassesQuery } from "@/queries/recordings";
import DetectionControls from "@/components/DetectionControls.vue";
import ServoAimPanel from "@/components/ServoAimPanel.vue";
import RecordingsPanel from "@/components/RecordingsPanel.vue";
import ServoDebugPanel from "@/components/ServoDebugPanel.vue";
import SplashPanel from "@/components/SplashPanel.vue";
import SplashStatusIndicator from "@/components/SplashStatusIndicator.vue";
import CameraFeed from "@/components/CameraFeed.vue";
import ExclusionZoneOverlay from "@/components/ExclusionZoneOverlay.vue";
import ExclusionZonesPanel from "@/components/ExclusionZonesPanel.vue";
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
    showMask: false,
    showRois: false,
    confThreshold: 0.4,
    pixelcountThreshold: 500,
    minArea: 500,
    mog2History: 500,
    mog2VarThreshold: 16,
    denoiseKernelsize: 7,
    maskTransparency: 0.5,
    servoPidKp: 1.0,
    servoPidKi: 0.0,
    servoPidKd: 0.0,
    servoDeadZone: 1.5,
    servoKalmanProcessNoise: 10.0,
    servoKalmanMeasNoise: 5.0,
});

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
const feedUrl = "/api/video_feed";
const debugFeedUrl = "/api/video_feed_raw";
const mainTopic = import.meta.env.VITE_ROS_IMAGE_TOPIC ?? "/image_left_raw";
const debugTopic = import.meta.env.VITE_ROS_DEBUG_TOPIC ?? "/image_right_raw";

// ── Queries ───────────────────────────────────────────────────────────────

const { data: trackerStatus } = useTrackerStatusQuery();
const currentFps = ref(0);
watchEffect(() => {
  currentFps.value = trackerStatus.value?.fps ?? 0;
});

const { data: eventClips } = useEventClipsQuery();
const toastedFiles = new Set<string>();
watch(eventClips, (clips) => {
  if (!clips) return;
  for (const c of clips) {
    if (!toastedFiles.has(c.file)) {
      toastedFiles.add(c.file);
      toast.add({
        severity: "info",
        summary: "Event clip saved",
        detail: `${c.filename} (${c.frameCount} frames)`,
        life: 8000,
      });
    }
  }
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

const { mutate: servoMove } = useMutation({
    mutationFn: (payload: { panAngle: number; tiltAngle: number }) =>
        api.coreApiServoMove({ servoMoveSchema: { panAngle: payload.panAngle, tiltAngle: payload.tiltAngle } }),
    onSuccess: data => debugPanel.value?.setResult(data.panAngle, data.tiltAngle),
});

// ── Initial data load into mutable refs (needed for v-model) ──────────────

onMounted(async () => {
    const current = await api.coreApiGetOptions();
    options.value = {
        showBoxes: current.showBoxes ?? true,
        showMask: current.showMask ?? false,
        showRois: current.showRois ?? false,
        confThreshold: current.confThreshold ?? 0.4,
        pixelcountThreshold: current.pixelcountThreshold ?? 500,
        minArea: current.minArea ?? 500,
        mog2History: current.mog2History ?? 500,
        mog2VarThreshold: current.mog2VarThreshold ?? 16,
        denoiseKernelsize: current.denoiseKernelsize ?? 7,
        maskTransparency: current.maskTransparency ?? 0.5,
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
        </div>
        <Tabs value="camera">
            <TabList>
                <Tab value="camera">Camera</Tab>
                <Tab value="detection">Detection</Tab>
                <Tab value="servo">Servo</Tab>
                <Tab value="debug">Debug</Tab>
                <Tab value="recordings">Recordings</Tab>
            </TabList>
            <TabPanels>
                <TabPanel value="camera">
                    <div class="feed-wrapper">
                        <CameraFeed :src="feedUrl" alt="camera feed" />
                        <div class="fps-badge">{{ currentFps.toFixed(1) }} FPS</div>
                        <div class="overlay-toggles">
                            <button :class="['ot-btn', { active: options.showBoxes }]"
                                @click="options.showBoxes = !options.showBoxes">Boxes</button>
                            <button :class="['ot-btn', { active: options.showMask }]"
                                @click="options.showMask = !options.showMask">Mask</button>
                            <button :class="['ot-btn', { active: options.showRois }]"
                                @click="options.showRois = !options.showRois">ROIs</button>
                        </div>
                    </div>
                </TabPanel>

                <TabPanel value="debug">
                    <div class="debug-feeds">
                        <div class="debug-feed-item">
                            <p class="feed-label">{{ mainTopic }}</p>
                            <CameraFeed :src="feedUrl" alt="main feed" />
                        </div>
                        <div class="debug-feed-item">
                            <p class="feed-label">{{ debugTopic }}</p>
                            <CameraFeed :src="debugFeedUrl" alt="raw feed" />
                        </div>
                    </div>
                </TabPanel>

                <TabPanel value="detection">
                    <div class="feed-wrapper">
                        <CameraFeed :src="feedUrl" alt="camera feed">
                            <template #overlay>
                                <ExclusionZoneOverlay :enabled="editingZones" />
                            </template>
                        </CameraFeed>
                        <div class="overlay-toggles">
                            <button :class="['ot-btn', { active: options.showBoxes }]"
                                @click="options.showBoxes = !options.showBoxes">Boxes</button>
                            <button :class="['ot-btn', { active: options.showMask }]"
                                @click="options.showMask = !options.showMask">Mask</button>
                            <button :class="['ot-btn', { active: options.showRois }]"
                                @click="options.showRois = !options.showRois">ROIs</button>
                            <button :class="['ot-btn', { active: editingZones }]"
                                @click="editingZones = !editingZones">Zones</button>
                        </div>
                    </div>
                    <ExclusionZonesPanel v-model="editingZones" />
                    <DetectionControls v-model="options" @reset-background="resetBackground()" />
                </TabPanel>

                <TabPanel value="servo">
                    <ServoAimPanel v-model="aimConfig" v-model:options="options" :classes="allClasses ?? []" />
                    <SplashPanel v-model="splashConfig" />
                    <ServoDebugPanel ref="debugPanel" @move="(pan, tilt) => servoMove({ panAngle: pan, tiltAngle: tilt })" />
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
.fps-badge {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: rgba(0, 0, 0, 0.55);
    color: #aaa;
    pointer-events: none;
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
.debug-feeds {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.debug-feed-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.feed-label {
    font-size: 0.75rem;
    font-family: monospace;
    color: var(--p-text-muted-color, #888);
    margin: 0;
}
</style>
