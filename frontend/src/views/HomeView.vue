<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { DefaultApi, type AimConfigSchema, type AimConfigSchemaPatch, type PikiOptions, type PikiOptionsPatch } from "@/api";
import DetectionControls from "@/components/DetectionControls.vue";
import ServoAimPanel from "@/components/ServoAimPanel.vue";
import ServoDebugPanel from "@/components/ServoDebugPanel.vue";
import CameraFeed from "@/components/CameraFeed.vue";
import Tab from "primevue/tab";
import TabList from "primevue/tablist";
import TabPanel from "primevue/tabpanel";
import TabPanels from "primevue/tabpanels";
import Tabs from "primevue/tabs";

const api = new DefaultApi();

const options = ref<PikiOptions>({
    mode: "boxes",
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
});

const aimConfig = ref<AimConfigSchema>({
    targetClasses: [],
    servoEnabled: false,
    targetLockDuration: 3.0,
});

const allClasses = ref<string[]>([]);
const debugPanel = ref<InstanceType<typeof ServoDebugPanel> | null>(null);
const feedUrl = "/api/video_feed";
const debugFeedUrl = "/api/video_feed_raw";
// These match the env vars on the server — shown as labels only.
const mainTopic = import.meta.env.VITE_ROS_IMAGE_TOPIC ?? "/image_left_raw";
const debugTopic = import.meta.env.VITE_ROS_DEBUG_TOPIC ?? "/image_right_raw";

const currentFps = ref(0);
let statusInterval: ReturnType<typeof setInterval> | null = null;

const { mutate: updateOptions } = useMutation({
    mutationFn: (opts: PikiOptionsPatch) => api.coreApiUpdateOptions({ pikiOptionsPatch: opts }),
});

const { mutate: resetBackground } = useMutation({
    mutationFn: () => api.coreApiResetBackground(),
});

const { mutate: updateAimConfig } = useMutation({
    mutationFn: (payload: AimConfigSchemaPatch) => api.coreApiUpdateAimConfig({ aimConfigSchemaPatch: payload }),
});

const { mutate: servoMove } = useMutation({
    mutationFn: (payload: { panAngle: number; tiltAngle: number }) =>
        api.coreApiServoMove({ servoMoveSchema: { panAngle: payload.panAngle, tiltAngle: payload.tiltAngle } }),
    onSuccess: data => debugPanel.value?.setResult(data.panAngle, data.tiltAngle),
});

async function pollTrackerStatus() {
    try {
        const s = await api.coreApiGetTrackerStatus();
        currentFps.value = s.fps;
    } catch { /* ignore */ }
}

onMounted(async () => {
    try {
        const current = await api.coreApiGetOptions();
        options.value = {
            mode: current.mode,
            confThreshold: current.confThreshold ?? 0.4,
            pixelcountThreshold: current.pixelcountThreshold ?? 500,
            minArea: current.minArea ?? 500,
            mog2History: current.mog2History ?? 500,
            mog2VarThreshold: current.mog2VarThreshold ?? 16,
            denoiseKernelsize: current.denoiseKernelsize ?? 7,
            maskTransparency: current.maskTransparency ?? 0.5,
        };
    } catch { /* use defaults */ }

    try {
        allClasses.value = await api.coreApiGetYoloClasses();
    } catch { /* ignore */ }

    try {
        const aim = await api.coreApiGetAimConfig();
        aimConfig.value = { targetClasses: aim.targetClasses, servoEnabled: aim.servoEnabled, targetLockDuration: aim.targetLockDuration ?? 3.0 };
    } catch { /* ignore */ }

    await pollTrackerStatus();
    statusInterval = setInterval(pollTrackerStatus, 1500);
});

onUnmounted(() => {
    if (statusInterval !== null) clearInterval(statusInterval);
});

watch(options, opts => updateOptions(opts), { deep: true });
watch(aimConfig, cfg => updateAimConfig(cfg), { deep: true });
</script>

<template>
    <div class="page">
        <Tabs value="camera">
            <TabList>
                <Tab value="camera">Camera</Tab>
                <Tab value="detection">Detection</Tab>
                <Tab value="servo">Servo</Tab>
                <Tab value="debug">Debug</Tab>
            </TabList>
            <TabPanels>
                <TabPanel value="camera">
                    <div class="feed-wrapper">
                        <CameraFeed :src="feedUrl" alt="camera feed" />
                        <div class="fps-badge">{{ currentFps.toFixed(1) }} FPS</div>
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
                        <CameraFeed :src="feedUrl" alt="camera feed" />
                    </div>
                    <DetectionControls v-model="options" @reset-background="resetBackground()" />
                </TabPanel>

                <TabPanel value="servo">
                    <ServoAimPanel v-model="aimConfig" v-model:options="options" :classes="allClasses" />
                    <ServoDebugPanel ref="debugPanel" @move="(pan, tilt) => servoMove({ panAngle: pan, tiltAngle: tilt })" />
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
