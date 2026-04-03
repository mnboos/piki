<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { DefaultApi, type AimConfigSchema, type AimConfigSchemaPatch, type PikiOptions, type PikiOptionsPatch } from "@/api";
import DetectionControls from "@/components/DetectionControls.vue";
import ServoAimPanel from "@/components/ServoAimPanel.vue";
import ServoDebugPanel from "@/components/ServoDebugPanel.vue";
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
});

const aimConfig = ref<AimConfigSchema>({
    targetClasses: [],
    servoEnabled: false,
});

const allClasses = ref<string[]>([]);
const debugPanel = ref<InstanceType<typeof ServoDebugPanel> | null>(null);
const feedUrl = "/api/video_feed";

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
        aimConfig.value = { targetClasses: aim.targetClasses, servoEnabled: aim.servoEnabled };
    } catch { /* ignore */ }
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
            </TabList>
            <TabPanels>
                <TabPanel value="camera">
                    <div class="feed-wrapper">
                        <img id="camera-feed" :src="feedUrl" alt="camera feed" />
                    </div>
                </TabPanel>

                <TabPanel value="detection">
                    <div class="feed-wrapper">
                        <img id="camera-feed-detection" :src="feedUrl" alt="camera feed" />
                    </div>
                    <DetectionControls v-model="options" @reset-background="resetBackground()" />
                </TabPanel>

                <TabPanel value="servo">
                    <ServoAimPanel v-model="aimConfig" :classes="allClasses" />
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
    border: 1px solid #ff000044;
    border-radius: 5px;
    overflow: hidden;
    line-height: 0;
}
#camera-feed,
#camera-feed-detection {
    width: 100%;
    display: block;
}
</style>
