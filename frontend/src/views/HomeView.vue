<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { DefaultApi, type AimConfigSchema, type AimConfigSchemaPatch, type PikiOptions, type PikiOptionsPatch } from "@/api";
import DetectionControls from "@/components/DetectionControls.vue";
import ServoAimPanel from "@/components/ServoAimPanel.vue";
import ServoDebugPanel from "@/components/ServoDebugPanel.vue";

const api = new DefaultApi();

const options = ref<PikiOptions>({
    mode: "boxes",
    confThreshold: 0.5,
    pixelcountThreshold: 500,
    minArea: 500,
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
            confThreshold: current.confThreshold ?? 0.5,
            pixelcountThreshold: current.pixelcountThreshold ?? 500,
            minArea: current.minArea ?? 500,
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
        <DetectionControls v-model="options" @reset-background="resetBackground()" />

        <div class="feed-wrapper">
            <img id="camera-feed" :src="feedUrl" alt="camera feed" />
        </div>

        <ServoAimPanel v-model="aimConfig" :classes="allClasses" />

        <ServoDebugPanel ref="debugPanel" @move="(pan, tilt) => servoMove({ panAngle: pan, tiltAngle: tilt })" />
    </div>
</template>

<style scoped>
.page {
    display: flex;
    flex-direction: column;
    gap: 1rem;
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
#camera-feed {
    width: 100%;
    display: block;
}
</style>
