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
import CameraFeed from "@/components/CameraFeed.vue";
import Tab from "primevue/tab";
import TabList from "primevue/tablist";
import TabPanel from "primevue/tabpanel";
import TabPanels from "primevue/tabpanels";
import Tabs from "primevue/tabs";
import Toast from "primevue/toast";
import { useToast } from "primevue/usetoast";

const api = new DefaultApi();

const options = ref<PikiOptions>({
    show_boxes: true,
    show_mask: false,
    show_rois: false,
    show_seg: false,
    conf_threshold: 0.4,
    pixelcount_threshold: 500,
    min_area: 500,
    mog2_history: 500,
    mog2_var_threshold: 16,
    denoise_kernelsize: 7,
    mask_transparency: 0.5,
    servo_pid_kp: 1.0,
    servo_pid_ki: 0.0,
    servo_pid_kd: 0.0,
    servo_dead_zone: 1.5,
});

const aimConfig = ref<AimConfigSchema>({
    target_classes: [],
    servo_enabled: false,
    target_lock_duration: 3.0,
    vertical_angle_offset: 0.0,
    pan_invert: false,
    tilt_invert: false,
});

const splashConfig = ref<SplashConfigSchema>({
    enabled: false,
    trigger_classes: [],
    delay_seconds: 0.5,
    duration_seconds: 1.0,
    cooldown_seconds: 10.0,
});

const debugPanel = ref<InstanceType<typeof ServoDebugPanel> | null>(null);
const toast = useToast();
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
        show_boxes: current.show_boxes ?? true,
        show_mask: current.show_mask ?? false,
        show_rois: current.show_rois ?? false,
        show_seg: current.show_seg ?? false,
        conf_threshold: current.conf_threshold ?? 0.4,
        pixelcount_threshold: current.pixelcount_threshold ?? 500,
        min_area: current.min_area ?? 500,
        mog2_history: current.mog2_history ?? 500,
        mog2_var_threshold: current.mog2_var_threshold ?? 16,
        denoise_kernelsize: current.denoise_kernelsize ?? 7,
        mask_transparency: current.mask_transparency ?? 0.5,
        servo_pid_kp: current.servo_pid_kp,
        servo_pid_ki: current.servo_pid_ki,
        servo_pid_kd: current.servo_pid_kd,
        servo_dead_zone: current.servo_dead_zone,
    };

    const aim = await api.coreApiGetAimConfig();
    aimConfig.value = {
        target_classes: aim.target_classes ?? [],
        servo_enabled: aim.servo_enabled ?? false,
        target_lock_duration: aim.target_lock_duration ?? 3.0,
        vertical_angle_offset: aim.vertical_angle_offset ?? 0.0,
        pan_invert: aim.pan_invert ?? false,
        tilt_invert: aim.tilt_invert ?? false,
    };

    const splash = await api.coreApiGetSplashConfig();
    splashConfig.value = {
        enabled: splash.enabled ?? false,
        trigger_classes: splash.trigger_classes ?? [],
        delay_seconds: splash.delay_seconds ?? 0.5,
        duration_seconds: splash.duration_seconds ?? 1.0,
        cooldown_seconds: splash.cooldown_seconds ?? 10.0,
    };
});

watch(options, opts => updateOptions(opts), { deep: true });
watch(aimConfig, cfg => updateAimConfig(cfg), { deep: true });
watch(splashConfig, cfg => updateSplashConfig(cfg), { deep: true });
</script>

<template>
    <div class="page">
        <Toast position="top-right" />
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
                            <button :class="['ot-btn', { active: options.show_boxes }]"
                                @click="options.show_boxes = !options.show_boxes">Boxes</button>
                            <button :class="['ot-btn', { active: options.show_mask }]"
                                @click="options.show_mask = !options.show_mask">Mask</button>
                            <button :class="['ot-btn', { active: options.show_rois }]"
                                @click="options.show_rois = !options.show_rois">ROIs</button>
                            <button :class="['ot-btn', { active: options.show_seg }]"
                                @click="options.show_seg = !options.show_seg">Seg</button>
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
                        <CameraFeed :src="feedUrl" alt="camera feed" />
                        <div class="overlay-toggles">
                            <button :class="['ot-btn', { active: options.show_boxes }]"
                                @click="options.show_boxes = !options.show_boxes">Boxes</button>
                            <button :class="['ot-btn', { active: options.show_mask }]"
                                @click="options.show_mask = !options.show_mask">Mask</button>
                            <button :class="['ot-btn', { active: options.show_rois }]"
                                @click="options.show_rois = !options.show_rois">ROIs</button>
                            <button :class="['ot-btn', { active: options.show_seg }]"
                                @click="options.show_seg = !options.show_seg">Seg</button>
                        </div>
                    </div>
                    <DetectionControls v-model="options" @reset-background="resetBackground()" />
                </TabPanel>

                <TabPanel value="servo">
                    <ServoAimPanel v-model="aimConfig" v-model:options="options" :classes="allClasses ?? []" />
                    <SplashPanel v-model="splashConfig" :classes="allClasses ?? []" />
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
