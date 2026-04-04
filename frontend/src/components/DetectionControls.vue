<script setup lang="ts">
import { computed } from "vue";
import { type PikiOptions } from "@/api";
import Button from "primevue/button";
import Fieldset from "primevue/fieldset";
import Select from "primevue/select";
import Slider from "primevue/slider";
import ToggleSwitch from "primevue/toggleswitch";

const model = defineModel<PikiOptions>({ required: true });
const emit = defineEmits<{ resetBackground: [] }>();

// Motion detection
const mog2History = computed({
    get: () => model.value.mog2History ?? 500,
    set: (v: number) => { model.value.mog2History = v; },
});
const mog2VarThreshold = computed({
    get: () => model.value.mog2VarThreshold ?? 16,
    set: (v: number) => { model.value.mog2VarThreshold = v; },
});
const denoiseKernelsize = computed({
    get: () => model.value.denoiseKernelsize ?? 7,
    set: (v: number) => { model.value.denoiseKernelsize = v; },
});
const pixelcountThreshold = computed({
    get: () => model.value.pixelcountThreshold ?? 500,
    set: (v: number) => { model.value.pixelcountThreshold = v; },
});
const minArea = computed({
    get: () => model.value.minArea ?? 500,
    set: (v: number) => { model.value.minArea = v; },
});

// Object detection
const confThreshold = computed({
    get: () => model.value.confThreshold ?? 0.4,
    set: (v: number) => { model.value.confThreshold = v; },
});

const trackerTypeOptions = [
    { name: "CSRT (accurate)", value: "CSRT" },
    { name: "KCF (fast)", value: "KCF" },
];

const trackingEnabled = computed({
    get: () => model.value.trackingEnabled ?? true,
    set: (v: boolean) => { model.value.trackingEnabled = v; },
});

const trackerLostThreshold = computed({
    get: () => model.value.trackerLostThreshold ?? 5,
    set: (v: number) => { model.value.trackerLostThreshold = v; },
});

// Servo smoothing
const servoSmoothFactor = computed({
    get: () => model.value.servoSmoothFactor ?? 0.3,
    set: (v: number) => { model.value.servoSmoothFactor = v; },
});
const servoDeadZone = computed({
    get: () => model.value.servoDeadZone ?? 1.5,
    set: (v: number) => { model.value.servoDeadZone = v; },
});

// Display
const maskTransparency = computed({
    get: () => model.value.maskTransparency ?? 0.5,
    set: (v: number) => { model.value.maskTransparency = v; },
});

const modeOptions = [
    { name: "Boxes", value: "boxes" },
    { name: "Mask", value: "mask" },
    { name: "ROIs", value: "rois" },
];

const kernelOptions = [1, 3, 5, 7, 9, 11, 13, 15].map(v => ({ name: String(v), value: v }));
</script>

<template>
    <div class="detection-controls">
        <Fieldset legend="Motion Detection" :toggleable="true">
            <div class="controls-grid">
                <div class="control-item">
                    <label class="control-label">MOG2 History: {{ mog2History }}</label>
                    <Slider v-model="mog2History" :min="50" :max="2000" :step="50" class="slider" />
                    <p class="help-text">Frames used to build the background model. Higher values produce a more stable background but react more slowly to scene changes (e.g. lights turning on).</p>
                </div>

                <div class="control-item">
                    <label class="control-label">MOG2 Variance Threshold: {{ mog2VarThreshold }}</label>
                    <Slider v-model="mog2VarThreshold" :min="4" :max="128" :step="2" class="slider" />
                    <p class="help-text">How different a pixel must be from the background model to be considered foreground. Lower = more sensitive to subtle motion; raise if static scenes produce false detections.</p>
                </div>

                <div class="control-item">
                    <label class="control-label">Denoise Kernel: {{ denoiseKernelsize === 1 ? "off" : denoiseKernelsize }}</label>
                    <Select v-model="denoiseKernelsize" :options="kernelOptions" option-label="name" option-value="value" class="kernel-select" />
                    <p class="help-text">Gaussian blur kernel size applied before background subtraction. Larger values suppress sensor noise but may merge nearby objects. Set to 1 to disable blurring.</p>
                </div>

                <div class="control-item">
                    <label class="control-label">Motion Pixel Threshold: {{ pixelcountThreshold }} px</label>
                    <Slider v-model="pixelcountThreshold" :min="50" :max="2000" :step="50" class="slider" />
                    <p class="help-text">Minimum number of changed pixels required to trigger the object detector. Raise to ignore small or distant motion (e.g. swaying leaves, camera noise).</p>
                </div>

                <div class="control-item">
                    <label class="control-label">Min Blob Area: {{ minArea }} px²</label>
                    <Slider v-model="minArea" :min="50" :max="2000" :step="50" class="slider" />
                    <p class="help-text">Minimum bounding-box area for a motion region to generate a detection ROI. Smaller blobs are discarded. Raise to ignore tiny movement fragments.</p>
                </div>

                <div class="control-item reset-item">
                    <Button label="Reset background" severity="danger" outlined @click="emit('resetBackground')" />
                    <p class="help-text">Clears the learned background model and forces MOG2 to relearn the current scene from scratch. Useful after large scene changes.</p>
                </div>
            </div>
        </Fieldset>

        <Fieldset legend="Object Detection" :toggleable="true">
            <div class="controls-grid">
                <div class="control-item">
                    <label class="control-label">Confidence Threshold: {{ confThreshold.toFixed(2) }}</label>
                    <Slider v-model="confThreshold" :min="0.1" :max="0.95" :step="0.05" class="slider" />
                    <p class="help-text">Minimum YOLO confidence score for a detection to be shown. Raise to display only high-confidence hits and reduce false positives.</p>
                </div>

                <div class="control-item">
                    <label class="control-label">Enable Tracking</label>
                    <div class="toggle-row">
                        <ToggleSwitch v-model="trackingEnabled" input-id="tracking-enabled" />
                        <label for="tracking-enabled" class="toggle-label" :class="trackingEnabled ? 'enabled' : 'disabled'">
                            {{ trackingEnabled ? "Enabled" : "Disabled" }}
                        </label>
                    </div>
                    <p class="help-text">When enabled, a CSRT/KCF tracker locks onto detected objects between YOLO runs. Disable to use only per-frame YOLO detections (higher CPU, no drift correction).</p>
                </div>

                <div class="control-item">
                    <label class="control-label">Tracker Algorithm</label>
                    <Select v-model="model.trackerType" :options="trackerTypeOptions" option-label="name" option-value="value" class="mode-select" :disabled="!trackingEnabled" />
                    <p class="help-text"><strong>CSRT</strong>: more accurate, handles scale and rotation changes well — recommended for moving targets. <strong>KCF</strong>: faster, less accurate — better if CPU is a bottleneck. After a YOLO detection the tracker runs on every frame until it loses the target, then YOLO takes over again.</p>
                </div>

                <div class="control-item">
                    <label class="control-label">Tracker Lost Threshold: {{ trackerLostThreshold }}</label>
                    <Slider v-model="trackerLostThreshold" :min="1" :max="30" :step="1" class="slider" :disabled="!trackingEnabled" />
                    <p class="help-text">How many consecutive frames the tracker must fail before tracking is reset. Higher values keep tracking alive through brief occlusions or low-contrast frames; lower values make the tracker give up faster when the object is genuinely gone.</p>
                </div>
            </div>
        </Fieldset>

        <Fieldset legend="Servo Smoothing" :toggleable="true">
            <div class="controls-grid">
                <div class="control-item">
                    <label class="control-label">Dead Zone: {{ servoDeadZone.toFixed(1) }}°</label>
                    <Slider v-model="servoDeadZone" :min="0" :max="10" :step="0.5" class="slider" />
                    <p class="help-text">Minimum angle change (in both axes) required to move the servo. Ignores tiny detection fluctuations. Raise to suppress jitter from noisy detections; set to 0 to disable.</p>
                </div>

                <div class="control-item">
                    <label class="control-label">Smoothing (α): {{ servoSmoothFactor.toFixed(2) }}</label>
                    <Slider v-model="servoSmoothFactor" :min="0.05" :max="1.0" :step="0.05" class="slider" />
                    <p class="help-text">EMA factor: how much of the new target angle to blend in each update. 1.0 = snap instantly (no smoothing); 0.3 = glide smoothly toward the target. Lower values reduce jitter but increase tracking lag.</p>
                </div>
            </div>
        </Fieldset>

        <Fieldset legend="Display" :toggleable="true">
            <div class="controls-grid">
                <div class="control-item">
                    <label class="control-label">Mode</label>
                    <Select v-model="model.mode" :options="modeOptions" option-label="name" option-value="value" class="mode-select" />
                    <p class="help-text"><strong>boxes</strong>: draw bounding boxes around detections only. <strong>mask</strong>: show the motion foreground mask as a colour overlay. <strong>rois</strong>: show the motion-based region-of-interest rectangles sent to the detector.</p>
                </div>

                <div class="control-item">
                    <label class="control-label">Mask Transparency: {{ maskTransparency.toFixed(2) }}</label>
                    <Slider v-model="maskTransparency" :min="0" :max="1" :step="0.05" class="slider" />
                    <p class="help-text">Opacity of the motion mask overlay used in <em>mask</em> and <em>rois</em> modes. 0 = fully transparent (overlay invisible), 1 = background fully replaced by the mask colour.</p>
                </div>
            </div>
        </Fieldset>
    </div>
</template>

<style scoped>
.detection-controls {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.controls-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
}
.control-item {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    min-width: 12rem;
    max-width: 20rem;
}
.reset-item {
    justify-content: flex-start;
}
.control-label {
    font-size: 0.875rem;
    font-weight: 500;
}
.slider {
    width: 12rem;
}
.mode-select,
.kernel-select {
    width: 10rem;
}
.help-text {
    font-size: 0.78rem;
    color: var(--p-text-muted-color, #888);
    margin: 0;
    line-height: 1.4;
}
.toggle-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.toggle-label {
    font-size: 0.875rem;
    cursor: pointer;
}
.toggle-label.enabled {
    color: var(--p-green-600);
    font-weight: 600;
}
.toggle-label.disabled {
    color: var(--p-text-muted-color);
}
</style>
