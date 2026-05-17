<script setup lang="ts">
import { computed } from "vue";
import { type PikiOptions } from "@/api";
import Button from "primevue/button";
import Select from "primevue/select";
import Slider from "primevue/slider";
import ToggleButton from "primevue/togglebutton";
import Tab from "primevue/tab";
import TabList from "primevue/tablist";
import TabPanel from "primevue/tabpanel";
import TabPanels from "primevue/tabpanels";
import Tabs from "primevue/tabs";

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

// Servo smoothing — removed (PID controls moved to ServoAimPanel)

// Display
const maskTransparency = computed({
    get: () => model.value.maskTransparency ?? 0.5,
    set: (v: number) => { model.value.maskTransparency = v; },
});

const kernelOptions = [1, 3, 5, 7, 9, 11, 13, 15].map(v => ({ name: String(v), value: v }));
</script>

<template>
    <div class="detection-controls">
        <Tabs value="motion">
            <TabList>
                <Tab value="motion">Motion</Tab>
                <Tab value="object">Object</Tab>
                <Tab value="display">Display</Tab>
            </TabList>
            <TabPanels>
                <TabPanel value="motion">
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
                </TabPanel>

                <TabPanel value="object">
                    <div class="controls-grid">
                        <div class="control-item">
                            <label class="control-label">Confidence Threshold: {{ confThreshold.toFixed(2) }}</label>
                            <Slider v-model="confThreshold" :min="0.1" :max="0.95" :step="0.05" class="slider" />
                            <p class="help-text">Minimum YOLO confidence score for a detection to be shown. Raise to display only high-confidence hits and reduce false positives.</p>
                        </div>

                    </div>
                </TabPanel>

                <TabPanel value="display">
                    <div class="controls-grid">
                        <div class="control-item">
                            <label class="control-label">Overlays</label>
                            <div class="toggle-group">
                                <ToggleButton v-model="model.showBoxes" on-label="Boxes" off-label="Boxes" class="tb-btn" />
                                <ToggleButton v-model="model.showMask" on-label="Mask" off-label="Mask" class="tb-btn" />
                                <ToggleButton v-model="model.showRois" on-label="ROIs" off-label="ROIs" class="tb-btn" />
                            </div>
                            <p class="help-text">Toggle overlays independently. <strong>Boxes</strong>: YOLO detection boxes. <strong>Mask</strong>: motion foreground mask. <strong>ROIs</strong>: tile rectangles sent to the detector.</p>
                        </div>

                        <div class="control-item">
                            <label class="control-label">Mask Transparency: {{ maskTransparency.toFixed(2) }}</label>
                            <Slider v-model="maskTransparency" :min="0" :max="1" :step="0.05" class="slider" />
                            <p class="help-text">Opacity of the motion mask overlay used in <em>mask</em> and <em>rois</em> modes. 0 = fully transparent (overlay invisible), 1 = background fully replaced by the mask colour.</p>
                        </div>
                    </div>
                </TabPanel>
            </TabPanels>
        </Tabs>
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
.toggle-group {
    display: flex;
    gap: 0.4rem;
}
.tb-btn {
    font-size: 0.8rem !important;
    padding: 0.3rem 0.7rem !important;
}
.help-text {
    font-size: 0.78rem;
    color: var(--p-text-muted-color, #888);
    margin: 0;
    line-height: 1.4;
}
</style>
