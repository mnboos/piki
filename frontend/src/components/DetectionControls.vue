<script setup lang="ts">
import { computed } from "vue";
import { type PikiOptions } from "@/api";
import Button from "primevue/button";
import Select from "primevue/select";
import Slider from "primevue/slider";

const model = defineModel<PikiOptions>({ required: true });
const emit = defineEmits<{ resetBackground: [] }>();

const confThreshold = computed({
    get: () => model.value.confThreshold ?? 0.5,
    set: (v: number) => { model.value.confThreshold = v; },
});
const pixelcountThreshold = computed({
    get: () => model.value.pixelcountThreshold ?? 500,
    set: (v: number) => { model.value.pixelcountThreshold = v; },
});
const minArea = computed({
    get: () => model.value.minArea ?? 500,
    set: (v: number) => { model.value.minArea = v; },
});

const modeOptions = [
    { name: "Boxes", value: "boxes" },
    { name: "Mask", value: "mask" },
    { name: "ROIs", value: "rois" },
];
</script>

<template>
    <div class="detection-controls">
        <div class="control-row">
            <div class="control-item">
                <label class="control-label">Mode</label>
                <Select v-model="model.mode" :options="modeOptions" option-label="name" option-value="value" />
            </div>

            <div class="control-item">
                <label class="control-label">Confidence: {{ confThreshold.toFixed(2) }}</label>
                <Slider v-model="confThreshold" :min="0.1" :max="0.95" :step="0.05" class="slider" />
            </div>

            <div class="control-item">
                <label class="control-label">Motion sensitivity: {{ pixelcountThreshold }} px</label>
                <Slider v-model="pixelcountThreshold" :min="50" :max="2000" :step="50" class="slider" />
            </div>

            <div class="control-item">
                <label class="control-label">Min area: {{ minArea }} px²</label>
                <Slider v-model="minArea" :min="50" :max="2000" :step="50" class="slider" />
            </div>

            <Button label="Reset background" severity="danger" outlined @click="emit('resetBackground')" />
        </div>
    </div>
</template>

<style scoped>
.detection-controls {
    padding: 0.75rem 0;
}
.control-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    align-items: flex-end;
}
.control-item {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    min-width: 10rem;
}
.control-label {
    font-size: 0.875rem;
    font-weight: 500;
}
.slider {
    width: 10rem;
}
</style>
