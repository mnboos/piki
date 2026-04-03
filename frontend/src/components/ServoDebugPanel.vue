<script setup lang="ts">
import { ref } from "vue";
import Panel from "primevue/panel";
import Button from "primevue/button";
import Slider from "primevue/slider";

const emit = defineEmits<{ move: [pan: number, tilt: number] }>();

// Preset positions: [label, pan°, tilt°]
const PRESETS: [string, number, number][] = [
    ["↖", -60, -45], ["↑", 0, -45], ["↗", 60, -45],
    ["←", -60,   0], ["·", 0,    0], ["→", 60,   0],
    ["↙", -60,  45], ["↓", 0,   45], ["↘", 60,  45],
];

const pan = ref(0);
const tilt = ref(0);
const result = ref<{ pan: number; tilt: number } | null>(null);

defineExpose({ setResult(p: number, t: number) { result.value = { pan: p, tilt: t }; } });

function applyPreset(p: number, t: number) {
    pan.value = p;
    tilt.value = t;
    emit("move", p, t);
}
</script>

<template>
    <Panel header="Manual Servo Debug">
        <div class="preset-grid">
            <Button
                v-for="([label, p, t]) in PRESETS"
                :key="label"
                :label="label"
                :title="`pan=${p}° tilt=${t}°`"
                outlined
                @click="applyPreset(p, t)"
                class="preset-btn"
            />
        </div>

        <div class="slider-row">
            <label class="slider-label">Pan: <span class="angle-value">{{ pan }}°</span></label>
            <Slider v-model="pan" :min="-90" :max="90" :step="1" class="angle-slider" />
        </div>

        <div class="slider-row">
            <label class="slider-label">Tilt: <span class="angle-value">{{ tilt }}°</span></label>
            <Slider v-model="tilt" :min="-90" :max="90" :step="1" class="angle-slider" />
        </div>

        <div class="move-row">
            <Button label="Move" @click="emit('move', pan, tilt)" />
            <span v-if="result" class="result-text">
                → pan {{ result.pan.toFixed(1) }}° tilt {{ result.tilt.toFixed(1) }}°
            </span>
        </div>
    </Panel>
</template>

<style scoped>
.preset-grid {
    display: grid;
    grid-template-columns: repeat(3, 2.5rem);
    gap: 0.25rem;
    margin-bottom: 1.25rem;
}
.preset-btn {
    width: 2.5rem;
    height: 2.5rem;
    padding: 0;
    font-size: 1.1rem;
    justify-content: center;
}
.slider-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}
.slider-label {
    font-size: 0.875rem;
    width: 7rem;
    flex-shrink: 0;
}
.angle-value {
    font-family: monospace;
    display: inline-block;
    width: 3rem;
}
.angle-slider {
    width: 12rem;
}
.move-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 0.5rem;
}
.result-text {
    font-size: 0.75rem;
    font-family: monospace;
    color: var(--p-text-muted-color);
}
</style>
