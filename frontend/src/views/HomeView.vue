<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { DefaultApi, type PikiOptions, type PikiOptionsPatch } from "@/api";

const modeOptions = ref([
    { name: "Boxes", value: "boxes" },
    { name: "Mask", value: "mask" },
    { name: "ROIs", value: "rois" },
]);

const api = new DefaultApi();

const options = ref<PikiOptions>({
    mode: "boxes",
    confThreshold: 0.5,
    pixelcountThreshold: 500,
    minArea: 500,
});

const { mutate: updateOptions } = useMutation({
    mutationFn: (options: PikiOptionsPatch) => api.coreApiUpdateOptions({ pikiOptionsPatch: options }),
});

const { mutate: resetBackground } = useMutation({
    mutationFn: () => api.coreApiResetBackground(),
});

// Load current values from backend on mount
onMounted(async () => {
    try {
        const current = await api.coreApiGetOptions();
        options.value = {
            mode: current.mode,
            confThreshold: current.confThreshold ?? 0.5,
            pixelcountThreshold: current.pixelcountThreshold ?? 500,
            minArea: current.minArea ?? 500,
        };
    } catch {
        // ignore — use defaults
    }
});

watch(options, options => updateOptions(options), { deep: true });

const feedUrl = "/api/video_feed"
</script>

<template>
    <div class="border-2 p-4">
        <div class="flex gap-4 items-center mb-3 flex-wrap">
            <label>
                Mode:
                <select v-model="options.mode" class="border rounded px-1">
                    <option v-for="o in modeOptions" :key="o.value" :value="o.value">{{ o.name }}</option>
                </select>
            </label>

            <label>
                Confidence: {{ (options.confThreshold ?? 0.5).toFixed(2) }}
                <input type="range" v-model.number="options.confThreshold" min="0.1" max="0.95" step="0.05" />
            </label>

            <label>
                Motion sensitivity (px): {{ options.pixelcountThreshold }}
                <input type="range" v-model.number="options.pixelcountThreshold" min="50" max="2000" step="50" />
            </label>

            <label>
                Min area (px²): {{ options.minArea }}
                <input type="range" v-model.number="options.minArea" min="50" max="2000" step="50" />
            </label>

            <button @click="resetBackground()" class="border rounded px-3 py-1 bg-red-100 hover:bg-red-200">
                Reset background
            </button>
        </div>

        <div style="border: #ff000044 1px solid; border-radius: 5px">
            <img id="camera-feed" :src="feedUrl" alt="feed" />
        </div>
    </div>
</template>
