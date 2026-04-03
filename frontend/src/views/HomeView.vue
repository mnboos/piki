<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { DefaultApi, type AimConfigSchema, type AimConfigSchemaPatch, type PikiOptions, type PikiOptionsPatch } from "@/api";

// Preset positions: [label, pan°, tilt°]
const PRESETS: [string, number, number][] = [
    ["↖", -60, -45], ["↑", 0, -45], ["↗", 60, -45],
    ["←", -60,   0], ["·", 0,    0], ["→", 60,   0],
    ["↙", -60,  45], ["↓", 0,   45], ["↘", 60,  45],
];

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

const aimConfig = ref<AimConfigSchema>({
    targetClasses: [],
    servoEnabled: false,
});

const allClasses = ref<string[]>([]);

// Manual servo debug state
const manualPan = ref(0);
const manualTilt = ref(0);
const manualResult = ref<{ pan: number; tilt: number } | null>(null);

const { mutate: updateOptions } = useMutation({
    mutationFn: (options: PikiOptionsPatch) => api.coreApiUpdateOptions({ pikiOptionsPatch: options }),
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
    onSuccess: (data) => { manualResult.value = { pan: data.panAngle, tilt: data.tiltAngle }; },
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
    } catch {
        // ignore — use defaults
    }

    try {
        allClasses.value = await api.coreApiGetYoloClasses();
    } catch {
        // ignore
    }

    try {
        const aim = await api.coreApiGetAimConfig();
        aimConfig.value = { targetClasses: aim.targetClasses, servoEnabled: aim.servoEnabled };
    } catch {
        // ignore
    }
});

watch(options, options => updateOptions(options), { deep: true });
watch(aimConfig, cfg => updateAimConfig(cfg), { deep: true });

function toggleClass(cls: string) {
    const idx = aimConfig.value.targetClasses.indexOf(cls);
    if (idx === -1) {
        aimConfig.value.targetClasses = [...aimConfig.value.targetClasses, cls];
    } else {
        aimConfig.value.targetClasses = aimConfig.value.targetClasses.filter(c => c !== cls);
    }
}

function applyPreset(pan: number, tilt: number) {
    manualPan.value = pan;
    manualTilt.value = tilt;
    servoMove({ panAngle: pan, tiltAngle: tilt });
}

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

        <!-- Servo Aim Controls -->
        <div class="border-2 mt-4 p-4 rounded">
            <div class="flex items-center gap-3 mb-3">
                <h2 class="font-bold text-lg">Servo Aiming</h2>
                <label class="flex items-center gap-2 cursor-pointer select-none">
                    <input
                        type="checkbox"
                        v-model="aimConfig.servoEnabled"
                        class="w-4 h-4 accent-green-600"
                    />
                    <span :class="aimConfig.servoEnabled ? 'text-green-700 font-semibold' : 'text-gray-500'">
                        {{ aimConfig.servoEnabled ? "Enabled" : "Disabled" }}
                    </span>
                </label>
            </div>

            <div>
                <p class="text-sm text-gray-600 mb-2">
                    Aim at detections matching these classes
                    <span v-if="aimConfig.targetClasses.length > 0" class="font-medium text-gray-800">
                        ({{ aimConfig.targetClasses.length }} selected)
                    </span>:
                </p>
                <div class="flex flex-wrap gap-x-3 gap-y-1">
                    <label
                        v-for="cls in allClasses"
                        :key="cls"
                        class="flex items-center gap-1 text-sm cursor-pointer select-none"
                    >
                        <input
                            type="checkbox"
                            :checked="aimConfig.targetClasses.includes(cls)"
                            @change="toggleClass(cls)"
                            class="w-3.5 h-3.5 accent-blue-600"
                        />
                        {{ cls }}
                    </label>
                </div>
            </div>
        </div>
        <!-- Manual Servo Debug -->
        <div class="border-2 mt-4 p-4 rounded">
            <h2 class="font-bold text-lg mb-3">Manual Servo Debug</h2>

            <!-- 3×3 preset grid -->
            <div class="grid grid-cols-3 gap-1 w-32 mb-4">
                <button
                    v-for="([label, pan, tilt]) in PRESETS" :key="label"
                    @click="applyPreset(pan, tilt)"
                    class="border rounded py-1 text-lg leading-none hover:bg-gray-100 active:bg-gray-200"
                    :title="`pan=${pan}° tilt=${tilt}°`"
                >{{ label }}</button>
            </div>

            <!-- Pan slider -->
            <label class="block mb-2 text-sm">
                Pan: <span class="font-mono w-12 inline-block">{{ manualPan }}°</span>
                <input type="range" v-model.number="manualPan" min="-90" max="90" step="1" class="ml-2 w-48 align-middle" />
            </label>

            <!-- Tilt slider -->
            <label class="block mb-3 text-sm">
                Tilt: <span class="font-mono w-12 inline-block">{{ manualTilt }}°</span>
                <input type="range" v-model.number="manualTilt" min="-90" max="90" step="1" class="ml-2 w-48 align-middle" />
            </label>

            <div class="flex items-center gap-3">
                <button
                    @click="servoMove({ panAngle: manualPan, tiltAngle: manualTilt })"
                    class="border rounded px-4 py-1 bg-blue-100 hover:bg-blue-200"
                >Move</button>
                <span v-if="manualResult" class="text-xs text-gray-500 font-mono">
                    → pan {{ manualResult.pan.toFixed(1) }}°  tilt {{ manualResult.tilt.toFixed(1) }}°
                </span>
            </div>
        </div>
    </div>
</template>
