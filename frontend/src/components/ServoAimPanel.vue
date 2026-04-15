<script setup lang="ts">
import { ref, computed, watch } from "vue";
import { type AimConfigSchema, type PikiOptions } from "@/api";
import Panel from "primevue/panel";
import Fieldset from "primevue/fieldset";
import ToggleSwitch from "primevue/toggleswitch";
import Checkbox from "primevue/checkbox";
import InputNumber from "primevue/inputnumber";
import Select from "primevue/select";
import SelectButton from "primevue/selectbutton";

const model = defineModel<AimConfigSchema>({ required: true });
const options = defineModel<PikiOptions>("options", { required: true });
defineProps<{ classes: string[] }>();

function toggleClass(cls: string) {
    const idx = model.value.targetClasses.indexOf(cls);
    if (idx === -1) {
        model.value.targetClasses = [...model.value.targetClasses, cls];
    } else {
        model.value.targetClasses = model.value.targetClasses.filter(c => c !== cls);
    }
}

// ── PID presets ──────────────────────────────────────────────────────────────

interface PidPreset {
    label: string;
    kp: number;
    ki: number;
    kd: number;
    deadZone: number;
    description: string;
}

const PID_PRESETS: PidPreset[] = [
    {
        label: "Slow & Stable",
        kp: 0.5, ki: 0.0, kd: 0.0, deadZone: 3.0,
        description: "Gentle tracking for slow-moving targets. Large dead zone suppresses jitter from noisy detections.",
    },
    {
        label: "Balanced",
        kp: 1.0, ki: 0.0, kd: 0.05, deadZone: 1.5,
        description: "Good all-round starting point. Moderate response with light derivative damping to prevent overshoot.",
    },
    {
        label: "Responsive",
        kp: 2.0, ki: 0.0, kd: 0.1, deadZone: 1.0,
        description: "Faster tracking for targets that move quickly. Higher gain with damping keeps it stable.",
    },
    {
        label: "Aggressive",
        kp: 4.0, ki: 0.05, kd: 0.2, deadZone: 0.5,
        description: "Maximum responsiveness. Minimal dead zone and integral correction for near-zero steady-state error.",
    },
];

function findMatchingPreset(kp?: number | null, ki?: number | null, kd?: number | null, dz?: number | null): PidPreset | null {
    return PID_PRESETS.find(p =>
        Math.abs(p.kp - (kp ?? 0)) < 0.001 &&
        Math.abs(p.ki - (ki ?? 0)) < 0.001 &&
        Math.abs(p.kd - (kd ?? 0)) < 0.001 &&
        Math.abs(p.deadZone - (dz ?? 0)) < 0.001
    ) ?? null;
}

// ── Auto / Manual mode ───────────────────────────────────────────────────────

const pidModeOptions = [
    { label: "Disabled", value: "disabled" },
    { label: "Auto", value: "auto" },
    { label: "Manual", value: "manual" },
];

const initialPreset = findMatchingPreset(
    options.value.servoPidKp,
    options.value.servoPidKi,
    options.value.servoPidKd,
    options.value.servoDeadZone,
);
const isDisabled = !options.value.servoPidKp && !options.value.servoPidKi && !options.value.servoPidKd;
const pidMode = ref<"disabled" | "auto" | "manual">(
    isDisabled ? "disabled" : initialPreset ? "auto" : "manual"
);
const selectedPreset = ref<PidPreset | null>(initialPreset ?? PID_PRESETS[1]);

// When a preset is chosen, push its values into options immediately.
watch(selectedPreset, preset => {
    if (!preset) return;
    options.value.servoPidKp = preset.kp;
    options.value.servoPidKi = preset.ki;
    options.value.servoPidKd = preset.kd;
    options.value.servoDeadZone = preset.deadZone;
});

// If the user switches to Auto, apply the currently-selected preset.
// If the user switches to Disabled, zero out all gains.
watch(pidMode, mode => {
    if (mode === "auto" && selectedPreset.value) {
        options.value.servoPidKp = selectedPreset.value.kp;
        options.value.servoPidKi = selectedPreset.value.ki;
        options.value.servoPidKd = selectedPreset.value.kd;
        options.value.servoDeadZone = selectedPreset.value.deadZone;
    } else if (mode === "disabled") {
        options.value.servoPidKp = 0;
        options.value.servoPidKi = 0;
        options.value.servoPidKd = 0;
    }
});

const presetDescription = computed(() => selectedPreset.value?.description ?? "");
</script>

<template>
    <Panel>
        <template #header>
            <div class="panel-header">
                <span class="panel-title">Servo Aiming</span>
                <div class="toggle-row">
                    <ToggleSwitch v-model="model.servoEnabled" input-id="servo-enabled" />
                    <label
                        for="servo-enabled"
                        class="toggle-label"
                        :class="model.servoEnabled ? 'enabled' : 'disabled'"
                    >
                        {{ model.servoEnabled ? "Enabled" : "Disabled" }}
                    </label>
                </div>
            </div>
        </template>

        <p class="hint">
            Aim at detections matching these classes
            <span v-if="model.targetClasses.length > 0" class="selected-count">
                ({{ model.targetClasses.length }} selected)
            </span>:
        </p>
        <div class="class-grid">
            <div v-for="cls in classes" :key="cls" class="class-item">
                <Checkbox
                    :input-id="`cls-${cls}`"
                    :model-value="model.targetClasses.includes(cls)"
                    :binary="true"
                    @update:model-value="toggleClass(cls)"
                />
                <label :for="`cls-${cls}`" class="class-label">{{ cls }}</label>
            </div>
        </div>

        <div class="lock-row">
            <label for="target-lock-duration" class="lock-label">Target lock duration (s)</label>
            <InputNumber
                input-id="target-lock-duration"
                v-model="model.targetLockDuration"
                :min="0"
                :max="60"
                :step="0.5"
                :min-fraction-digits="1"
                :max-fraction-digits="1"
                class="lock-input"
            />
        </div>

        <Fieldset legend="PID Controller" class="pid-fieldset" :toggleable="true">
            <!-- Auto / Manual toggle -->
            <div class="pid-mode-row">
                <SelectButton v-model="pidMode" :options="pidModeOptions" option-label="label" option-value="value" />
            </div>

            <!-- Auto: preset picker -->
            <div v-if="pidMode === 'auto'" class="pid-auto">
                <Select
                    v-model="selectedPreset"
                    :options="PID_PRESETS"
                    option-label="label"
                    placeholder="Choose a preset…"
                    class="preset-select"
                />
                <p v-if="presetDescription" class="pid-help preset-desc">{{ presetDescription }}</p>
                <div class="preset-summary">
                    <span class="preset-chip">Kp {{ selectedPreset?.kp.toFixed(2) }}</span>
                    <span class="preset-chip">Ki {{ selectedPreset?.ki.toFixed(2) }}</span>
                    <span class="preset-chip">Kd {{ selectedPreset?.kd.toFixed(2) }}</span>
                    <span class="preset-chip">Dead zone {{ selectedPreset?.deadZone.toFixed(1) }}°</span>
                </div>
            </div>

            <!-- Manual: individual knobs -->
            <div v-else-if="pidMode === 'manual'" class="pid-grid">
                <div class="pid-item">
                    <label for="pid-kp" class="pid-label">Kp <span class="pid-sub">(proportional)</span></label>
                    <InputNumber
                        input-id="pid-kp"
                        v-model="options.servoPidKp"
                        :min="0"
                        :max="20"
                        :step="0.1"
                        :min-fraction-digits="2"
                        :max-fraction-digits="2"
                        class="pid-input"
                    />
                    <p class="pid-help">How aggressively the servo reacts to the current tracking error. Higher = faster response but may overshoot.</p>
                </div>

                <div class="pid-item">
                    <label for="pid-ki" class="pid-label">Ki <span class="pid-sub">(integral)</span></label>
                    <InputNumber
                        input-id="pid-ki"
                        v-model="options.servoPidKi"
                        :min="0"
                        :max="5"
                        :step="0.01"
                        :min-fraction-digits="2"
                        :max-fraction-digits="2"
                        class="pid-input"
                    />
                    <p class="pid-help">Corrects persistent steady-state error. Raise if the servo consistently stops short of the target. Keep near 0 to avoid wind-up.</p>
                </div>

                <div class="pid-item">
                    <label for="pid-kd" class="pid-label">Kd <span class="pid-sub">(derivative)</span></label>
                    <InputNumber
                        input-id="pid-kd"
                        v-model="options.servoPidKd"
                        :min="0"
                        :max="5"
                        :step="0.01"
                        :min-fraction-digits="2"
                        :max-fraction-digits="2"
                        class="pid-input"
                    />
                    <p class="pid-help">Damps oscillations by reacting to the rate of error change. Raise if the servo overshoots or oscillates around the target.</p>
                </div>

                <div class="pid-item">
                    <label for="pid-dead-zone" class="pid-label">Dead zone <span class="pid-sub">(°)</span></label>
                    <InputNumber
                        input-id="pid-dead-zone"
                        v-model="options.servoDeadZone"
                        :min="0"
                        :max="10"
                        :step="0.1"
                        :min-fraction-digits="1"
                        :max-fraction-digits="1"
                        class="pid-input"
                    />
                    <p class="pid-help">Tracking errors smaller than this angle are ignored, suppressing jitter from noisy detections. Set to 0 to disable.</p>
                </div>
            </div>
        </Fieldset>
    </Panel>
</template>

<style scoped>
.panel-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}
.panel-title {
    font-size: 1rem;
    font-weight: 600;
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
.hint {
    font-size: 0.875rem;
    color: var(--p-text-muted-color);
    margin-bottom: 0.75rem;
}
.selected-count {
    font-weight: 600;
    color: var(--p-text-color);
}
.class-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 1rem;
}
.class-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.class-label {
    font-size: 0.875rem;
    cursor: pointer;
    user-select: none;
}
.lock-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0.75rem;
}
.lock-label {
    font-size: 0.875rem;
    color: var(--p-text-muted-color);
    white-space: nowrap;
}
.lock-input {
    width: 5rem;
}
.pid-fieldset {
    margin-top: 1rem;
}
.pid-mode-row {
    margin-bottom: 1rem;
}
.pid-auto {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}
.preset-select {
    width: 14rem;
}
.preset-desc {
    max-width: 30rem;
}
.preset-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.25rem;
}
.preset-chip {
    font-size: 0.78rem;
    font-family: monospace;
    background: var(--p-surface-100, #2a2a2a);
    border: 1px solid var(--p-surface-300, #444);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    color: var(--p-text-muted-color);
}
.pid-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 1.25rem;
}
.pid-item {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    min-width: 11rem;
    max-width: 18rem;
}
.pid-label {
    font-size: 0.875rem;
    font-weight: 500;
}
.pid-sub {
    font-weight: 400;
    color: var(--p-text-muted-color);
}
.pid-input {
    width: 7rem;
}
.pid-help {
    font-size: 0.78rem;
    color: var(--p-text-muted-color, #888);
    margin: 0;
    line-height: 1.4;
}
</style>
