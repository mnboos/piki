<script setup lang="ts">
import { type SplashConfigSchema } from "@/api";
import Panel from "primevue/panel";
import ToggleSwitch from "primevue/toggleswitch";
import InputNumber from "primevue/inputnumber";
import Slider from "primevue/slider";
import Button from "primevue/button";

const model = defineModel<SplashConfigSchema>({ required: true });
defineProps<{
    firePump: () => void;
    firePumpPending?: boolean;
}>();
</script>

<template>
    <Panel>
        <template #header>
            <div class="panel-header">
                <span class="panel-title">Splash (Relay)</span>
                <div class="toggle-row">
                    <ToggleSwitch v-model="model.enabled" input-id="splash-enabled" />
                    <label
                        for="splash-enabled"
                        class="toggle-label"
                        :class="model.enabled ? 'enabled' : 'disabled'"
                    >
                        {{ model.enabled ? "Enabled" : "Disabled" }}
                    </label>
                </div>
            </div>
        </template>

        <p class="hint">Uses the same classes configured in Servo Aiming above.</p>

        <div class="pump-row">
            <label for="pump-duty" class="timing-label">Pump power: {{ model.pumpDuty ?? 100 }}%</label>
            <Slider
                v-model="model.pumpDuty"
                :min="0"
                :max="100"
                :step="5"
                class="pump-slider"
            />
        </div>

        <div class="timing-grid">
            <div class="timing-item">
                <label for="splash-delay" class="timing-label">Delay (s)</label>
                <InputNumber
                    input-id="splash-delay"
                    v-model="model.delaySeconds"
                    :min="0"
                    :max="30"
                    :step="0.1"
                    :min-fraction-digits="1"
                    :max-fraction-digits="1"
                    class="timing-input"
                />
                <p class="timing-help">Wait after target lock before firing.</p>
            </div>

            <div class="timing-item">
                <label for="splash-duration" class="timing-label">Duration (s)</label>
                <InputNumber
                    input-id="splash-duration"
                    v-model="model.durationSeconds"
                    :min="0.01"
                    :max="10"
                    :step="0.1"
                    :min-fraction-digits="2"
                    :max-fraction-digits="2"
                    class="timing-input"
                />
                <p class="timing-help">How long the relay stays active.</p>
            </div>

            <div class="timing-item">
                <label for="splash-cooldown" class="timing-label">Cooldown (s)</label>
                <InputNumber
                    input-id="splash-cooldown"
                    v-model="model.cooldownSeconds"
                    :min="0"
                    :max="600"
                    :step="1"
                    :min-fraction-digits="0"
                    class="timing-input"
                />
                <p class="timing-help">Minimum time between splashes.</p>
            </div>
        </div>

        <Button
            label="Fire Pump"
            icon="pi pi-bolt"
            :loading="firePumpPending"
            severity="warn"
            size="small"
            class="fire-btn"
            @click="firePump"
        />
    </Panel>
</template>

<style scoped>
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
}
.panel-title {
    font-weight: 700;
    font-size: 1rem;
}
.toggle-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.toggle-label {
    font-size: 0.8rem;
    font-weight: 600;
    user-select: none;
}
.toggle-label.enabled {
    color: var(--p-green-500, #22c55e);
}
.toggle-label.disabled {
    color: var(--p-text-muted-color, #888);
}
.hint {
    font-size: 0.8rem;
    color: var(--p-text-muted-color);
    margin: 0.5rem 0;
}
.pump-row {
    margin: 0.75rem 0;
}
.pump-slider {
    width: 100%;
    margin-top: 0.25rem;
}
.timing-grid {
    display: flex;
    gap: 1rem;
    margin-top: 0.75rem;
}
.timing-item {
    flex: 1;
    min-width: 0;
}
.timing-label {
    font-size: 0.8rem;
    font-weight: 600;
    display: block;
    margin-bottom: 0.25rem;
}
.timing-input {
    width: 100%;
}
.timing-help {
    font-size: 0.7rem;
    color: var(--p-text-muted-color);
    margin: 0.2rem 0 0;
}
.fire-btn {
    margin-top: 0.75rem;
    width: 100%;
}
</style>
