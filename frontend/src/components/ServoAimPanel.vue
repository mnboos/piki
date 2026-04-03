<script setup lang="ts">
import { type AimConfigSchema } from "@/api";
import Panel from "primevue/panel";
import ToggleSwitch from "primevue/toggleswitch";
import Checkbox from "primevue/checkbox";

const model = defineModel<AimConfigSchema>({ required: true });
defineProps<{ classes: string[] }>();

function toggleClass(cls: string) {
    const idx = model.value.targetClasses.indexOf(cls);
    if (idx === -1) {
        model.value.targetClasses = [...model.value.targetClasses, cls];
    } else {
        model.value.targetClasses = model.value.targetClasses.filter(c => c !== cls);
    }
}
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
</style>
