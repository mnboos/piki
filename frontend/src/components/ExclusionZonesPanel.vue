<script setup lang="ts">
import { ref } from "vue";
import Button from "primevue/button";
import InputText from "primevue/inputtext";
import ToggleButton from "primevue/togglebutton";
import {
    useDeleteExclusionZoneMutation,
    useExclusionZonesQuery,
    useUpdateExclusionZoneMutation,
} from "@/queries/exclusionZones";
import type { ExclusionZoneSchema } from "@/api";

const editing = defineModel<boolean>({ required: true });

const zonesQuery = useExclusionZonesQuery();
const updateMutation = useUpdateExclusionZoneMutation();
const deleteMutation = useDeleteExclusionZoneMutation();

const renamingId = ref<number | null>(null);
const renameDraft = ref("");

function startRename(z: ExclusionZoneSchema) {
    if (!z.id) return;
    renamingId.value = z.id;
    renameDraft.value = z.name ?? "";
}

function commitRename(z: ExclusionZoneSchema) {
    if (!z.id || renamingId.value !== z.id) return;
    const newName = renameDraft.value.trim() || "zone";
    if (newName !== z.name) {
        updateMutation.mutate({ zoneId: z.id, patch: { name: newName } });
    }
    renamingId.value = null;
}

function toggleEnabled(z: ExclusionZoneSchema) {
    if (!z.id) return;
    updateMutation.mutate({ zoneId: z.id, patch: { enabled: !z.enabled } });
}

function remove(z: ExclusionZoneSchema) {
    if (!z.id) return;
    if (!confirm(`Delete exclusion zone "${z.name}"?`)) return;
    deleteMutation.mutate(z.id);
}
</script>

<template>
    <div class="ezp">
        <div class="ezp-header">
            <span class="ezp-title">Exclusion zones</span>
            <ToggleButton v-model="editing" on-label="Editing" off-label="Edit" class="ezp-edit" />
        </div>

        <p class="ezp-hint" v-if="editing">
            Click-and-drag on the camera feed to add a rectangle. Drag the body to
            move; drag the corners to resize.
        </p>
        <p class="ezp-hint" v-else>
            Areas drawn here are ignored by detection, aiming, and recording triggers.
            Enable <strong>Edit</strong> to draw or modify zones.
        </p>

        <div v-if="zonesQuery.isLoading.value" class="ezp-empty">Loading…</div>
        <div v-else-if="!zonesQuery.data.value || zonesQuery.data.value.length === 0" class="ezp-empty">
            No exclusion zones yet.
        </div>

        <ul v-else class="ezp-list">
            <li v-for="z in zonesQuery.data.value" :key="z.id ?? -1" class="ezp-item">
                <div class="ezp-name">
                    <template v-if="renamingId === z.id">
                        <InputText
                            v-model="renameDraft"
                            class="ezp-rename-input"
                            size="small"
                            autofocus
                            @keyup.enter="commitRename(z)"
                            @keyup.escape="renamingId = null"
                            @blur="commitRename(z)"
                        />
                    </template>
                    <template v-else>
                        <span class="ezp-name-text" :class="{ 'ezp-name-text--off': !z.enabled }" @dblclick="startRename(z)">{{ z.name }}</span>
                    </template>
                </div>
                <div class="ezp-actions">
                    <Button
                        :icon="z.enabled ? 'pi pi-eye-slash' : 'pi pi-eye'"
                        :severity="z.enabled ? 'secondary' : 'success'"
                        size="small"
                        text
                        rounded
                        :aria-label="z.enabled ? 'Disable' : 'Enable'"
                        @click="toggleEnabled(z)"
                    />
                    <Button
                        icon="pi pi-pencil"
                        size="small"
                        text
                        rounded
                        aria-label="Rename"
                        @click="startRename(z)"
                    />
                    <Button
                        icon="pi pi-trash"
                        severity="danger"
                        size="small"
                        text
                        rounded
                        aria-label="Delete"
                        @click="remove(z)"
                    />
                </div>
            </li>
        </ul>
    </div>
</template>

<style scoped>
.ezp {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.ezp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
}
.ezp-title {
    font-weight: 600;
}
.ezp-edit {
    font-size: 0.8rem !important;
    padding: 0.3rem 0.7rem !important;
}
.ezp-hint {
    font-size: 0.78rem;
    color: var(--p-text-muted-color, #888);
    margin: 0;
}
.ezp-empty {
    font-size: 0.85rem;
    color: var(--p-text-muted-color, #888);
}
.ezp-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.ezp-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.25rem 0.25rem 0.25rem 0.5rem;
    border-radius: 0.4rem;
    background: var(--p-surface-50, rgba(255, 255, 255, 0.04));
}
.ezp-name {
    flex: 1;
    min-width: 0;
}
.ezp-name-text {
    font-size: 0.9rem;
    cursor: text;
}
.ezp-name-text--off {
    opacity: 0.5;
    text-decoration: line-through;
}
.ezp-rename-input {
    width: 100%;
}
.ezp-actions {
    display: flex;
    gap: 0.1rem;
}
</style>
