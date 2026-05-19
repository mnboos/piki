<script setup lang="ts">
import { ref, watch } from "vue";
import { type EventClipSchema, type VideoInfo } from "@/api";
import {
  useEventClipsQuery,
  useEventRecordingConfigQuery,
  useUpdateEventRecordingConfigMutation,
  useRecordingStatusQuery,
  useStartRecordingMutation,
  useStopRecordingMutation,
  useVideosQuery,
  useReplayStatusQuery,
  useStartReplayMutation,
  useStopReplayMutation,
  useYoloClassesQuery,
} from "@/queries/recordings";
import Panel from "primevue/panel";
import Button from "primevue/button";
import ToggleSwitch from "primevue/toggleswitch";
import Checkbox from "primevue/checkbox";
import InputNumber from "primevue/inputnumber";
import Fieldset from "primevue/fieldset";

// ── Queries ───────────────────────────────────────────────────────────────

const {
  data: recordingStatus,
} = useRecordingStatusQuery();
const {
  data: eventConfig,
} = useEventRecordingConfigQuery();
const {
  data: allClasses,
} = useYoloClassesQuery();
const {
  data: eventClipsRaw,
} = useEventClipsQuery();
const {
  data: videosRaw,
  refetch: refetchVideos,
} = useVideosQuery();
const {
  data: replayStatus,
} = useReplayStatusQuery();

// ── Mutations ─────────────────────────────────────────────────────────────

const { mutate: updateEventConfig } = useUpdateEventRecordingConfigMutation();
const { mutate: startRecording } = useStartRecordingMutation();
const { mutate: stopRecording } = useStopRecordingMutation();
const { mutate: startReplay } = useStartReplayMutation();
const { mutate: stopReplay } = useStopReplayMutation();

// ── Accumulated event clips (backend drains queue on each poll) ───────────

const eventClips = ref<EventClipSchema[]>([]);
const seenFiles = new Set<string>();

watch(eventClipsRaw, (clips) => {
  if (!clips) return;
  for (const c of clips) {
    if (!seenFiles.has(c.file)) {
      seenFiles.add(c.file);
      eventClips.value.unshift(c);
    }
  }
});

// ── Trigger class toggle ──────────────────────────────────────────────────

function toggleTriggerClass(cls: string) {
  if (!eventConfig.value) return;
  const current = eventConfig.value.triggerClasses ?? [];
  const idx = current.indexOf(cls);
  if (idx === -1) {
    updateEventConfig({ triggerClasses: [...current, cls] });
  } else {
    updateEventConfig({ triggerClasses: current.filter(c => c !== cls) });
  }
}

// ── Upload ────────────────────────────────────────────────────────────────

const CSRF_COOKIE = "csrftoken";
function getCsrf(): string {
  const match = document.cookie.match(new RegExp(`(?:^|; )${CSRF_COOKIE}=([^;]*)`));
  return match ? match[1] : "";
}

const uploadMessage = ref("");

async function uploadVideo(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  const form = new FormData();
  form.append("file", file);

  const r = await fetch("/api/videos/upload", {
    method: "POST",
    headers: { "X-CSRFToken": getCsrf() },
    body: form,
  });
  if (!r.ok) {
    uploadMessage.value = "Upload failed";
    return;
  }
  uploadMessage.value = "Uploaded!";
  input.value = "";
  await refetchVideos();
}

async function deleteVideo(id: number) {
  await fetch(`/api/videos/${id}`, { method: "DELETE", headers: { "X-CSRFToken": getCsrf() } });
  await refetchVideos();
}

// ── Helpers ───────────────────────────────────────────────────────────────

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
</script>

<template>
  <div class="rpanel">
    <!-- Recording -->
    <Panel header="Recording">
      <div class="recording-controls">
        <Button
          v-if="!recordingStatus?.isRecording"
          label="Start Recording"
          icon="pi pi-circle-fill"
          severity="danger"
          @click="startRecording()"
          :disabled="replayStatus?.isReplaying"
        />
        <Button
          v-else
          label="Stop Recording"
          icon="pi pi-stop"
          severity="secondary"
          @click="stopRecording()"
        />
        <span v-if="recordingStatus?.isRecording" class="rec-stats">
          {{ recordingStatus?.elapsedSeconds?.toFixed(1) }}s
          &middot; {{ recordingStatus?.frameCount }} frames
        </span>
      </div>
    </Panel>

    <!-- Event Recording Config -->
    <Panel>
      <template #header>
        <div class="panel-header">
          <span class="panel-title">Event-Triggered Recording</span>
          <div class="toggle-row">
            <ToggleSwitch
              :model-value="eventConfig?.enabled ?? false"
              input-id="event-enabled"
              @update:model-value="(v: boolean) => updateEventConfig({ enabled: v })"
            />
            <label
              for="event-enabled"
              class="toggle-label"
              :class="eventConfig?.enabled ? 'enabled' : 'disabled'"
            >
              {{ eventConfig?.enabled ? "Enabled" : "Disabled" }}
            </label>
          </div>
        </div>
      </template>

      <div class="event-config-grid">
        <div class="event-config-item">
          <label for="pre-buffer" class="event-label">Pre-buffer (s)</label>
          <InputNumber
            input-id="pre-buffer"
            :model-value="eventConfig?.preBufferSeconds ?? 5"
            :min="1" :max="30" :step="1"
            class="event-input"
            @update:model-value="(v: number) => updateEventConfig({ preBufferSeconds: v })"
          />
          <p class="event-help">Seconds of video kept in memory before a trigger.</p>
        </div>

        <div class="event-config-item">
          <label for="post-trigger" class="event-label">Post-trigger (s)</label>
          <InputNumber
            input-id="post-trigger"
            :model-value="eventConfig?.postTriggerSeconds ?? 10"
            :min="1" :max="60" :step="1"
            class="event-input"
            @update:model-value="(v: number) => updateEventConfig({ postTriggerSeconds: v })"
          />
          <p class="event-help">Seconds to record after a detection triggers.</p>
        </div>

        <div class="event-config-item">
          <label for="cooldown" class="event-label">Cooldown (s)</label>
          <InputNumber
            input-id="cooldown"
            :model-value="eventConfig?.cooldownSeconds ?? 30"
            :min="0" :max="300" :step="5"
            class="event-input"
            @update:model-value="(v: number) => updateEventConfig({ cooldownSeconds: v })"
          />
          <p class="event-help">Minimum time between automatic recordings.</p>
        </div>
      </div>

      <!-- Status indicators -->
      <div v-if="eventConfig?.enabled" class="event-status-row">
        <span v-if="recordingStatus?.eventActive" class="event-status active">Recording event clip...</span>
        <span v-else-if="(recordingStatus?.eventCooldownRemaining ?? 0) > 0" class="event-status cooldown">
          Cooldown: {{ recordingStatus?.eventCooldownRemaining?.toFixed(1) }}s
        </span>
        <span v-else class="event-status ready">Waiting for trigger</span>
      </div>

      <Fieldset legend="Trigger Classes" class="trigger-fieldset" :toggleable="true">
        <p class="hint">
          Start recording when any of these classes are detected
          <span v-if="(eventConfig?.triggerClasses?.length ?? 0) > 0" class="selected-count">
            ({{ eventConfig?.triggerClasses?.length }} selected)
          </span>:
        </p>
        <div class="class-grid">
          <div v-for="cls in allClasses" :key="cls" class="class-item">
            <Checkbox
              :input-id="`evt-cls-${cls}`"
              :model-value="eventConfig?.triggerClasses?.includes(cls)"
              :binary="true"
              @update:model-value="toggleTriggerClass(cls)"
            />
            <label :for="`evt-cls-${cls}`" class="class-label">{{ cls }}</label>
          </div>
        </div>
      </Fieldset>
    </Panel>

    <!-- Event Clips -->
    <Panel header="Event Clips">
      <div v-if="eventClips.length === 0" class="empty">No event clips yet.</div>
      <div v-for="(c, i) in eventClips" :key="i" class="clip-row">
        <span class="clip-name">{{ c.filename }}</span>
        <span class="clip-meta">{{ c.frameCount }} frames &middot; {{ new Date(c.time).toLocaleString() }}</span>
        <div class="clip-actions">
          <Button
            as="a"
            :href="'/api/videos/' + c.videoId + '/download'"
            label="Download"
            icon="pi pi-download"
            size="small"
            severity="info"
          />
          <Button
            label="Play"
            icon="pi pi-play"
            size="small"
            severity="success"
            @click="startReplay(c.videoId)"
          />
        </div>
      </div>
    </Panel>

    <!-- Upload -->
    <Panel header="Upload Video">
      <div class="upload-row">
        <input type="file" accept="video/*" @change="uploadVideo" class="upload-input" />
        <span v-if="uploadMessage" class="upload-msg">{{ uploadMessage }}</span>
      </div>
    </Panel>

    <!-- Replay status -->
    <Panel v-if="replayStatus?.isReplaying" header="Replay Active">
      <div class="replay-status">
        <span class="replay-info">
          {{ replayStatus?.videoFilename }} &middot;
          frame {{ replayStatus?.currentFrame }} / {{ replayStatus?.totalFrames }} &middot;
          {{ replayStatus?.videoFps }} FPS
        </span>
        <Button label="Stop Replay" severity="warning" size="small" @click="stopReplay()" />
      </div>
      <p class="replay-hint">Switch to the <strong>Camera</strong> tab to see the replay feed.</p>
    </Panel>

    <!-- Saved Videos -->
    <Panel header="Saved Videos">
      <div v-if="!videosRaw || videosRaw.length === 0" class="empty">No videos yet.</div>
      <div v-for="v in videosRaw" :key="v.id" class="video-row">
        <span class="video-name">{{ v.filename }}</span>
        <span class="video-meta">{{ formatSize(v.sizeBytes ?? 0) }} &middot; {{ v.source }}</span>
        <span class="video-meta">{{ new Date(v.createdAt!).toLocaleString() }}</span>
        <div class="video-actions">
          <Button
            label="Play"
            icon="pi pi-play"
            size="small"
            severity="success"
            @click="startReplay(v.id)"
            :disabled="recordingStatus?.isRecording"
          />
          <Button
            as="a"
            :href="'/api/videos/' + v.id + '/download'"
            label="Download"
            icon="pi pi-download"
            size="small"
            severity="info"
          />
          <Button
            label="Delete"
            icon="pi pi-trash"
            size="small"
            severity="secondary"
            @click="deleteVideo(v.id)"
          />
        </div>
      </div>
    </Panel>
  </div>
</template>

<style scoped>
.rpanel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.recording-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.rec-stats {
  font-family: monospace;
  font-size: 0.875rem;
  color: var(--p-text-muted-color);
}
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
.event-config-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 1.25rem;
}
.event-config-item {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  min-width: 9rem;
  max-width: 14rem;
}
.event-label {
  font-size: 0.875rem;
  font-weight: 500;
}
.event-input {
  width: 6rem;
}
.event-help {
  font-size: 0.78rem;
  color: var(--p-text-muted-color, #888);
  margin: 0;
  line-height: 1.4;
}
.event-status-row {
  margin-top: 0.75rem;
  display: flex;
  gap: 0.5rem;
}
.event-status {
  font-size: 0.8rem;
  font-family: monospace;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
}
.event-status.active {
  color: var(--p-red-500);
  background: rgba(255, 0, 0, 0.08);
}
.event-status.cooldown {
  color: var(--p-yellow-500);
  background: rgba(255, 200, 0, 0.08);
}
.event-status.ready {
  color: var(--p-green-500);
  background: rgba(0, 255, 0, 0.06);
}
.trigger-fieldset {
  margin-top: 1rem;
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
  max-height: 12rem;
  overflow-y: auto;
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
.upload-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.upload-input {
  font-size: 0.875rem;
}
.upload-msg {
  font-size: 0.8rem;
  color: var(--p-text-muted-color);
}
.replay-status {
  display: flex;
  align-items: center;
  gap: 1rem;
}
.replay-info {
  font-family: monospace;
  font-size: 0.875rem;
}
.replay-hint {
  font-size: 0.8rem;
  color: var(--p-text-muted-color);
  margin: 0.5rem 0 0 0;
}
.empty {
  font-size: 0.875rem;
  color: var(--p-text-muted-color);
}
.video-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--p-content-border-color, #333);
  flex-wrap: wrap;
}
.video-name {
  font-weight: 600;
  font-size: 0.875rem;
  flex: 1;
  min-width: 12rem;
}
.video-meta {
  font-size: 0.75rem;
  color: var(--p-text-muted-color);
  font-family: monospace;
}
.video-actions {
  display: flex;
  gap: 0.25rem;
  margin-left: auto;
}
.clip-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.4rem 0;
  border-bottom: 1px solid var(--p-content-border-color, #333);
  flex-wrap: wrap;
}
.clip-name {
  font-weight: 600;
  font-size: 0.875rem;
  font-family: monospace;
  flex: 1;
  min-width: 14rem;
}
.clip-meta {
  font-size: 0.75rem;
  color: var(--p-text-muted-color);
  font-family: monospace;
}
.clip-actions {
  display: flex;
  gap: 0.25rem;
  margin-left: auto;
}
</style>
