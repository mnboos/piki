<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import Panel from "primevue/panel";
import Button from "primevue/button";

interface VideoInfo {
  id: number;
  filename: string;
  url: string;
  size_bytes: number;
  source: string;
  created_at: string;
}

interface RecordingStatus {
  is_recording: boolean;
  elapsed_seconds: number;
  frame_count: number;
  file_path: string;
}

interface ReplayStatus {
  is_replaying: boolean;
  video_filename: string;
  current_frame: number;
  total_frames: number;
  video_fps: number;
}

const CSRF_COOKIE = "csrftoken";

function getCsrf(): string {
  const match = document.cookie.match(new RegExp(`(?:^|; )${CSRF_COOKIE}=([^;]*)`));
  return match ? match[1] : "";
}

const recording = ref<RecordingStatus>({ is_recording: false, elapsed_seconds: 0, frame_count: 0, file_path: "" });
const videos = ref<VideoInfo[]>([]);
const replay = ref<ReplayStatus>({ is_replaying: false, video_filename: "", current_frame: 0, total_frames: 0, video_fps: 0 });
const uploadMessage = ref("");
let pollInterval: ReturnType<typeof setInterval> | null = null;

async function fetchVideos() {
  try {
    const r = await fetch("/api/videos");
    videos.value = await r.json();
  } catch { /* ignore */ }
}

async function startRecording() {
  const r = await fetch("/api/recording/start", { method: "POST", headers: { "X-CSRFToken": getCsrf() } });
  if (r.status === 409) {
    const err = await r.json();
    uploadMessage.value = err.detail || "Conflict";
    return;
  }
  recording.value = await r.json();
  uploadMessage.value = "";
  if (!pollInterval) pollInterval = setInterval(pollRecordingStatus, 1000);
}

async function stopRecording() {
  const r = await fetch("/api/recording/stop", { method: "POST", headers: { "X-CSRFToken": getCsrf() } });
  recording.value = await r.json();
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
  await fetchVideos();
}

async function pollRecordingStatus() {
  try {
    const r = await fetch("/api/recording/status");
    recording.value = await r.json();
    if (!recording.value.is_recording && pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  } catch { /* ignore */ }
}

async function uploadVideo(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;

  const form = new FormData();
  form.append("file", file);

  const r = await fetch("/api/videos/upload", { method: "POST", headers: { "X-CSRFToken": getCsrf() }, body: form });
  if (!r.ok) {
    uploadMessage.value = "Upload failed";
    return;
  }
  uploadMessage.value = "Uploaded!";
  input.value = "";
  await fetchVideos();
}

async function deleteVideo(id: number) {
  await fetch(`/api/videos/${id}`, { method: "DELETE", headers: { "X-CSRFToken": getCsrf() } });
  await fetchVideos();
}

async function startReplay(id: number) {
  const r = await fetch(`/api/replay/start/${id}`, { method: "POST", headers: { "X-CSRFToken": getCsrf() } });
  if (r.status === 409) {
    uploadMessage.value = "Cannot replay right now";
    return;
  }
  replay.value = await r.json();
  uploadMessage.value = "";
  if (!pollInterval) pollInterval = setInterval(pollReplayStatus, 500);
}

async function stopReplay() {
  await fetch("/api/replay/stop", { method: "POST", headers: { "X-CSRFToken": getCsrf() } });
  replay.value = { is_replaying: false, video_filename: "", current_frame: 0, total_frames: 0, video_fps: 0 };
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

async function pollReplayStatus() {
  try {
    const r = await fetch("/api/replay/status");
    replay.value = await r.json();
    if (!replay.value.is_replaying && pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  } catch { /* ignore */ }
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

onMounted(fetchVideos);
onUnmounted(() => { if (pollInterval) clearInterval(pollInterval); });
</script>

<template>
  <div class="rpanel">
    <!-- Recording -->
    <Panel header="Recording">
      <div class="recording-controls">
        <Button
          v-if="!recording.is_recording"
          label="Start Recording"
          icon="pi pi-circle-fill"
          severity="danger"
          @click="startRecording"
          :disabled="replay.is_replaying"
        />
        <Button
          v-else
          label="Stop Recording"
          icon="pi pi-stop"
          severity="secondary"
          @click="stopRecording"
        />
        <span v-if="recording.is_recording" class="rec-stats">
          {{ recording.elapsed_seconds.toFixed(1) }}s &middot; {{ recording.frame_count }} frames
        </span>
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
    <Panel v-if="replay.is_replaying" header="Replay Active">
      <div class="replay-status">
        <span class="replay-info">
          {{ replay.video_filename }} &middot;
          frame {{ replay.current_frame }} / {{ replay.total_frames }} &middot;
          {{ replay.video_fps }} FPS
        </span>
        <Button label="Stop Replay" severity="warning" size="small" @click="stopReplay" />
      </div>
      <p class="replay-hint">Switch to the <strong>Camera</strong> tab to see the replay feed.</p>
    </Panel>

    <!-- Video list -->
    <Panel header="Saved Videos">
      <div v-if="videos.length === 0" class="empty">No videos yet.</div>
      <div v-for="v in videos" :key="v.id" class="video-row">
        <span class="video-name">{{ v.filename }}</span>
        <span class="video-meta">{{ formatSize(v.size_bytes) }} &middot; {{ v.source }}</span>
        <span class="video-meta">{{ new Date(v.created_at).toLocaleString() }}</span>
        <div class="video-actions">
          <Button
            label="Play"
            icon="pi pi-play"
            size="small"
            severity="success"
            @click="startReplay(v.id)"
            :disabled="recording.is_recording"
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
</style>
