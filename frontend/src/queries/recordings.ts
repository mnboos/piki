import { useQuery, useMutation, useQueryClient } from "@tanstack/vue-query";
import { DefaultApi, type EventRecordingConfigSchemaPatch } from "@/api";

const api = new DefaultApi();

// Live state previously polled here (tracker/splash/event-clips/recording/replay
// status) now arrives over WebSocket — see `@/composables/useEventStream`.

// ── Event recording config ────────────────────────────────────────────────

export function useEventRecordingConfigQuery() {
  return useQuery({
    queryKey: ["eventRecordingConfig"],
    queryFn: () => api.coreApiGetEventRecordingConfig(),
    staleTime: 2000,
  });
}

export function useUpdateEventRecordingConfigMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (patch: EventRecordingConfigSchemaPatch) =>
      api.coreApiUpdateEventRecordingConfig({ eventRecordingConfigSchemaPatch: patch }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["eventRecordingConfig"] });
    },
  });
}

// ── Manual recording start / stop ─────────────────────────────────────────

export function useStartRecordingMutation() {
  return useMutation({
    mutationFn: () => api.coreApiRecordingStart(),
  });
}

export function useStopRecordingMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.coreApiRecordingStop(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["videos"] });
    },
  });
}

// ── Videos list ───────────────────────────────────────────────────────────

export function useVideosQuery() {
  return useQuery({
    queryKey: ["videos"],
    queryFn: () => api.coreApiVideosList(),
    staleTime: 2000,
  });
}

// ── Replay ────────────────────────────────────────────────────────────────

export function useStartReplayMutation() {
  return useMutation({
    mutationFn: (videoId: number) => api.coreApiReplayStart({ videoId }),
  });
}

export function useStopReplayMutation() {
  return useMutation({
    mutationFn: () => api.coreApiReplayStop(),
  });
}

// ── YOLO classes ──────────────────────────────────────────────────────────

export function useYoloClassesQuery() {
  return useQuery({
    queryKey: ["yoloClasses"],
    queryFn: () => api.coreApiGetYoloClasses(),
    staleTime: Infinity,
  });
}

// ── Upload / delete videos ────────────────────────────────────────────────

export function useUploadVideoMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return fetch("/api/videos/upload", {
        method: "POST",
        headers: { "X-CSRFToken": getCookie("csrftoken") },
        body: form,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["videos"] });
    },
  });
}

export function useDeleteVideoMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (videoId: number) =>
      fetch(`/api/videos/${videoId}`, {
        method: "DELETE",
        headers: { "X-CSRFToken": getCookie("csrftoken") },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["videos"] });
    },
  });
}

function getCookie(name: string): string {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`));
  return match ? match[1] : "";
}
