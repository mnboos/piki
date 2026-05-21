import { useQuery, useMutation, useQueryClient } from "@tanstack/vue-query";
import { DefaultApi, type EventRecordingConfigSchemaPatch } from "@/api";

const api = new DefaultApi();

// ── Tracker status (polled) ────────────────────────────────────────────────

export function useTrackerStatusQuery() {
  return useQuery({
    queryKey: ["trackerStatus"],
    queryFn: () => api.coreApiGetTrackerStatus(),
    refetchInterval: 1500,
    staleTime: 0,
  });
}

// ── Splash status (polled, fast for responsive indicator) ─────────────────

export function useSplashStatusQuery() {
  return useQuery({
    queryKey: ["splashStatus"],
    queryFn: () => api.coreApiGetSplashStatus(),
    refetchInterval: 500,
    staleTime: 0,
  });
}

// ── Event clips (polled, shared by RecordingsPanel list + HomeView toast) ──

export function useEventClipsQuery() {
  return useQuery({
    queryKey: ["eventClips"],
    queryFn: () => api.coreApiGetEventClips(),
    refetchInterval: 1000,
    staleTime: 0,
  });
}

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

// ── Recording status (polled) ─────────────────────────────────────────────

export function useRecordingStatusQuery() {
  return useQuery({
    queryKey: ["recordingStatus"],
    queryFn: () => api.coreApiRecordingStatus(),
    refetchInterval: 1000,
    staleTime: 0,
  });
}

// ── Manual recording start / stop ─────────────────────────────────────────

export function useStartRecordingMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.coreApiRecordingStart(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["recordingStatus"] });
    },
  });
}

export function useStopRecordingMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.coreApiRecordingStop(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["recordingStatus"] });
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

export function useReplayStatusQuery() {
  return useQuery({
    queryKey: ["replayStatus"],
    queryFn: () => api.coreApiReplayStatus(),
    refetchInterval: ({ state }) => (state.data?.isReplaying ? 500 : false),
    staleTime: 0,
  });
}

export function useStartReplayMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (videoId: number) => api.coreApiReplayStart({ videoId }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["replayStatus"] });
    },
  });
}

export function useStopReplayMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => api.coreApiReplayStop(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["replayStatus"] });
    },
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
