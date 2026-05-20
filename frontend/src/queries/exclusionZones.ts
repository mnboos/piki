import { useMutation, useQuery, useQueryClient } from "@tanstack/vue-query";
import { DefaultApi, type ExclusionZoneSchema, type ExclusionZoneSchemaPatch } from "@/api";

const api = new DefaultApi();

const KEY = ["exclusionZones"] as const;

export function useExclusionZonesQuery() {
    return useQuery({
        queryKey: KEY,
        queryFn: () => api.coreApiListExclusionZones(),
        staleTime: 1000,
    });
}

export function useCreateExclusionZoneMutation() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (payload: ExclusionZoneSchema) =>
            api.coreApiCreateExclusionZone({ exclusionZoneSchema: payload }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: KEY });
        },
    });
}

export function useUpdateExclusionZoneMutation() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (args: { zoneId: number; patch: ExclusionZoneSchemaPatch }) =>
            api.coreApiUpdateExclusionZone({
                zoneId: args.zoneId,
                exclusionZoneSchemaPatch: args.patch,
            }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: KEY });
        },
    });
}

export function useDeleteExclusionZoneMutation() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (zoneId: number) => api.coreApiDeleteExclusionZone({ zoneId }),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: KEY });
        },
    });
}
