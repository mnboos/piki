<script setup lang="ts">
import { computed } from "vue";
import { useQuery } from "@tanstack/vue-query";
import { DefaultApi } from "@/api";
import Card from "primevue/card";
import ProgressBar from "primevue/progressbar";
import Tooltip from "primevue/tooltip";

const vTooltip = Tooltip;

const api = new DefaultApi();

const { data: m, isLoading, isError } = useQuery({
    queryKey: ["metrics"],
    queryFn: () => api.coreApiGetMetrics(),
    refetchInterval: 2000,
    staleTime: 1500,
});

function fmtBytes(b: number | null | undefined): string {
    if (b == null) return "n/a";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let v = b;
    let u = 0;
    while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
    return `${v.toFixed(v < 10 ? 1 : 0)} ${units[u]}`;
}

function fmtRate(bps: number | null | undefined): string {
    return `${fmtBytes(bps ?? 0)}/s`;
}

function fmtFreq(mhz: number | null | undefined): string {
    if (mhz == null) return "n/a";
    return mhz >= 1000 ? `${(mhz / 1000).toFixed(2)} GHz` : `${Math.round(mhz)} MHz`;
}

function fmtUptime(s: number | null | undefined): string {
    if (s == null) return "n/a";
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const min = Math.floor((s % 3600) / 60);
    if (d > 0) return `${d}d ${h}h ${min}m`;
    if (h > 0) return `${h}h ${min}m`;
    return `${min}m`;
}

function tempClass(c: number): string {
    if (c >= 85) return "temp-hot";
    if (c >= 75) return "temp-warm";
    return "";
}

const tempEntries = computed<[string, number][]>(() =>
    Object.entries(m.value?.tempsC ?? {}) as [string, number][],
);
</script>

<template>
    <div v-if="isLoading" class="mp-loading">Loading metrics…</div>
    <div v-else-if="isError || !m" class="mp-error">Failed to load metrics.</div>
    <div v-else class="mp-grid">
        <!-- CPU -->
        <Card>
            <template #title>CPU</template>
            <template #content>
                <div class="mp-row mp-row--big">
                    <span class="mp-val">{{ m.cpu.percentTotal.toFixed(0) }}%</span>
                    <span class="mp-sub">{{ m.cpu.coreCount }} cores</span>
                </div>
                <ProgressBar :value="m.cpu.percentTotal" :show-value="false" class="mp-progress" />
                <div class="mp-cores">
                    <div v-for="(p, i) in m.cpu.percentPerCore" :key="i" class="mp-core">
                        <div class="mp-core-bar" :style="{ height: `${p}%` }" />
                        <span class="mp-core-label">{{ i }}</span>
                    </div>
                </div>
                <div class="mp-row mp-row--small">
                    <span>load avg</span>
                    <span>{{ m.loadAvg.map(n => n.toFixed(2)).join(" / ") }}</span>
                </div>
                <div class="mp-row mp-row--small">
                    <span>freq</span>
                    <span>{{ fmtFreq(m.cpu.freqMhzPerCore[0]) }}</span>
                </div>
            </template>
        </Card>

        <!-- Memory -->
        <Card>
            <template #title>Memory</template>
            <template #content>
                <div class="mp-row mp-row--big">
                    <span class="mp-val">{{ m.memory.percent.toFixed(0) }}%</span>
                    <span class="mp-sub">
                        {{ fmtBytes(m.memory.usedBytes) }} / {{ fmtBytes(m.memory.totalBytes) }}
                    </span>
                </div>
                <ProgressBar :value="m.memory.percent" :show-value="false" class="mp-progress" />
                <div class="mp-row mp-row--small">
                    <span>available</span>
                    <span>{{ fmtBytes(m.memory.availableBytes) }}</span>
                </div>
                <template v-if="m.swap.totalBytes > 0">
                    <div class="mp-row mp-row--small">
                        <span>swap</span>
                        <span>
                            {{ fmtBytes(m.swap.usedBytes) }} / {{ fmtBytes(m.swap.totalBytes) }}
                        </span>
                    </div>
                </template>
            </template>
        </Card>

        <!-- Temperatures -->
        <Card>
            <template #title>Temperatures</template>
            <template #content>
                <div v-for="[zone, c] in tempEntries" :key="zone" class="mp-row">
                    <span class="mp-label">{{ zone.toUpperCase() }}</span>
                    <span :class="['mp-val mp-val--med', tempClass(c)]">{{ c.toFixed(1) }} °C</span>
                </div>
                <div v-if="tempEntries.length === 0" class="mp-sub">No thermal zones available</div>
            </template>
        </Card>

        <!-- Accelerators -->
        <Card>
            <template #title>Accelerators</template>
            <template #content>
                <div class="mp-row">
                    <span class="mp-label">BPU load</span>
                    <span class="mp-val mp-val--med">
                        {{ m.bpu.loadPercent != null ? `${m.bpu.loadPercent}%` : "n/a" }}
                    </span>
                </div>
                <ProgressBar
                    v-if="m.bpu.loadPercent != null"
                    :value="m.bpu.loadPercent"
                    :show-value="false"
                    class="mp-progress"
                />
                <div class="mp-row mp-row--small">
                    <span>BPU freq</span>
                    <span>{{ fmtFreq(m.bpu.freqMhz) }}</span>
                </div>
                <div class="mp-row mp-row--small">
                    <span v-tooltip.left="'Encoder utilisation not exposed by the Horizon SDK on this image. The static clock is shown instead.'">
                        VPU clock <span class="mp-info">ⓘ</span>
                    </span>
                    <span>{{ fmtFreq(m.vpu.clockMhz) }}</span>
                </div>
                <div class="mp-row mp-row--small">
                    <span v-tooltip.left="'GPU GC8000 utilisation not exposed by the kernel.'">
                        GPU freq <span class="mp-info">ⓘ</span>
                    </span>
                    <span>{{ fmtFreq(m.gpu.freqMhz) }}</span>
                </div>
                <div class="mp-row mp-row--small">
                    <span>DDR freq</span>
                    <span>{{ fmtFreq(m.ddr.freqMhz) }}</span>
                </div>
                <div class="mp-row mp-row--small">
                    <span v-tooltip.left="'ISP/VPS utilisation not exposed.'">
                        ISP load <span class="mp-info">ⓘ</span>
                    </span>
                    <span>n/a</span>
                </div>
            </template>
        </Card>

        <!-- Storage -->
        <Card>
            <template #title>Storage (/)</template>
            <template #content>
                <div class="mp-row mp-row--big">
                    <span class="mp-val">{{ m.diskRoot.percent.toFixed(0) }}%</span>
                    <span class="mp-sub">
                        {{ fmtBytes(m.diskRoot.usedBytes) }} / {{ fmtBytes(m.diskRoot.totalBytes) }}
                    </span>
                </div>
                <ProgressBar :value="m.diskRoot.percent" :show-value="false" class="mp-progress" />
                <div class="mp-row mp-row--small">
                    <span>free</span>
                    <span>{{ fmtBytes(m.diskRoot.freeBytes) }}</span>
                </div>
            </template>
        </Card>

        <!-- Network -->
        <Card>
            <template #title>Network</template>
            <template #content>
                <div class="mp-row">
                    <span class="mp-label">↓ rx</span>
                    <span class="mp-val mp-val--med">{{ fmtRate(m.net.rxBytesPerS) }}</span>
                </div>
                <div class="mp-row">
                    <span class="mp-label">↑ tx</span>
                    <span class="mp-val mp-val--med">{{ fmtRate(m.net.txBytesPerS) }}</span>
                </div>
                <div
                    v-for="iface in m.net.interfaces"
                    :key="iface.name"
                    class="mp-row mp-row--small"
                >
                    <span>{{ iface.name }}</span>
                    <span>
                        ↓ {{ fmtRate(iface.rxBytesPerS) }} &nbsp; ↑ {{ fmtRate(iface.txBytesPerS) }}
                    </span>
                </div>
                <div class="mp-row mp-row--small">
                    <span>uptime</span>
                    <span>{{ fmtUptime(m.uptimeS) }}</span>
                </div>
            </template>
        </Card>
    </div>
</template>

<style scoped>
.mp-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
    padding: 0.5rem 0;
}
.mp-loading, .mp-error {
    padding: 2rem;
    text-align: center;
    color: var(--p-text-muted-color, #888);
}
.mp-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.15rem 0;
}
.mp-row--big {
    margin-bottom: 0.4rem;
}
.mp-row--small {
    font-size: 0.8rem;
    color: var(--p-text-muted-color, #888);
}
.mp-val {
    font-size: 1.6rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.mp-val--med {
    font-size: 1.05rem;
    font-weight: 500;
}
.mp-sub {
    font-size: 0.8rem;
    color: var(--p-text-muted-color, #888);
}
.mp-label {
    font-weight: 500;
}
.mp-progress {
    height: 0.45rem !important;
    margin: 0.25rem 0 0.5rem 0;
}
.mp-cores {
    display: flex;
    gap: 0.25rem;
    align-items: flex-end;
    height: 60px;
    margin: 0.4rem 0 0.6rem 0;
    border-bottom: 1px solid var(--p-content-border-color, #444);
}
.mp-core {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
    position: relative;
}
.mp-core-bar {
    width: 100%;
    background: var(--p-primary-color, #4caf50);
    border-radius: 2px 2px 0 0;
    transition: height 0.3s ease;
    min-height: 1px;
}
.mp-core-label {
    position: absolute;
    bottom: -1.1rem;
    font-size: 0.65rem;
    color: var(--p-text-muted-color, #888);
}
.temp-warm {
    color: #f5a623;
}
.temp-hot {
    color: #e74c3c;
}
.mp-info {
    font-size: 0.7rem;
    color: var(--p-text-muted-color, #888);
    cursor: help;
}
</style>
