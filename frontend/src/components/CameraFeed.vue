<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from "vue";
import Skeleton from "primevue/skeleton";
import ExclamationTriangleIcon from "@primevue/icons/exclamationtriangle";
import { useBackendHost } from "@/utils";

const props = defineProps<{ alt?: string }>();

type Status = "connecting" | "connected" | "error";
const status = ref<Status>("connecting");
const video = ref<HTMLVideoElement | null>(null);

// Backoff between reconnect attempts when the negotiation fails.
const RECONNECT_INITIAL_MS = 1000;
const RECONNECT_MAX_MS = 10_000;
let reconnectMs = RECONNECT_INITIAL_MS;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

let pc: RTCPeerConnection | null = null;
let cancelled = false;

async function negotiate() {
    if (cancelled) return;
    closePc();
    status.value = "connecting";

    // STUN matters even on LAN: modern Chrome replaces host-candidate local IPs
    // with `xyz.local` mDNS names that aiortc cannot resolve. The browser also
    // needs a STUN server to gather srflx candidates, which lets the pair
    // succeed when host candidates are obfuscated.
    pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });
    pc.addTransceiver("video", { direction: "recvonly" });

    pc.addEventListener("track", e => {
        if (e.track.kind !== "video") return;
        if (video.value && e.streams[0]) {
            video.value.srcObject = e.streams[0];
        }
    });

    pc.addEventListener("connectionstatechange", () => {
        const s = pc?.connectionState;
        if (s === "connected") {
            status.value = "connected";
            reconnectMs = RECONNECT_INITIAL_MS;
        } else if (s === "failed" || s === "closed" || s === "disconnected") {
            if (!cancelled) {
                status.value = "error";
                scheduleReconnect();
            }
        }
    });

    try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        // Wait for ICE gathering to complete so the SDP we send has all
        // candidates inline (we use no trickle ICE on the server).
        await iceComplete(pc);

        const resp = await fetch(useBackendHost() + "/api/webrtc/offer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sdp: pc.localDescription!.sdp,
                type: pc.localDescription!.type,
            }),
        });
        if (!resp.ok) {
            throw new Error(`offer failed: ${resp.status}`);
        }
        const answer = await resp.json();
        if (cancelled) return;
        await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
    } catch (err) {
        console.error("WebRTC negotiation failed", err);
        if (!cancelled) {
            status.value = "error";
            scheduleReconnect();
        }
    }
}

function iceComplete(pc: RTCPeerConnection): Promise<void> {
    if (pc.iceGatheringState === "complete") return Promise.resolve();
    return new Promise(resolve => {
        const check = () => {
            if (pc.iceGatheringState === "complete") {
                pc.removeEventListener("icegatheringstatechange", check);
                resolve();
            }
        };
        pc.addEventListener("icegatheringstatechange", check);
    });
}

function scheduleReconnect() {
    if (reconnectTimer !== null || cancelled) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        reconnectMs = Math.min(reconnectMs * 2, RECONNECT_MAX_MS);
        negotiate();
    }, reconnectMs);
}

function closePc() {
    if (pc) {
        try { pc.close(); } catch { /* ignore */ }
        pc = null;
    }
    if (video.value) {
        video.value.srcObject = null;
    }
}

onMounted(() => {
    cancelled = false;
    negotiate();
});

onBeforeUnmount(() => {
    cancelled = true;
    if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
    closePc();
});
</script>

<template>
    <div class="cf-wrapper">
        <Skeleton v-if="status === 'connecting'" class="cf-skeleton" />
        <div v-else-if="status === 'error'" class="cf-error">
            <ExclamationTriangleIcon class="cf-error-icon" />
            <span class="cf-error-text">No feed — reconnecting…</span>
        </div>

        <video
            ref="video"
            autoplay
            muted
            playsinline
            class="cf-img"
            :class="{ 'cf-img--hidden': status !== 'connected' }"
            :aria-label="alt"
        />

        <div class="cf-overlay">
            <slot name="overlay" />
        </div>
    </div>
</template>

<style scoped>
.cf-wrapper {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: var(--p-surface-100, #1a1a1a);
    overflow: hidden;
}
.cf-skeleton {
    position: absolute;
    inset: 0;
    width: 100% !important;
    height: 100% !important;
    border-radius: 0;
}
.cf-error {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    color: var(--p-text-muted-color, #888);
}
.cf-error-icon {
    width: 2rem;
    height: 2rem;
}
.cf-error-text {
    font-size: 0.85rem;
}
.cf-img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    background: black;
}
.cf-img--hidden {
    visibility: hidden;
}
.cf-overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
}
</style>
