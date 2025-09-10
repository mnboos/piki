<template>
    <div class="border-2">
        hello, this is the stream:
        <div style="border: #ff000044 1px solid; border-radius: 5px">
            <video ref="video" autoplay muted playsinline></video>
        </div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, useTemplateRef } from "vue";

const pc = new RTCPeerConnection();
const ws = new WebSocket("ws://localhost:8000/ws/video/");

const video = useTemplateRef<HTMLVideoElement>("video");

onMounted(() => {
    pc.ontrack = event => {
        if (video.value) {
            video.value.srcObject = event.streams[0];
        }
    };

    ws.onopen = async () => {
        const offer = await pc.createOffer({
            offerToReceiveVideo: true,
            offerToReceiveAudio: false, // You can set this to true if you ever add audio
        });
        await pc.setLocalDescription(offer);
        ws.send(JSON.stringify(offer));
    };

    ws.onmessage = async event => {
        try {
            const answer = JSON.parse(event.data);
            console.log("Received answer from server:", answer);
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
        } catch (e) {
            console.error("Error parsing answer or setting remote description:", e);
        }
    };
});
</script>
