<template>
    <div class="border-2">
        hello, this is the stream:
        <div style="border: #ff000044 1px solid; border-radius: 5px">
            <canvas ref="video" width="640" height="480"></canvas>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, useTemplateRef } from "vue";

const ws = new WebSocket("ws://localhost:8000/ws/video/");
ws.binaryType = "arraybuffer";

const video = useTemplateRef<HTMLCanvasElement>("video");

const ctx = computed(() => video.value?.getContext("2d"));

function renderRawFrame(rgbData, width, height) {
    // Create ImageData from raw RGB

    if (!ctx.value) return;

    console.log("yay");

    const imageData = ctx.value.createImageData(width, height);
    const pixels = imageData.data;

    // Convert RGB to RGBA
    for (let i = 0; i < rgbData.length; i += 3) {
        const pixelIndex = (i / 3) * 4;
        pixels[pixelIndex] = rgbData[i]; // R
        pixels[pixelIndex + 1] = rgbData[i + 1]; // G
        pixels[pixelIndex + 2] = rgbData[i + 2]; // B
        pixels[pixelIndex + 3] = 255; // A
    }

    ctx.value.putImageData(imageData, 0, 0);
}

ws.onmessage = async event => {
    // if (event.data instanceof ArrayBuffer) {
    const view = new DataView(event.data);

    // Parse header
    const magic = view.getUint32(0, true);
    // if (magic !== 0xdeadbeef) return;

    const width = view.getUint16(4, true);
    const height = view.getUint16(6, true);
    const timestamp = view.getBigUint64(8, true);

    // Extract raw RGB data
    const rgbData = new Uint8Array(event.data, 16);

    // Render to canvas
    renderRawFrame(rgbData, width, height);
    // }
};

ws.onopen = function () {
    startStream();
};

function startStream() {
    ws.send(
        JSON.stringify({
            type: "start_stream",
            width: 640,
            height: 480,
            fps: 30,
        }),
    );
}
onMounted(() => {});
</script>
