<script setup lang="ts">
import { ref, watch } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { DefaultApi, type PikiOptions, type PikiOptionsPatch } from "@/api";

const modeOptions = ref([
    { name: "Boxes", value: "boxes" },
    { name: "Mask", value: "mask" },
]);

const api = new DefaultApi();

const options = ref<PikiOptions>({ mode: "boxes" });

const { mutate: updateOptions } = useMutation({
    mutationFn: (options: PikiOptionsPatch) => api.coreApiUpdateOptions({ pikiOptionsPatch: options }),
});

watch(options, options => updateOptions(options), { deep: true });

const feedUrl = "/api/video_feed"
</script>

<template>
    <div class="border-2">
        hello, this is the stream:
        <select v-model="options.mode">
            <option disabled value="">Please select one</option>
            <option v-for="o in modeOptions" :value="o.value">{{ o.name }}</option>
        </select>
        <div style="border: #ff000044 1px solid; border-radius: 5px">
            <img id="camera-feed" :src="feedUrl" alt="feed" />
        </div>
    </div>
</template>
