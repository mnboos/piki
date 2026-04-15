<script setup lang="ts">
import { ref, watch } from "vue";
import Skeleton from "primevue/skeleton";

const props = defineProps<{ src: string; alt?: string }>();

type Status = "loading" | "loaded" | "error";
const status = ref<Status>("loading");

// Reset to loading whenever the src changes (e.g. tab shown again).
watch(() => props.src, () => { status.value = "loading"; });
</script>

<template>
    <div class="cf-wrapper">
        <!-- Skeleton shown while waiting for the first frame -->
        <Skeleton v-if="status === 'loading'" class="cf-skeleton" />

        <!-- Error state when the backend is unreachable -->
        <div v-else-if="status === 'error'" class="cf-error">
            <i class="pi pi-video pi-spin cf-error-icon" />
            <span class="cf-error-text">No feed — backend offline</span>
        </div>

        <!--
            Keep the img in the DOM so the MJPEG stream connects immediately.
            Hide it until the first frame arrives to avoid a broken-image flash.
        -->
        <img
            :src="src"
            :alt="alt"
            class="cf-img"
            :class="{ 'cf-img--hidden': status !== 'loaded' }"
            @load="status = 'loaded'"
            @error="status = 'error'"
        />
    </div>
</template>

<style scoped>
.cf-wrapper {
    position: relative;
    width: 100%;
    /* 4:3 aspect ratio placeholder until the real frame dimensions are known */
    aspect-ratio: 4 / 3;
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
    font-size: 2rem;
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
}
.cf-img--hidden {
    visibility: hidden;
}
</style>
