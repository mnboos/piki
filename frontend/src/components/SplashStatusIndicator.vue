<script setup lang="ts">
import { computed } from "vue";
import { useSplashStatus, useSplashDisplayRemaining } from "@/composables/useEventStream";

const data = useSplashStatus();
const remaining = useSplashDisplayRemaining();

const visible = computed(() => data.value?.enabled || data.value?.state !== "idle");

const dotClass = computed(() => {
  switch (data.value?.state) {
    case "armed": return "dot armed";
    case "firing": return "dot firing";
    case "cooldown": return "dot cooldown";
    default: return "dot idle";
  }
});

const label = computed(() => {
  switch (data.value?.state) {
    case "armed": return `Armed (${remaining.value.toFixed(1)}s)`;
    case "firing": return `Firing (${remaining.value.toFixed(1)}s)`;
    case "cooldown": return `Cooldown (${remaining.value.toFixed(0)}s)`;
    default: return "Splash ready";
  }
});
</script>

<template>
  <div v-if="visible" class="splash-indicator">
    <span :class="dotClass"></span>
    <span class="label">{{ label }}</span>
  </div>
</template>

<style scoped>
.splash-indicator {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.5);
  color: #eee;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dot.idle {
  background: #888;
}

.dot.armed {
  background: #f59e0b;
  animation: pulse 0.8s ease-in-out infinite;
}

.dot.firing {
  background: #ef4444;
  animation: pulse 0.4s ease-in-out infinite;
}

.dot.cooldown {
  background: #a3a322;
}

.label {
  white-space: nowrap;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(1.3); }
}
</style>
