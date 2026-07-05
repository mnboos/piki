<script setup lang="ts">
import { computed } from "vue";
import { useGamepadStatus } from "@/composables/useEventStream";

const data = useGamepadStatus();

const dotClass = computed(() => {
  if (!data.value?.connected) return "dot disconnected";
  if (data.value?.enabled) return "dot enabled";
  return "dot connected";
});

const label = computed(() => {
  if (!data.value?.connected) return "No controller";
  if (!data.value?.enabled) return "Controller ready (press Start)";
  return `Pan ${data.value.pan?.toFixed(1)}°  Tilt ${data.value.tilt?.toFixed(1)}°`;
});
</script>

<template>
  <div class="gamepad-indicator">
    <span :class="dotClass"></span>
    <span class="label">{{ label }}</span>
  </div>
</template>

<style scoped>
.gamepad-indicator {
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

.dot.disconnected {
  background: #555;
}

.dot.connected {
  background: #f59e0b;
}

.dot.enabled {
  background: #22c55e;
  animation: pulse 0.8s ease-in-out infinite;
}

.label {
  white-space: nowrap;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.6; transform: scale(1.3); }
}
</style>
