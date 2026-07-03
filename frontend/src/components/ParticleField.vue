<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";

const canvasRef = ref(null);

let animationId = 0;
let resizeObserver = null;
let points = [];

function createPoints(width, height) {
  const spacing = Math.max(18, Math.min(26, Math.floor(width / 55)));
  const cols = Math.ceil(width / spacing) + 2;
  const rows = Math.ceil(height / spacing) + 2;
  const originX = width / 2;
  const originY = height / 2;

  points = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const x = (col - cols / 2) * spacing + originX;
      const y = (row - rows / 2) * spacing + originY;
      const dx = x - originX;
      const dy = y - originY;
      const distance = Math.hypot(dx, dy);
      points.push({
        baseX: x,
        baseY: y,
        distance,
        seed: (row * 17 + col * 31) * 0.11,
      });
    }
  }
}

function resizeCanvas() {
  const canvas = canvasRef.value;
  if (!canvas?.parentElement) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  const ctx = canvas.getContext("2d");
  ctx?.setTransform(ratio, 0, 0, ratio, 0, 0);
  createPoints(rect.width, rect.height);
}

function drawFrame(time) {
  const canvas = canvasRef.value;
  const ctx = canvas?.getContext("2d");
  if (!canvas || !ctx) return;

  const width = parseFloat(canvas.style.width || "0");
  const height = parseFloat(canvas.style.height || "0");
  const t = time * 0.0012;

  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createRadialGradient(width * 0.5, height * 0.42, 10, width * 0.5, height * 0.5, width * 0.55);
  gradient.addColorStop(0, "rgba(255, 86, 170, 0.26)");
  gradient.addColorStop(0.48, "rgba(110, 145, 255, 0.16)");
  gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  for (const point of points) {
    const radial = point.distance * 0.03;
    const wave = Math.sin(radial - t + point.seed) * 12;
    const drift = Math.cos(radial * 0.65 + t * 0.75 + point.seed) * 7;
    const angle = Math.atan2(point.baseY - height / 2, point.baseX - width / 2);

    const x = point.baseX + Math.cos(angle) * wave;
    const y = point.baseY + Math.sin(angle) * wave + drift;

    const hue = 318 + Math.sin(point.seed + t * 0.35) * 34 + Math.max(-28, 32 - point.distance * 0.03);
    const pulse = (Math.sin(t * 2.2 + point.seed) + 1) * 0.5;
    const alpha = Math.max(0.22, 0.9 - point.distance / Math.max(width, height)) + pulse * 0.12;
    const radius = Math.max(1.2, 3.1 - point.distance * 0.0025 + pulse * 0.45);

    ctx.beginPath();
    ctx.shadowColor = `hsla(${hue}, 92%, 72%, 0.42)`;
    ctx.shadowBlur = 8 + pulse * 8;
    ctx.fillStyle = `hsla(${hue}, 94%, 62%, ${Math.min(alpha, 0.95)})`;
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    if ((point.seed * 1000) % 7 < 1.4) {
      ctx.beginPath();
      ctx.strokeStyle = `hsla(${hue}, 94%, 72%, ${0.18 + pulse * 0.14})`;
      ctx.lineWidth = 1;
      ctx.moveTo(x - radius * 2.4, y);
      ctx.lineTo(x + radius * 2.4, y);
      ctx.moveTo(x, y - radius * 2.4);
      ctx.lineTo(x, y + radius * 2.4);
      ctx.stroke();
    }
  }

  animationId = requestAnimationFrame(drawFrame);
}

onMounted(() => {
  resizeCanvas();
  animationId = requestAnimationFrame(drawFrame);
  resizeObserver = new ResizeObserver(() => resizeCanvas());
  if (canvasRef.value?.parentElement) {
    resizeObserver.observe(canvasRef.value.parentElement);
  }
  window.addEventListener("resize", resizeCanvas);
});

onBeforeUnmount(() => {
  if (animationId) cancelAnimationFrame(animationId);
  resizeObserver?.disconnect();
  window.removeEventListener("resize", resizeCanvas);
});
</script>

<template>
  <div class="particle-field" aria-hidden="true">
    <canvas ref="canvasRef" class="particle-canvas"></canvas>
  </div>
</template>
