<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";
import { appState, refreshTraining } from "@/stores/appStore";

const sourceHitRef = ref(null);
const topicHitRef = ref(null);
let sourceHitChart = null;
let topicHitChart = null;

const comparisonRuns = computed(() => appState.training?.comparisonRuns || []);
// 核心5：comparisonRuns 保存 Baseline、PPO、DPO、ORPO、LinUCB、DDQN 等对照实验。
const baselineSeries = computed(() => comparisonRuns.value[0]?.series || appState.training?.evaluationSeries || []);
const bestRun = computed(() => appState.training?.bestRun || {});

const colors = {
  Baseline: "#8a94a6",
  PPO: "#2563eb",
  DPO: "#16a34a",
  LINUCB: "#0891b2",
  "DUELING DDQN": "#7c3aed",
  ORPO: "#d97706",
};

const methodCards = computed(() =>
  comparisonRuns.value.map((run) => {
    const metrics = run.metrics || {};
    const improvement = run.improvement || {};
    return {
      name: run.name,
      algorithm: run.algorithm || "RL",
      reward: Number(metrics.trained_average_reward || 0),
      sourceHit: Number(metrics.trained_average_source_hit || 0),
      topicHit: Number(metrics.trained_average_topic_hit || 0),
      rewardGain: Number(improvement.rewardGain || 0),
      sourceGain: Number(improvement.sourceHitGain || 0),
      topicGain: Number(improvement.topicHitGain || 0),
      outputPath: run.outputPath || "",
    };
  }),
);

function signed(value) {
  // 把提升量格式化成带正负号的百分比/小数展示。
  const number = Number(value || 0);
  return `${number >= 0 ? "+" : ""}${number.toFixed(4)}`;
}

function gainClass(value) {
  return Number(value || 0) >= 0 ? "positive" : "negative";
}

function seriesForRun(run, key) {
  // 从某个算法的逐题评测结果中取出 Source Hit 或 Topic Hit 序列。
  return (run.series || []).map((item) => Number(item[key] || 0));
}

// 核心5：构造 ECharts 折线图配置。每个问题一格横轴，对比 baseline 与各 RL 策略。
function makeOption(title, baselineKey, trainedKey) {
  const labels = baselineSeries.value.map((item) => item.label);
  const legend = ["Baseline", ...comparisonRuns.value.map((run) => run.algorithm || run.name)];
  const chartSeries = [
    {
      name: "Baseline",
      type: "line",
      smooth: true,
      showSymbol: false,
      lineStyle: { color: colors.Baseline, width: 2 },
      data: baselineSeries.value.map((item) => Number(item[baselineKey] || 0)),
    },
    ...comparisonRuns.value.map((run) => {
      const algorithm = run.algorithm || run.name;
      const isBest = algorithm === bestRun.value.algorithm;
      return {
        name: algorithm,
        type: "line",
        smooth: true,
        showSymbol: isBest,
        symbolSize: isBest ? 6 : 4,
        lineStyle: {
          color: colors[algorithm] || "#475569",
          width: isBest ? 4 : 3,
          type: algorithm === "ORPO" ? "dashed" : "solid",
        },
        itemStyle: { color: colors[algorithm] || "#475569" },
        data: seriesForRun(run, trainedKey),
        z: isBest ? 4 : 2,
      };
    }),
  ];

  return {
    color: legend.map((item) => colors[item] || "#475569"),
    tooltip: {
      trigger: "axis",
      formatter(items) {
        const row = baselineSeries.value[items[0]?.dataIndex || 0] || {};
        const lines = [`${row.label || ""} ${row.query || ""}`.trim()];
        for (const item of items) {
          lines.push(`${item.marker}${item.seriesName}: ${Number(item.value || 0).toFixed(4)}`);
        }
        return lines.join("<br/>");
      },
    },
    legend: {
      data: legend,
      top: 2,
      type: "scroll",
    },
    grid: {
      left: 48,
      right: 24,
      top: 54,
      bottom: 48,
    },
    xAxis: {
      type: "category",
      data: labels,
      axisLabel: {
        interval: 0,
        rotate: labels.length > 18 ? 45 : 0,
      },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
    },
    series: chartSeries,
    graphic: [
      {
        type: "text",
        right: 18,
        top: 22,
        style: {
          text: title,
          fill: "#111827",
          fontSize: 14,
          fontWeight: 700,
        },
      },
    ],
  };
}

async function renderCharts() {
  await nextTick();
  if (!baselineSeries.value.length) return;

  sourceHitChart?.dispose();
  topicHitChart?.dispose();

  if (sourceHitRef.value) {
    // 核心5：Source Hit 图，表达“是否找到了预期论文/文件来源”。
    sourceHitChart = echarts.init(sourceHitRef.value);
    sourceHitChart.setOption(makeOption("Source Hit", "baselineSourceHit", "trainedSourceHit"));
  }

  if (topicHitRef.value) {
    // 核心5：Topic Hit 图，表达“是否覆盖了预期主题”。
    topicHitChart = echarts.init(topicHitRef.value);
    topicHitChart.setOption(makeOption("Topic Hit", "baselineTopicHit", "trainedTopicHit"));
  }
}

function resizeCharts() {
  sourceHitChart?.resize();
  topicHitChart?.resize();
}

watch(() => appState.training, renderCharts, { deep: true });
onMounted(() => {
  renderCharts();
  window.addEventListener("resize", resizeCharts);
});
onBeforeUnmount(() => {
  window.removeEventListener("resize", resizeCharts);
  sourceHitChart?.dispose();
  topicHitChart?.dispose();
});
</script>

<template>
  <section class="page-fill page-single training-page">
    <article class="card full-card">
      <header class="card-header">
        <div>
          <div class="eyebrow">RL Policy Comparison</div>
          <h2 class="card-title">Baseline vs PPO vs DPO vs ORPO</h2>
          <div class="card-subtitle">Source Hit and Topic Hit across the same evaluation questions.</div>
        </div>
        <el-button type="primary" plain @click="refreshTraining">Refresh</el-button>
      </header>

      <div v-if="appState.training?.available" class="card-body page-scroll">
        <div class="training-summary">
          <div><strong>Best run</strong>{{ bestRun.algorithm || "-" }} / {{ bestRun.name || "-" }}</div>
          <div><strong>Best reward gain</strong>{{ signed(bestRun.rewardGain) }}</div>
          <div><strong>Active output</strong>{{ appState.training.run?.outputPath || "-" }}</div>
        </div>

        <div class="method-grid">
          <div
            v-for="item in methodCards"
            :key="item.name"
            class="method-card"
            :class="{ winner: item.algorithm === bestRun.algorithm }"
          >
            <div class="method-head">
              <span>{{ item.algorithm }}</span>
              <small>{{ item.name }}</small>
            </div>
            <div class="metric-row">
              <span>Reward</span>
              <strong>{{ item.reward.toFixed(4) }}</strong>
              <em :class="gainClass(item.rewardGain)">{{ signed(item.rewardGain) }}</em>
            </div>
            <div class="metric-row">
              <span>Source Hit</span>
              <strong>{{ item.sourceHit.toFixed(4) }}</strong>
              <em :class="gainClass(item.sourceGain)">{{ signed(item.sourceGain) }}</em>
            </div>
            <div class="metric-row">
              <span>Topic Hit</span>
              <strong>{{ item.topicHit.toFixed(4) }}</strong>
              <em :class="gainClass(item.topicGain)">{{ signed(item.topicGain) }}</em>
            </div>
          </div>
        </div>

        <div class="chart-grid">
          <section class="chart-panel">
            <h3>Source Hit</h3>
            <div ref="sourceHitRef" class="echart-box training-focus-chart" />
          </section>
          <section class="chart-panel">
            <h3>Topic Hit</h3>
            <div ref="topicHitRef" class="echart-box training-focus-chart" />
          </section>
        </div>
      </div>

      <div v-else class="card-body">
        <div class="training-empty">No retrieval policy evaluation has been generated yet.</div>
      </div>
    </article>
  </section>
</template>

<style scoped>
.training-summary {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}

.training-summary div,
.method-card,
.chart-panel {
  border: 1px solid #d9dee8;
  background: #ffffff;
  border-radius: 8px;
}

.training-summary div {
  padding: 12px 14px;
  color: #263241;
  overflow-wrap: anywhere;
}

.training-summary strong {
  display: block;
  margin-bottom: 4px;
  color: #667085;
  font-size: 12px;
  text-transform: uppercase;
}

.method-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(220px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}

.method-card {
  padding: 14px;
}

.method-card.winner {
  border-color: #16a34a;
  box-shadow: inset 0 0 0 1px rgba(22, 163, 74, 0.22);
}

.method-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}

.method-head span {
  font-weight: 800;
  color: #111827;
}

.method-head small {
  color: #667085;
  text-align: right;
  overflow-wrap: anywhere;
}

.metric-row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  align-items: center;
  gap: 10px;
  padding: 7px 0;
  border-top: 1px solid #eef1f5;
  color: #465466;
}

.metric-row strong {
  color: #111827;
}

.metric-row em {
  min-width: 74px;
  font-style: normal;
  font-weight: 700;
  text-align: right;
}

.positive {
  color: #15803d;
}

.negative {
  color: #b45309;
}

.chart-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}

.chart-panel {
  padding: 14px 14px 8px;
  min-width: 0;
}

.chart-panel h3 {
  margin: 0 0 8px;
  font-size: 16px;
  color: #111827;
}

.training-focus-chart {
  width: 100%;
  height: 360px;
}

.training-empty {
  padding: 36px;
  text-align: center;
  color: #667085;
}

@media (max-width: 980px) {
  .training-summary,
  .method-grid,
  .chart-grid {
    grid-template-columns: 1fr;
  }
}
</style>
