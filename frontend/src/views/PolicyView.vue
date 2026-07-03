<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";
import { appState, refreshPolicyEvaluation, refreshTraining } from "@/stores/appStore";

const compareRef = ref(null);
const rewardRef = ref(null);
const splitRef = ref(null);
const qaRef = ref(null);
const qwenOptimizationRef = ref(null);
let compareChart = null;
let rewardChart = null;
let splitChart = null;
let qaChart = null;
let qwenOptimizationChart = null;

const algorithmGroups = [
  {
    title: "离散动作路径规划：DQN 系列",
    summary: "适合把每个十字路口、转向口或候选道路当作一个离散动作来选。",
    rows: [
      ["DQN", "离散动作", "单智能体路径决策", "结构直观、易解释、适合演示按路口逐点规划", "动作太多时容易不稳，对连续控制不自然"],
      ["Double DQN", "离散动作", "比 DQN 更稳的路径选择", "减轻 Q 值高估，训练更稳", "仍然受离散动作空间限制"],
      ["Dueling DQN", "离散动作", "状态价值与动作优势拆分", "对路口价值评估更细", "实现稍复杂，解释成本更高"],
      ["Rainbow DQN", "离散动作", "需要更强基线时", "把多种增强策略合并，效果常更好", "工程复杂度高，不适合第一版演示"],
    ],
  },
  {
    title: "连续控制路径规划：PPO / DDPG / TD3 / SAC",
    summary: "适合速度控制、转向角控制、轨迹平滑和连续动作路径规划。",
    rows: [
      ["PPO", "连续 / 可离散化", "多步累计收益、阶段策略学习", "稳定、好调参、很适合做项目展示", "样本效率一般"],
      ["DDPG", "连续动作", "精细控制路径或速度曲线", "可直接输出连续动作", "训练容易不稳定、对超参数敏感"],
      ["TD3", "连续动作", "比 DDPG 更稳的连续控制", "缓解 Q 值过估计，连续控制更可靠", "依旧需要较细致的调参"],
      ["SAC", "连续动作", "需要探索能力更强的连续控制任务", "探索更积极、鲁棒性通常不错", "解释上不如 PPO 直观"],
    ],
  },
  {
    title: "多智能体路径规划：MADDPG / MAPPO / QMIX",
    summary: "适合多车协同、编队避障、多机器人路径规划等场景。",
    rows: [
      ["MADDPG", "多智能体连续动作", "多车/多机器人协同控制", "每个智能体可独立动作，又能联合训练", "稳定性和实现复杂度都较高"],
      ["MAPPO", "多智能体 PPO", "多智能体阶段策略优化", "继承 PPO 的稳定性，更适合项目展示", "训练成本更高"],
      ["QMIX", "多智能体离散动作", "多个智能体共同选离散动作", "适合网格/路口式联合规划", "对连续道路控制不自然"],
    ],
  },
  {
    title: "传统基础算法：Q-learning / SARSA",
    summary: "适合作为强化学习路径规划的入门基线与教学对照。",
    rows: [
      ["Q-learning", "离散动作", "最经典的 value-based 基线", "简单、概念清晰、适合教学", "状态空间一大就吃力"],
      ["SARSA", "离散动作", "在线策略学习的传统基线", "更保守，适合解释 on-policy 学习", "通常不如现代深度强化学习灵活"],
    ],
  },
];

const currentProjectHighlights = computed(() => {
  const metrics = appState.training?.run?.metrics || {};
  return [
    {
      label: "当前已落地",
      value: "DQN 检索策略学习",
    },
    {
      label: "最自然的下一步",
      value: "PPO 路径规划 / 工具调用阶段策略",
    },
    {
      label: "当前 reward 增益",
      value: `${Number(metrics.reward_gain_vs_baseline || 0).toFixed(4)}`,
    },
    {
      label: "当前 source hit 提升",
      value: `${Number(metrics.baseline_average_source_hit || 0).toFixed(4)} → ${Number(metrics.trained_average_source_hit || 0).toFixed(4)}`,
    },
  ];
});

const policyEvalHighlights = computed(() => {
  const evaluation = appState.policyEvaluation || {};
  const dataset = evaluation.dataset || {};
  const questionBank = evaluation.questionBank || {};
  const optimization = evaluation.optimization || {};
  return [
    {
      label: "问答数据集",
      value: `${dataset.questionCount || 0} 题`,
    },
    {
      label: "已跑问答结果",
      value: `${questionBank.completed || 0}/${questionBank.total || 0}`,
    },
    {
      label: "最佳策略试验",
      value: optimization.bestName || "-",
    },
    {
      label: "策略 reward 提升",
      value: `+${Number(optimization.rewardGain || 0).toFixed(4)}`,
    },
  ];
});

async function refreshPolicyPanel() {
  await Promise.all([refreshTraining(), refreshPolicyEvaluation()]);
}

async function renderCharts() {
  await nextTick();
  const evaluation = appState.policyEvaluation || {};
  const dataset = evaluation.dataset || {};
  const questionBank = evaluation.questionBank || {};
  const optimization = evaluation.optimization || {};

  if (splitRef.value && evaluation.available) {
    splitChart?.dispose();
    splitChart = echarts.init(splitRef.value);
    splitChart.setOption({
      tooltip: { trigger: "item" },
      legend: { bottom: 0 },
      series: [
        {
          name: "3:1:1 split",
          type: "pie",
          radius: ["48%", "72%"],
          center: ["50%", "45%"],
          data: [
            { name: "Train", value: Number(dataset.split?.train || 0) },
            { name: "Validation", value: Number(dataset.split?.validation || 0) },
            { name: "Test", value: Number(dataset.split?.test || 0) },
          ],
          color: ["#ff547a", "#66d9a6", "#6f8cff"],
        },
      ],
    });
  }

  if (qaRef.value && evaluation.available) {
    qaChart?.dispose();
    qaChart = echarts.init(qaRef.value);
    qaChart.setOption({
      tooltip: { trigger: "axis" },
      grid: { left: 48, right: 18, top: 36, bottom: 30 },
      xAxis: { type: "category", data: ["Completed", "Avg Sources", "Avg Tool Traces"] },
      yAxis: { type: "value" },
      series: [
        {
          name: "Question Bank",
          type: "bar",
          itemStyle: { color: "#6f8cff" },
          data: [
            Number(questionBank.completed || 0),
            Number(questionBank.averageSources || 0),
            Number(questionBank.averageToolTraces || 0),
          ],
        },
      ],
    });
  }

  if (qwenOptimizationRef.value && evaluation.available) {
    qwenOptimizationChart?.dispose();
    qwenOptimizationChart = echarts.init(qwenOptimizationRef.value);
    qwenOptimizationChart.setOption({
      tooltip: { trigger: "axis" },
      legend: { data: ["Baseline", "Optimized Policy"] },
      grid: { left: 48, right: 18, top: 40, bottom: 30 },
      xAxis: { type: "category", data: ["Reward", "Source Hit", "Topic Hit", "Point Recall"] },
      yAxis: { type: "value", max: 1 },
      series: [
        {
          name: "Baseline",
          type: "bar",
          itemStyle: { color: "#ff9bc6" },
          data: [
            Number(optimization.baseline?.reward || 0),
            Number(optimization.baseline?.sourceHit || 0),
            Number(optimization.baseline?.topicHit || 0),
            Number(optimization.baseline?.pointRecall || 0),
          ],
        },
        {
          name: "Optimized Policy",
          type: "bar",
          itemStyle: { color: "#ff547a" },
          data: [
            Number(optimization.trained?.reward || 0),
            Number(optimization.trained?.sourceHit || 0),
            Number(optimization.trained?.topicHit || 0),
            Number(optimization.trained?.pointRecall || 0),
          ],
        },
      ],
    });
  }

  if (!appState.training?.available) return;
  const metrics = appState.training.run?.metrics || {};
  const curve = appState.training.curve || [];

  if (compareRef.value) {
    compareChart?.dispose();
    compareChart = echarts.init(compareRef.value);
    compareChart.setOption({
      tooltip: { trigger: "axis" },
      legend: { data: ["Baseline", "DQN"] },
      grid: { left: 48, right: 16, top: 40, bottom: 30 },
      xAxis: {
        type: "category",
        data: ["Average Reward", "Source Hit", "Topic Hit", "Point Recall"],
      },
      yAxis: { type: "value", max: 1 },
      series: [
        {
          name: "Baseline",
          type: "bar",
          itemStyle: { color: "#ff9bc6" },
          data: [
            Number(metrics.baseline_average_reward || 0),
            Number(metrics.baseline_average_source_hit || 0),
            Number(metrics.baseline_average_topic_hit || 0),
            Number(metrics.baseline_average_point_recall || 0),
          ],
        },
        {
          name: "DQN",
          type: "bar",
          itemStyle: { color: "#ff547a" },
          data: [
            Number(metrics.trained_average_reward || 0),
            Number(metrics.trained_average_source_hit || 0),
            Number(metrics.trained_average_topic_hit || 0),
            Number(metrics.trained_average_point_recall || 0),
          ],
        },
      ],
    });
  }

  if (rewardRef.value) {
    rewardChart?.dispose();
    rewardChart = echarts.init(rewardRef.value);
    rewardChart.setOption({
      tooltip: { trigger: "axis" },
      grid: { left: 48, right: 16, top: 30, bottom: 30 },
      xAxis: {
        type: "category",
        data: curve.map((_, index) => index + 1),
      },
      yAxis: { type: "value" },
      series: [
        {
          name: "Episode Reward",
          type: "line",
          smooth: true,
          showSymbol: false,
          lineStyle: { color: "#ff547a", width: 3 },
          areaStyle: { color: "rgba(255, 84, 122, 0.14)" },
          data: curve.map((item) => Number(item.reward || 0)),
        },
      ],
    });
  }
}

watch(() => appState.training, renderCharts, { deep: true });
watch(() => appState.policyEvaluation, renderCharts, { deep: true });
onMounted(renderCharts);
onBeforeUnmount(() => {
  compareChart?.dispose();
  rewardChart?.dispose();
  splitChart?.dispose();
  qaChart?.dispose();
  qwenOptimizationChart?.dispose();
});
</script>

<template>
  <section class="page-fill page-single">
    <article class="card full-card">
      <header class="card-header">
        <div>
          <div class="eyebrow">Policy Impact</div>
          <h2 class="card-title">策略作用展示</h2>
          <div class="card-subtitle">这里把路径规划相关强化学习算法按动作空间和任务类型分层整理，并保留当前项目已经落地的 DQN 结果。</div>
        </div>
        <el-button type="primary" plain @click="refreshPolicyPanel">刷新评测结果</el-button>
      </header>

      <div class="card-body page-scroll">
        <div class="info-stat-grid">
          <div v-for="item in currentProjectHighlights" :key="item.label" class="info-stat">
            <div class="label">{{ item.label }}</div>
            <div class="value compact">{{ item.value }}</div>
          </div>
        </div>

        <div v-if="appState.policyEvaluation?.available" class="section-block">
          <h3 class="section-title">Qwen / 策略优化效果数据库证据</h3>
          <div class="status-note">
            这里读取 training_data 与 outputs 中已经跑完的问答结果、RAG 评测和 reward sweep，用 3:1:1 划分视角展示当前策略优化在本项目数据上的表现。
          </div>

          <div class="info-stat-grid">
            <div v-for="item in policyEvalHighlights" :key="item.label" class="info-stat">
              <div class="label">{{ item.label }}</div>
              <div class="value compact">{{ item.value }}</div>
            </div>
          </div>

          <div class="policy-explainer-grid">
            <div class="chart-card">
              <h3>问答数据集 3:1:1 划分</h3>
              <div ref="splitRef" class="echart-box" />
            </div>
            <div class="chart-card">
              <h3>问答结果覆盖情况</h3>
              <div ref="qaRef" class="echart-box" />
            </div>
          </div>

          <div class="chart-card">
            <h3>Baseline vs 优化策略</h3>
            <div ref="qwenOptimizationRef" class="echart-box" />
          </div>
        </div>

        <div class="policy-explainer-grid">
          <div class="policy-note">
            <h3>你这个项目里应该怎么理解它们</h3>
            <p>
              如果我们把“路口选择”看成离散动作，那么 DQN 系列最自然；如果把“速度、转向、整段路径平滑”看成连续控制，
              PPO / TD3 / SAC 更贴切；如果以后扩到多车协同或多机器人路径规划，多智能体算法才是真正的主角。
            </p>
          </div>
          <div class="policy-note">
            <h3>为什么现在先用 DQN</h3>
            <p>
              因为你当前系统最先落地的是“检索策略学习”和“按路口解释的路径规划演示”，它们都天然更像离散动作任务。
              这让 DQN 的解释性和工程复杂度都比较合适。
            </p>
          </div>
        </div>

        <div class="section-block" v-for="group in algorithmGroups" :key="group.title">
          <h3 class="section-title">{{ group.title }}</h3>
          <div class="status-note">{{ group.summary }}</div>
          <div class="table-wrap">
            <table class="algo-table">
              <thead>
                <tr>
                  <th>算法</th>
                  <th>动作空间</th>
                  <th>适合任务</th>
                  <th>优势</th>
                  <th>局限</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in group.rows" :key="row[0]">
                  <td>{{ row[0] }}</td>
                  <td>{{ row[1] }}</td>
                  <td>{{ row[2] }}</td>
                  <td>{{ row[3] }}</td>
                  <td>{{ row[4] }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div v-if="appState.training?.available" class="section-block">
          <h3 class="section-title">当前项目里的 DQN 实证结果</h3>
          <div class="chart-card">
            <h3>Baseline vs DQN</h3>
            <div ref="compareRef" class="echart-box" />
          </div>

          <div class="chart-card">
            <h3>DQN Reward 曲线</h3>
            <div ref="rewardRef" class="echart-box" />
          </div>
        </div>
      </div>
    </article>
  </section>
</template>
