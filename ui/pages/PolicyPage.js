import { appState, refreshTraining } from "../store.js";

const { nextTick, onBeforeUnmount, onMounted, ref, watch } = Vue;

export default {
  name: "PolicyPage",
  setup() {
    const compareRef = ref(null);
    const rewardRef = ref(null);
    let compareChart = null;
    let rewardChart = null;

    const renderCharts = async () => {
      await nextTick();
      if (!appState.training?.available || !window.echarts) return;
      const metrics = appState.training.run?.metrics || {};
      const curve = appState.training.curve || [];

      if (compareRef.value) {
        compareChart?.dispose();
        compareChart = window.echarts.init(compareRef.value);
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
              itemStyle: { color: "#5c7cff" },
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
        rewardChart = window.echarts.init(rewardRef.value);
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
              lineStyle: { color: "#45c38a", width: 3 },
              areaStyle: { color: "rgba(69, 195, 138, 0.14)" },
              data: curve.map((item) => Number(item.reward || 0)),
            },
          ],
        });
      }
    };

    watch(() => appState.training, renderCharts, { deep: true });
    onMounted(renderCharts);
    onBeforeUnmount(() => {
      compareChart?.dispose();
      rewardChart?.dispose();
    });

    return {
      appState,
      compareRef,
      refreshTraining,
      rewardRef,
    };
  },
  template: `
    <section class="page-fill page-single">
      <article class="card full-card">
        <header class="card-header">
          <div>
            <div class="eyebrow">Policy Impact</div>
            <h2 class="card-title">策略作用展示</h2>
            <div class="card-subtitle">这一页专门解释：为什么我们先用 DQN 做检索策略学习，以及 PPO 接下来会接到哪里。</div>
          </div>
          <el-button type="primary" plain @click="refreshTraining">刷新策略结果</el-button>
        </header>
        <div v-if="appState.training?.available" class="card-body page-scroll">
          <div class="info-stat-grid">
            <div class="info-stat">
              <div class="label">当前已经落地</div>
              <div class="value compact">DQN 检索策略学习</div>
            </div>
            <div class="info-stat">
              <div class="label">下一步适合扩展</div>
              <div class="value compact">PPO 多步工具决策 / 路径规划</div>
            </div>
          </div>

          <div class="policy-explainer-grid">
            <div class="policy-note">
              <h3>为什么先做检索策略学习</h3>
              <p>因为它把 Agent 的状态、动作和奖励讲得最清楚：问题是什么、该怎么改写查询、偏向哪些主题、最后是否真的找到了更对的资料。</p>
            </div>
            <div class="policy-note">
              <h3>DQN 当前在优化什么</h3>
              <p>当前 DQN 不是在直接改写大模型输出，而是在优化 Agent 的检索决策。我们已经看到训练后 Reward、Source Hit、Topic Hit 和 Point Recall 都比 baseline 更高。</p>
            </div>
          </div>

          <div class="chart-card">
            <h3>Baseline vs DQN</h3>
            <div ref="compareRef" class="echart-box"></div>
          </div>

          <div class="chart-card">
            <h3>DQN Reward 曲线</h3>
            <div ref="rewardRef" class="echart-box"></div>
          </div>
        </div>

        <div v-else class="card-body">
          <div class="training-empty">还没有策略训练结果。等训练跑起来后，这一页会自动长出对比图和曲线。</div>
        </div>
      </article>
    </section>
  `,
};
