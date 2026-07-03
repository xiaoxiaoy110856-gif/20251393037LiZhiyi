import { appState, refreshTraining } from "../store.js";

export default {
  name: "TrainingPage",
  setup() {
    return { appState, refreshTraining };
  },
  template: `
    <section class="page-fill page-single">
      <article class="card full-card">
        <header class="card-header">
          <div>
            <div class="eyebrow">Training Center</div>
            <h2 class="card-title">训练中心</h2>
            <div class="card-subtitle">这里聚焦最近一次检索策略训练：训练指标、代表 episode 和基线对比。</div>
          </div>
          <el-button type="primary" plain @click="refreshTraining">刷新训练结果</el-button>
        </header>

        <div v-if="appState.training?.available" class="card-body page-scroll">
          <div class="training-summary">
            <div><strong>训练名称：</strong>{{ appState.training.run?.name || '-' }}</div>
            <div><strong>状态：</strong>{{ appState.training.run?.status || '-' }}</div>
            <div><strong>更新时间：</strong>{{ appState.training.run?.updatedAt || '-' }}</div>
            <div><strong>输出目录：</strong>{{ appState.training.run?.outputPath || '-' }}</div>
          </div>

          <div class="training-metric-grid">
            <div class="training-metric">
              <div class="label">训练后 Average Reward</div>
              <div class="value">{{ Number(appState.training.run?.metrics?.trained_average_reward || 0).toFixed(4) }}</div>
            </div>
            <div class="training-metric">
              <div class="label">Baseline Average Reward</div>
              <div class="value">{{ Number(appState.training.run?.metrics?.baseline_average_reward || 0).toFixed(4) }}</div>
            </div>
            <div class="training-metric">
              <div class="label">训练后 Source Hit</div>
              <div class="value">{{ Number(appState.training.run?.metrics?.trained_average_source_hit || 0).toFixed(4) }}</div>
            </div>
            <div class="training-metric">
              <div class="label">Baseline Source Hit</div>
              <div class="value">{{ Number(appState.training.run?.metrics?.baseline_average_source_hit || 0).toFixed(4) }}</div>
            </div>
          </div>

          <div v-if="appState.training.episode" class="episode-card">
            <h3>代表性 Episode</h3>
            <div class="episode-query">{{ appState.training.episode.query }}</div>
            <div class="episode-grid">
              <div class="episode-row"><span>DQN 选择动作</span><b>{{ appState.training.episode.chosen_action }}</b></div>
              <div class="episode-row"><span>DQN Reward</span><b>{{ Number(appState.training.episode.reward || 0).toFixed(4) }}</b></div>
              <div class="episode-row"><span>Baseline 动作</span><b>{{ appState.training.baselineEpisode?.chosen_action || '-' }}</b></div>
              <div class="episode-row"><span>Baseline Reward</span><b>{{ Number(appState.training.baselineEpisode?.reward || 0).toFixed(4) }}</b></div>
            </div>
          </div>
        </div>

        <div v-else class="card-body">
          <div class="training-empty">还没有训练记录。等我们继续跑 reward sweep 或 DQN 训练时，这里就会自动填起来。</div>
        </div>
      </article>
    </section>
  `,
};
