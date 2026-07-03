import { appState, rebuildKnowledge } from "../store.js";

const { computed } = Vue;

export default {
  name: "KnowledgePage",
  setup() {
    const topics = computed(() =>
      Object.entries(appState.knowledge?.topics || {})
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
    );

    return {
      appState,
      rebuildKnowledge,
      topics,
    };
  },
  template: `
    <section class="page-fill page-single">
      <article class="card full-card">
        <header class="card-header">
          <div>
            <div class="eyebrow">Knowledge Base</div>
            <h2 class="card-title">知识库</h2>
            <div class="card-subtitle">查看当前轨迹与强化学习知识库的规模、主题分布和研究焦点。</div>
          </div>
          <el-button type="primary" plain @click="rebuildKnowledge">重建知识库</el-button>
        </header>
        <div class="card-body page-scroll">
          <div class="info-stat-grid" v-if="appState.knowledge">
            <div class="info-stat">
              <div class="label">知识库标题</div>
              <div class="value compact">{{ appState.knowledge.title || '-' }}</div>
            </div>
            <div class="info-stat">
              <div class="label">文档数</div>
              <div class="value">{{ appState.knowledge.documentCount || 0 }}</div>
            </div>
            <div class="info-stat">
              <div class="label">Chunk 数</div>
              <div class="value">{{ appState.knowledge.chunkCount || 0 }}</div>
            </div>
            <div class="info-stat">
              <div class="label">研究焦点</div>
              <div class="value compact">{{ appState.knowledge.researchFocus || '-' }}</div>
            </div>
          </div>

          <div class="section-block">
            <h3 class="section-title">主题分布</h3>
            <div class="knowledge-topic-list">
              <div v-for="topic in topics" :key="topic.name" class="topic-card">
                <div class="name">{{ topic.name }}</div>
                <div class="count">{{ topic.count }}</div>
              </div>
            </div>
          </div>
        </div>
      </article>
    </section>
  `,
};
