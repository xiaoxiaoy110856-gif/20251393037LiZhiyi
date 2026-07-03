import { appState, refreshHealth } from "../store.js";

const { computed } = Vue;

export default {
  name: "StatusPage",
  setup() {
    const rows = computed(() => {
      const health = appState.health || {};
      return [
        { label: "LLM 后端", value: health.llmBackend || "-" },
        { label: "模型", value: health.llmModel || "-" },
        { label: "LLM 状态", value: health.llmReady ? "可用" : "不可用", level: health.llmReady ? "ok" : "warn" },
        { label: "Agent", value: health.agentEnabled ? "已启用" : "未启用" },
        { label: "RAG", value: health.ragEnabled ? "已启用" : "未启用" },
        { label: "Embedding 模型", value: health.embeddingModel || "-" },
        { label: "知识库文档数", value: String(health.knowledgeDocuments ?? 0) },
        { label: "数据库后端", value: health.dbBackend || "-" },
        { label: "数据库状态", value: health.dbReady ? "已连接" : "未连接", level: health.dbReady ? "ok" : "warn" },
        { label: "数据库详情", value: health.dbDetail || "-" },
      ];
    });

    return {
      refreshHealth,
      rows,
    };
  },
  template: `
    <section class="page-fill page-single">
      <article class="card full-card">
        <header class="card-header">
          <div>
            <div class="eyebrow">System Status</div>
            <h2 class="card-title">系统状态</h2>
            <div class="card-subtitle">统一查看模型、数据库、Agent 和 RAG 的当前运行状态。</div>
          </div>
          <el-button type="primary" plain @click="refreshHealth">刷新状态</el-button>
        </header>
        <div class="card-body page-scroll">
          <div class="health-list">
            <div v-for="row in rows" :key="row.label" class="health-row">
              <span>{{ row.label }}</span>
              <strong :class="row.level || ''">{{ row.value }}</strong>
            </div>
          </div>
        </div>
      </article>
    </section>
  `,
};
