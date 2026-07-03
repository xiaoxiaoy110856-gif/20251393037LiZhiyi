<script setup>
import { computed } from "vue";
import { appState, refreshHealth } from "@/stores/appStore";

const rows = computed(() => {
  const health = appState.health || {};
  return [
    { label: "LLM 后端", value: health.llmBackend || "-" },
    { label: "模型名称", value: health.llmModel || "-" },
    { label: "LLM 状态", value: health.llmReady ? "可用" : "不可用", level: health.llmReady ? "ok" : "warn" },
    { label: "Agent", value: health.agentEnabled ? "已启用" : "未启用" },
    { label: "RAG", value: health.ragEnabled ? "已启用" : "未启用" },
    { label: "Embedding 模型", value: health.embeddingModel || "-" },
    { label: "知识库文档数", value: String(health.knowledgeDocuments ?? 0) },
    { label: "UI 模式", value: health.uiMode || "assistant" },
    { label: "高级页面", value: health.advancedUiEnabled ? "已显示" : "已隐藏" },
    { label: "数据库后端", value: health.dbBackend || "-" },
    { label: "数据库状态", value: health.dbReady ? "已连接" : "未连接", level: health.dbReady ? "ok" : "warn" },
    { label: "数据库详情", value: health.dbDetail || "-" },
  ];
});

const contractRows = computed(() => {
  const contract = appState.health?.assistantContract || {};
  return [
    { label: "本地模型优先", value: contract.localModel ? "满足" : "待检查" },
    { label: "连续会话记忆", value: contract.conversationMemory ? "满足" : "待检查" },
    { label: "知识库检索", value: contract.knowledgeRetrieval ? "满足" : "未启用" },
    { label: "聊天界面优先", value: contract.minimalChatFirst ? "满足" : "待调整" },
    { label: "训练/策略可选", value: contract.advancedTrainingOptional ? "满足" : "待调整" },
  ];
});
</script>

<template>
  <section class="page-fill page-single">
    <article class="card full-card">
      <header class="card-header">
        <div>
          <div class="eyebrow">System Status</div>
          <h2 class="card-title">系统状态</h2>
          <div class="card-subtitle">检查本地模型、RAG、会话记忆、数据库和聊天优先原则是否稳定。</div>
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

        <div class="section-block">
          <h3 class="section-title">本地助手规则对齐</h3>
          <div class="health-list">
            <div v-for="row in contractRows" :key="row.label" class="health-row">
              <span>{{ row.label }}</span>
              <strong>{{ row.value }}</strong>
            </div>
          </div>
        </div>
      </div>
    </article>
  </section>
</template>
