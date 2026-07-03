<script setup>
import { computed, onMounted } from "vue";
import { useRoute, useRouter } from "vue-router";
import { appState, bootstrapStore, createSession, selectSession } from "@/stores/appStore";
import ParticleField from "@/components/ParticleField.vue";

const router = useRouter();
const route = useRoute();

const navItems = [
  { path: "/workspace", label: "对话助手" },
  { path: "/knowledge", label: "知识库" },
  { path: "/status", label: "系统状态" },
  { path: "/training", label: "训练中心" },
  { path: "/policy", label: "策略展示" },
  { path: "/repos", label: "仓库工具" },
  { path: "/pathdemo", label: "路径演示" },
];

const ragLabel = computed(() => (appState.health?.ragEnabled ? "RAG 已启用" : "RAG 未启用"));
const dbLabel = computed(() => (appState.health?.dbReady ? "DB 已连接" : "DB 未连接"));

async function onCreateSession() {
  await createSession();
  router.push("/workspace");
}

async function onSelectSession(sessionId) {
  await selectSession(sessionId);
  if (route.path !== "/workspace") {
    router.push("/workspace");
  }
}

onMounted(async () => {
  await bootstrapStore();
});
</script>

<template>
  <div class="workspace-shell">
    <ParticleField />
    <div class="workspace-glow" />

    <div class="workspace">
      <header class="topbar">
        <div class="topbar-brand">
          <div class="topbar-mark">A</div>
          <div class="topbar-copy">
            <div class="title">本地知识库 AI 助手</div>
            <div class="subtitle">对话规则受本地助手文档约束，地图、训练和策略功能保持完整</div>
          </div>
        </div>

        <nav class="topbar-nav">
          <ul class="nav-list">
            <li v-for="item in navItems" :key="item.path">
              <RouterLink :to="item.path" class="nav-link" :class="{ active: route.path === item.path }">
                {{ item.label }}
              </RouterLink>
            </li>
          </ul>
        </nav>

        <div class="topbar-meta">
          <span class="topbar-chip">{{ ragLabel }}</span>
          <span class="topbar-chip">{{ dbLabel }}</span>
        </div>
      </header>

      <div class="workspace-body">
        <aside class="sidebar">
          <section class="sidebar-card">
            <h3 class="sidebar-title">对话原则</h3>
            <p class="sidebar-copy">
              对话回答遵循本地助手规则：优先承接上下文，能引用知识库证据，证据不足时明确说明。其他功能模块保持原有入口。
            </p>
          </section>

          <section class="sidebar-card session-card">
            <div class="session-toolbar">
              <div>
                <h3 class="sidebar-title">会话导航</h3>
                <div class="session-meta">{{ appState.sessions.length }} 个会话</div>
              </div>
              <el-button size="small" type="primary" plain @click="onCreateSession">新会话</el-button>
            </div>

            <div class="session-menu">
              <div
                v-for="session in appState.sessions"
                :key="session.id"
                class="session-item"
                :class="{ active: appState.sessionId === session.id }"
                @click="onSelectSession(session.id)"
              >
                <div class="session-entry">
                  <strong>{{ session.title || "新会话" }}</strong>
                  <span>{{ session.summary || "等待第一条消息..." }}</span>
                </div>
              </div>
            </div>
          </section>
        </aside>

        <main class="content-shell">
          <RouterView />
        </main>
      </div>
    </div>
  </div>
</template>
