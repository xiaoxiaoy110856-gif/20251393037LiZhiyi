import { appState, bootstrapStore, createSession, selectSession } from "./store.js";
import WorkspacePage from "./pages/WorkspacePage.js";
import KnowledgePage from "./pages/KnowledgePage.js";
import StatusPage from "./pages/StatusPage.js";
import TrainingPage from "./pages/TrainingPage.js";
import PolicyPage from "./pages/PolicyPage.js";
import RepoToolsPage from "./pages/RepoToolsPage.js";
import PathDemoPage from "./pages/PathDemoPage.js";

const { computed, createApp, onMounted } = Vue;
const { createRouter, createWebHashHistory } = VueRouter;

const routes = [
  { path: "/", redirect: "/workspace" },
  { path: "/workspace", name: "workspace", component: WorkspacePage, meta: { label: "对话工作台" } },
  { path: "/knowledge", name: "knowledge", component: KnowledgePage, meta: { label: "知识库" } },
  { path: "/status", name: "status", component: StatusPage, meta: { label: "系统状态" } },
  { path: "/training", name: "training", component: TrainingPage, meta: { label: "训练中心" } },
  { path: "/policy", name: "policy", component: PolicyPage, meta: { label: "策略作用展示" } },
  { path: "/repos", name: "repos", component: RepoToolsPage, meta: { label: "仓库工具" } },
  { path: "/pathdemo", name: "pathdemo", component: PathDemoPage, meta: { label: "路径规划演示" } },
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

const AppShell = {
  name: "AppShell",
  setup() {
    const navItems = routes.filter((item) => item.meta?.label);
    const currentPath = computed(() => router.currentRoute.value.path);
    const ragLabel = computed(() => (appState.health?.ragEnabled ? "RAG 已启用" : "RAG 未启用"));
    const dbLabel = computed(() => (appState.health?.dbReady ? "DB 已连接" : "DB 未连接"));

    const onCreateSession = async () => {
      await createSession();
      router.push("/workspace");
    };

    const onSelectSession = async (sessionId) => {
      await selectSession(sessionId);
      router.push("/workspace");
    };

    onMounted(async () => {
      await bootstrapStore();
    });

    return {
      appState,
      currentPath,
      dbLabel,
      navItems,
      onCreateSession,
      onSelectSession,
      ragLabel,
    };
  },
  template: `
    <div class="workspace">
      <header class="topbar">
        <div class="topbar-brand">
          <div class="topbar-mark">A</div>
          <div class="topbar-copy">
            <div class="title">轨迹强化学习研究 Agent</div>
            <div class="subtitle">面向轨迹、强化学习与知识检索的本地研究工作台</div>
          </div>
        </div>

        <nav class="topbar-nav">
          <ul class="nav-list">
            <li v-for="item in navItems" :key="item.path">
              <router-link :to="item.path" class="nav-link" :class="{ active: currentPath === item.path }">{{ item.meta.label }}</router-link>
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
            <h3 class="sidebar-title">研究工作流</h3>
            <p class="sidebar-copy">先把 Agent 本体做完整，再用检索策略学习和路径规划演示把强化学习如何优化 Agent 讲清楚。</p>
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
                  <strong>{{ session.title || '新会话' }}</strong>
                  <span>{{ session.summary || '等待新的消息…' }}</span>
                </div>
              </div>
            </div>
          </section>
        </aside>

        <main class="content-shell">
          <router-view />
        </main>
      </div>
    </div>
  `,
};

createApp(AppShell).use(router).use(ElementPlus).mount("#app");
