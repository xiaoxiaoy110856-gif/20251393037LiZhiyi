import { createRouter, createWebHistory } from "vue-router";
import WorkspaceView from "@/views/WorkspaceView.vue";
import KnowledgeView from "@/views/KnowledgeView.vue";
import StatusView from "@/views/StatusView.vue";
import TrainingView from "@/views/TrainingView.vue";
import PolicyView from "@/views/PolicyView.vue";
import RepoToolsView from "@/views/RepoToolsView.vue";
import PathDemoView from "@/views/PathDemoView.vue";

const routes = [
  { path: "/", redirect: "/workspace" },
  { path: "/workspace", name: "workspace", component: WorkspaceView, meta: { label: "对话助手", core: true } },
  { path: "/knowledge", name: "knowledge", component: KnowledgeView, meta: { label: "知识库", core: true } },
  { path: "/status", name: "status", component: StatusView, meta: { label: "系统状态", core: true } },
  { path: "/training", name: "training", component: TrainingView, meta: { label: "训练中心", core: false } },
  { path: "/policy", name: "policy", component: PolicyView, meta: { label: "策略展示", core: false } },
  { path: "/repos", name: "repos", component: RepoToolsView, meta: { label: "仓库工具", core: false } },
  { path: "/pathdemo", name: "pathdemo", component: PathDemoView, meta: { label: "路径演示", core: false } },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
