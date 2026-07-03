export async function fetchJson(url, options = {}) {
  // 通用 API 请求函数：统一处理 JSON 解析和错误抛出。
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

export const api = {
  health: () => fetchJson("/api/health"),
  knowledge: () => fetchJson("/api/knowledge"),
  rebuildKnowledge: () => fetchJson("/api/knowledge/rebuild", { method: "POST" }),
  sessions: () => fetchJson("/api/sessions"),
  session: (id) => fetchJson(`/api/sessions/${encodeURIComponent(id)}`),
  createSession: (title) =>
    fetchJson("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  // 核心4：聊天输入框提交路径。appStore 读取 queryInput 后通过这里请求后端 /api/chat。
  chat: (payload) =>
    fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  // 核心3：图片侧边栏直接生成接口；聊天里的图片请求通常走 Agent 工具链。
  generateImage: (payload) =>
    fetchJson("/api/images/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  retrievalTraining: () => fetchJson("/api/retrieval-training/latest"),
  policyEvaluation: () => fetchJson("/api/policy-evaluation"),
  repos: () => fetchJson("/api/repos"),
  cloneRepo: (payload) =>
    fetchJson("/api/repos/clone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  readLocalFile: (payload) =>
    fetchJson("/api/files/read", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  proposeFileEdit: (payload) =>
    fetchJson("/api/files/propose-edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  applyFileEdit: (payload) =>
    fetchJson("/api/files/apply-edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  trajectories: (limit = 20) => fetchJson(`/api/trajectories?limit=${encodeURIComponent(limit)}`),
  saveTrajectory: (payload) =>
    fetchJson("/api/trajectories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};
