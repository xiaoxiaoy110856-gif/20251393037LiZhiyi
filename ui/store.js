import { api } from "./api.js";

const { reactive } = Vue;

export const defaultContextPreview =
  "这里会显示当前回答使用到的知识库上下文、附件摘要和工具轨迹，方便我们一起看 Agent 到底依据了什么。";

export const appState = reactive({
  sessions: [],
  sessionId: null,
  health: null,
  knowledge: null,
  training: null,
  repos: [],
  topK: 5,
  queryInput: "",
  sending: false,
  chatStatus: "就绪",
  contextPreview: defaultContextPreview,
  sources: [],
  composerAttachment: null,
  cloneForm: {
    repoUrl: "",
    branch: "",
    targetName: "",
  },
  cloneStatus: "",
  cloneLoading: false,
});

function upsertSession(session) {
  const normalized = {
    id: session.id,
    title: session.title || "新会话",
    summary: session.summary || "",
    messages: session.messages || [],
    created_at: session.created_at || "",
    updated_at: session.updated_at || "",
  };
  const index = appState.sessions.findIndex((item) => item.id === normalized.id);
  if (index >= 0) {
    appState.sessions.splice(index, 1, { ...appState.sessions[index], ...normalized });
  } else {
    appState.sessions.unshift(normalized);
  }
}

export async function refreshHealth() {
  appState.health = await api.health();
}

export async function refreshKnowledge() {
  appState.knowledge = await api.knowledge();
}

export async function rebuildKnowledge() {
  await api.rebuildKnowledge();
  await refreshKnowledge();
}

export async function refreshTraining() {
  appState.training = await api.retrievalTraining();
}

export async function refreshRepos() {
  const data = await api.repos();
  appState.repos = data.items || [];
}

export async function refreshSessions() {
  const data = await api.sessions();
  const existing = new Map(appState.sessions.map((item) => [item.id, item]));
  appState.sessions = (data.sessions || []).map((session) => ({
    ...(existing.get(session.id) || {}),
    ...session,
    messages: existing.get(session.id)?.messages || session.messages || [],
  }));
  if (!appState.sessionId && appState.sessions.length) {
    appState.sessionId = appState.sessions[0].id;
  }
}

export async function loadSession(sessionId) {
  if (!sessionId) return null;
  const data = await api.session(sessionId);
  upsertSession(data.session);
  appState.sessionId = sessionId;
  return data.session;
}

export async function createSession(title = "新会话") {
  const data = await api.createSession(title);
  upsertSession(data.session);
  appState.sessionId = data.session.id;
  return data.session;
}

export async function ensureActiveSession() {
  if (!appState.sessionId) {
    await createSession();
  }
  const current = appState.sessions.find((item) => item.id === appState.sessionId);
  if (!current?.messages?.length) {
    await loadSession(appState.sessionId);
  }
  return appState.sessionId;
}

export async function selectSession(sessionId) {
  appState.sessionId = sessionId;
  await loadSession(sessionId);
}

export async function sendCurrentMessage() {
  const query = appState.queryInput.trim();
  if (!query || appState.sending) return;

  await ensureActiveSession();
  const current = appState.sessions.find((item) => item.id === appState.sessionId);
  if (current) {
    current.messages = [...(current.messages || []), { role: "user", content: query }];
  }

  const attachment = appState.composerAttachment;
  const payload = {
    query,
    session_id: appState.sessionId,
    top_k: appState.topK,
    attachment_name: attachment?.name || "",
    attachment_text: attachment?.text || "",
  };

  appState.queryInput = "";
  appState.composerAttachment = null;
  appState.sending = true;
  appState.chatStatus = "正在生成回答...";

  try {
    const data = await api.chat(payload);
    upsertSession({
      ...data.session,
      messages: data.history || [],
    });
    appState.sessionId = data.session.id;
    appState.sources = data.sources || [];
    appState.contextPreview = data.contextPreview || defaultContextPreview;
    appState.chatStatus = "就绪";
  } catch (error) {
    appState.chatStatus = `发送失败：${error.message}`;
    throw error;
  } finally {
    appState.sending = false;
  }
}

export async function cloneRepo() {
  if (!appState.cloneForm.repoUrl.trim() || appState.cloneLoading) return;
  appState.cloneLoading = true;
  appState.cloneStatus = "正在克隆仓库...";
  try {
    const data = await api.cloneRepo({
      repo_url: appState.cloneForm.repoUrl.trim(),
      branch: appState.cloneForm.branch.trim(),
      target_name: appState.cloneForm.targetName.trim(),
    });
    appState.repos = data.repos || [];
    appState.cloneStatus = `已克隆到：${data.targetPath || data.path || "trl/repos"}`;
    appState.cloneForm.repoUrl = "";
    appState.cloneForm.branch = "";
    appState.cloneForm.targetName = "";
  } catch (error) {
    appState.cloneStatus = `克隆失败：${error.message}`;
    throw error;
  } finally {
    appState.cloneLoading = false;
  }
}

export async function attachLocalFile(file) {
  if (!file) return;
  const text = await file.text();
  appState.composerAttachment = {
    name: file.name,
    size: file.size,
    text: text.slice(0, 20000),
  };
}

export function clearAttachment() {
  appState.composerAttachment = null;
}

export async function bootstrapStore() {
  await Promise.all([refreshHealth(), refreshKnowledge(), refreshTraining(), refreshRepos(), refreshSessions()]);
  if (appState.sessionId) {
    await loadSession(appState.sessionId);
  }
}
