import { reactive } from "vue";
import { api } from "@/api";

export const defaultContextPreview = "No retrieval context yet. Ask a question or attach a file to build context.";
const sandboxSessionKey = "trl:sandbox:session";
const sandboxApprovalsKey = "trl:sandbox:approvals";

function browserStorage() {
  // 浏览器存储封装：用于保存沙盒授权等轻量状态。
  if (typeof window === "undefined" || !window.sessionStorage) return null;
  return window.sessionStorage;
}

function createSandboxSessionId() {
  // 沙盒会话 ID：同一浏览器窗口复用，配合“同窗口同命令只确认一次”的授权逻辑。
  const storage = browserStorage();
  const existing = storage?.getItem(sandboxSessionKey);
  if (existing) return existing;
  const generated = `window_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  storage?.setItem(sandboxSessionKey, generated);
  return generated;
}

function loadSandboxApprovals() {
  // 从浏览器本地存储读取已经确认过的沙盒授权 scope。
  const storage = browserStorage();
  if (!storage) return {};
  try {
    return JSON.parse(storage.getItem(sandboxApprovalsKey) || "{}");
  } catch {
    return {};
  }
}

function saveSandboxApprovals(approvals) {
  // 把沙盒授权结果写入本地存储，让同窗口同命令不重复确认。
  browserStorage()?.setItem(sandboxApprovalsKey, JSON.stringify(approvals));
}

export const appState = reactive({
  sessions: [],
  sessionId: null,
  health: null,
  knowledge: null,
  training: null,
  policyEvaluation: null,
  repos: [],
  sandboxSessionId: createSandboxSessionId(),
  sandboxApprovals: loadSandboxApprovals(),
  topK: 5,
  selectedModelId: "",
  queryInput: "",
  sending: false,
  chatStatus: "Ready",
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
  fileReadForm: {
    path: "PROJECT_OVERVIEW.md",
    startLine: 1,
    endLine: 80,
    maxBytes: 20000,
  },
  fileReadResult: null,
  fileReadLoading: false,
  fileReadStatus: "",
  fileEditForm: {
    path: "",
    instruction: "",
    maxChars: 24000,
  },
  fileEditProposal: null,
  fileEditLoading: false,
  fileEditApplying: false,
  fileEditStatus: "",
  imagePrompt: "",
  composerImagePrompt: "",
  imageToolOpen: false,
  imageGenerating: false,
  generatedImage: null,
  imageStatus: "",
});

async function ensureSandboxApproval(scope, description) {
  // 沙盒授权检查：同一个 scope 已确认过就直接放行，否则弹窗询问用户。
  if (appState.sandboxApprovals[scope]) return true;
  const message = [
    "需要本窗口授权后才能执行这个本地操作：",
    "",
    description,
    "",
    "确认后，同一个浏览器窗口里同一种命令不再重复询问。",
  ].join("\n");
  const approved = typeof window === "undefined" ? true : window.confirm(message);
  if (!approved) {
    throw new Error("Sandbox approval denied by user.");
  }
  appState.sandboxApprovals[scope] = new Date().toISOString();
  saveSandboxApprovals(appState.sandboxApprovals);
  return true;
}

export function resetSandboxApprovals() {
  // 清空当前窗口的沙盒授权记录，下一次敏感操作会重新确认。
  appState.sandboxApprovals = {};
  saveSandboxApprovals(appState.sandboxApprovals);
}

function normalizeSession(session) {
  // 统一会话对象结构，保证 messages/sources/contextPreview 等字段存在。
  return {
    id: session.id,
    title: session.title || "New chat",
    summary: session.summary || "",
    messages: session.messages || [],
    created_at: session.created_at || "",
    updated_at: session.updated_at || "",
  };
}

function upsertSession(session) {
  // 新会话插入列表，已有会话则原地更新，避免侧边栏重复。
  const normalized = normalizeSession(session);
  const index = appState.sessions.findIndex((item) => item.id === normalized.id);
  if (index >= 0) {
    appState.sessions.splice(index, 1, { ...appState.sessions[index], ...normalized });
  } else {
    appState.sessions.unshift(normalized);
  }
}

export async function refreshHealth() {
  // 刷新后端健康状态，包括模型、数据库、RAG、Agent、ComfyUI。
  appState.health = await api.health();
  const options = appState.health?.modelOptions || [];
  if (!appState.selectedModelId && options.length) {
    appState.selectedModelId = options[0].id;
  }
}

export async function refreshKnowledge() {
  // 刷新知识库概览，展示文档数、主题和样例标题。
  appState.knowledge = await api.knowledge();
}

export async function rebuildKnowledge() {
  // 触发后端重建知识库索引。
  await api.rebuildKnowledge();
  await refreshKnowledge();
}

export async function refreshTraining() {
  // 核心5：刷新强化学习训练结果，供 TrainingView 折线图使用。
  appState.training = await api.retrievalTraining();
}

export async function refreshPolicyEvaluation() {
  // 核心5：刷新策略评估数据，供 PolicyView/训练对比图使用。
  appState.policyEvaluation = await api.policyEvaluation();
}

export async function refreshRepos() {
  // 刷新本地仓库列表，用于项目文件/仓库面板。
  const data = await api.repos();
  appState.repos = data.items || [];
}

export async function refreshSessions() {
  // 刷新所有聊天会话，并自动选择第一个会话。
  const data = await api.sessions();
  const existing = new Map(appState.sessions.map((item) => [item.id, item]));
  appState.sessions = (data.sessions || []).map((session) => ({
    ...(existing.get(session.id) || {}),
    ...normalizeSession(session),
    messages: existing.get(session.id)?.messages || session.messages || [],
  }));
  if (!appState.sessionId && appState.sessions.length) {
    appState.sessionId = appState.sessions[0].id;
  }
}

export async function loadSession(sessionId) {
  // 加载某个会话详情，包括完整消息历史。
  if (!sessionId) return null;
  const data = await api.session(sessionId);
  upsertSession(data.session);
  appState.sessionId = sessionId;
  return data.session;
}

export async function createSession(title = "New chat") {
  // 创建新聊天会话，并把当前会话切换到新 ID。
  const data = await api.createSession(title);
  upsertSession(data.session);
  appState.sessionId = data.session.id;
  return data.session;
}

export async function ensureActiveSession() {
  // 确保当前有可用会话；没有会话时自动创建一个。
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
  // 切换当前会话，并加载该会话消息。
  appState.sessionId = sessionId;
  await loadSession(sessionId);
}

// 核心4：读取聊天输入框文字，先在前端追加用户消息，再把 payload 发送到 /api/chat。
export async function sendCurrentMessage() {
  const query = appState.queryInput.trim();
  if (!query || appState.sending) return;

  await ensureActiveSession();
  const attachment = appState.composerAttachment;
  const asksForAttachedFile = /这个文件|该文件|上传的文件|附件|文档|分析文件/i.test(query);
  if (asksForAttachedFile && !attachment) {
    appState.chatStatus = "Please attach a file first.";
    return;
  }
  const visibleQuery = attachment
    ? `${query}\n\n[Attached file: ${attachment.name}, ${Math.ceil(attachment.size / 1024)} KB, ${attachment.text.length} chars included]`
    : query;
  const current = appState.sessions.find((item) => item.id === appState.sessionId);
  if (current) {
    current.messages = [
      ...(current.messages || []),
      {
        role: "user",
        content: visibleQuery,
        attachment: attachment
          ? {
              name: attachment.name,
              size: attachment.size,
              chars: attachment.text.length,
            }
          : null,
      },
    ];
  }

  // 核心4：这个 payload 是“前端输入框状态 -> 后端 Agent/Qwen 聊天接口”的桥。
  const payload = {
    query,
    session_id: appState.sessionId,
    top_k: appState.topK,
    model_id: appState.selectedModelId,
    attachment_name: attachment?.name || "",
    attachment_text: attachment?.text || "",
  };

  appState.queryInput = "";
  appState.composerAttachment = null;
  appState.sending = true;
  appState.chatStatus = "Thinking...";

  try {
    const data = await api.chat(payload);
    upsertSession({
      ...data.session,
      messages: data.history || [],
    });
    appState.sessionId = data.session.id;
    appState.sources = data.sources || [];
    appState.contextPreview = data.contextPreview || defaultContextPreview;
    appState.chatStatus = "Ready";
  } catch (error) {
    appState.chatStatus = `Chat failed: ${error.message}`;
    throw error;
  } finally {
    appState.sending = false;
  }
}

export async function cloneRepo() {
  // 克隆 Git 仓库到本地工作区，执行前需要沙盒授权。
  if (!appState.cloneForm.repoUrl.trim() || appState.cloneLoading) return;
  await ensureSandboxApproval("clone_repo", "clone_repo：允许把 GitHub 仓库克隆到本地 repos 目录。");
  appState.cloneLoading = true;
  appState.cloneStatus = "Cloning repository...";
  try {
    const data = await api.cloneRepo({
      repo_url: appState.cloneForm.repoUrl.trim(),
      branch: appState.cloneForm.branch.trim(),
      target_name: appState.cloneForm.targetName.trim(),
      sandbox_session_id: appState.sandboxSessionId,
    });
    appState.repos = data.repos || [];
    appState.cloneStatus = `Clone complete: ${data.target || data.path || "trl/repos"}`;
    appState.cloneForm.repoUrl = "";
    appState.cloneForm.branch = "";
    appState.cloneForm.targetName = "";
  } catch (error) {
    appState.cloneStatus = `Clone failed: ${error.message}`;
    throw error;
  } finally {
    appState.cloneLoading = false;
  }
}

export async function readLocalFile() {
  // 核心4：读取本地文件工具的前端触发函数，会先经过沙盒授权。
  const path = appState.fileReadForm.path.trim();
  if (!path || appState.fileReadLoading) return;
  await ensureSandboxApproval("read_file", "read_file：允许读取当前 workspace 内的普通文本文件。敏感文件仍会被后端拦截。");
  appState.fileReadLoading = true;
  appState.fileReadStatus = "Reading local file through workspace sandbox...";
  try {
    const data = await api.readLocalFile({
      path,
      start_line: Number(appState.fileReadForm.startLine) || 1,
      end_line: Number(appState.fileReadForm.endLine) || null,
      max_bytes: Number(appState.fileReadForm.maxBytes) || 20000,
      sandbox_session_id: appState.sandboxSessionId,
    });
    appState.fileReadResult = data;
    appState.fileReadStatus = `Read complete: ${data.path || path}`;
  } catch (error) {
    appState.fileReadStatus = `Read failed: ${error.message}`;
    throw error;
  } finally {
    appState.fileReadLoading = false;
  }
}

export async function proposeFileEdit() {
  // 核心4：请求后端生成文件修改方案，只展示 diff，不立即写入文件。
  const path = appState.fileEditForm.path.trim();
  const instruction = appState.fileEditForm.instruction.trim();
  if (!path || !instruction || appState.fileEditLoading) return;
  await ensureSandboxApproval("propose_file_edit", "propose_file_edit：允许读取目标文件并让模型生成修改提案，不会直接写入。");
  appState.fileEditLoading = true;
  appState.fileEditProposal = null;
  appState.fileEditStatus = "Generating a safe edit proposal...";
  try {
    const data = await api.proposeFileEdit({
      path,
      instruction,
      max_chars: Number(appState.fileEditForm.maxChars) || 24000,
      model_id: appState.selectedModelId,
      sandbox_session_id: appState.sandboxSessionId,
    });
    appState.fileEditProposal = data;
    appState.fileEditStatus = data.changed ? "Proposal ready. Review the diff before applying." : "Proposal ready, but no changes were detected.";
  } catch (error) {
    appState.fileEditStatus = `Proposal failed: ${error.message}`;
    throw error;
  } finally {
    appState.fileEditLoading = false;
  }
}

export async function applyFileEdit() {
  // 核心4：应用文件修改的前端入口，会先经过沙盒授权再写入。
  const proposal = appState.fileEditProposal;
  if (!proposal || !proposal.changed || appState.fileEditApplying) return;
  await ensureSandboxApproval("apply_file_edit", "apply_file_edit：允许把已审核的修改写入本地文件；写入前会自动备份原文件。");
  appState.fileEditApplying = true;
  appState.fileEditStatus = "Applying edit and writing backup...";
  try {
    const data = await api.applyFileEdit({
      path: proposal.path,
      new_content: proposal.newContent,
      sha256_before: proposal.sha256Before,
      instruction: proposal.instruction,
      sandbox_session_id: appState.sandboxSessionId,
    });
    appState.fileEditStatus = `Applied. Backup: ${data.backupPath}`;
    appState.fileEditProposal = { ...proposal, applied: true, applyResult: data };
  } catch (error) {
    appState.fileEditStatus = `Apply failed: ${error.message}`;
    throw error;
  } finally {
    appState.fileEditApplying = false;
  }
}

export async function attachLocalFile(file) {
  // 核心4：把用户上传/选择的文件读成文本，后续随聊天消息一起发给后端。
  if (!file) return;
  const text = await file.text();
  appState.composerAttachment = {
    name: file.name,
    size: file.size,
    text: text.slice(0, 20000),
  };
}

export function clearAttachment() {
  // 清除当前聊天输入框绑定的附件。
  appState.composerAttachment = null;
}

export async function generateImageFromPrompt() {
  // 核心3：图片侧边栏生成入口，直接调用 /api/images/generate。
  const prompt = appState.imagePrompt.trim();
  if (!prompt || appState.imageGenerating) return;
  appState.imageGenerating = true;
  appState.imageStatus = "Generating image...";
  try {
    const data = await api.generateImage({
      prompt,
      model: appState.selectedModelId,
      size: "1024x1024",
      quality_mode: "high",
      batch_size: 1,
      allow_retry: true,
      use_highres_fix: true,
    });
    appState.generatedImage = data;
    appState.imageStatus = data.detail || (data.images?.length ? "Image generated." : "Image request completed.");
  } catch (error) {
    appState.imageStatus = `Image generation failed: ${error.message}`;
    throw error;
  } finally {
    appState.imageGenerating = false;
  }
}

export function openImageTool() {
  // 打开图片工具面板，并把当前聊天输入同步为图片 prompt 初稿。
  appState.imageToolOpen = !appState.imageToolOpen;
  if (appState.imageToolOpen && !appState.composerImagePrompt.trim()) {
    appState.composerImagePrompt = appState.queryInput.trim();
  }
}

export async function sendImageGenerationFromComposer() {
  // 核心3：从聊天输入框触发图片生成，走 /api/chat，让 Agent 调用高级图片工具。
  const prompt = (appState.composerImagePrompt || appState.queryInput).trim();
  if (!prompt || appState.sending || appState.imageGenerating) return;

  await ensureActiveSession();
  const query = `请根据以下描述生成图片：${prompt}`;
  const current = appState.sessions.find((item) => item.id === appState.sessionId);
  if (current) {
    current.messages = [...(current.messages || []), { role: "user", content: query }];
  }

  appState.queryInput = "";
  appState.composerImagePrompt = "";
  appState.imageToolOpen = false;
  appState.sending = true;
  appState.imageGenerating = true;
  appState.chatStatus = "Generating image...";
  appState.imageStatus = "Generating image in chat...";

  try {
    const data = await api.chat({
      query,
      session_id: appState.sessionId,
      top_k: appState.topK,
      model_id: appState.selectedModelId,
    });
    upsertSession({
      ...data.session,
      messages: data.history || [],
    });
    appState.sessionId = data.session.id;
    appState.sources = data.sources || [];
    appState.contextPreview = data.contextPreview || defaultContextPreview;
    appState.chatStatus = "Ready";

    const imageTrace = (data.toolTraces || []).find((trace) => trace.name === "generate_image_advanced" || trace.name === "generate_image");
    if (imageTrace?.output) {
      appState.generatedImage = imageTrace.output;
      appState.imageStatus = imageTrace.output.detail || (imageTrace.output.images?.length ? "Image generated." : "Image request completed.");
    } else {
      appState.imageStatus = "Image request completed in chat.";
    }
  } catch (error) {
    appState.chatStatus = `Image generation failed: ${error.message}`;
    appState.imageStatus = `Image generation failed: ${error.message}`;
    throw error;
  } finally {
    appState.sending = false;
    appState.imageGenerating = false;
  }
}

export async function bootstrapStore() {
  // 前端启动时的初始化流程：加载健康状态、知识库、训练结果、会话等。
  await Promise.all([refreshHealth(), refreshKnowledge(), refreshTraining(), refreshPolicyEvaluation(), refreshRepos(), refreshSessions()]);
  if (appState.sessionId) {
    await loadSession(appState.sessionId);
  }
}
