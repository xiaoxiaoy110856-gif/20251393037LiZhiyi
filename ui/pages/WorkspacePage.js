import { appState, attachLocalFile, clearAttachment, sendCurrentMessage } from "../store.js";
import { formatRichText, typesetMath } from "../format.js";

const { computed, nextTick, onMounted, ref, watch } = Vue;

export default {
  name: "WorkspacePage",
  setup() {
    const pageRoot = ref(null);
    const messageBox = ref(null);
    const fileInput = ref(null);

    const currentSession = computed(() => appState.sessions.find((item) => item.id === appState.sessionId) || null);
    const messages = computed(() => currentSession.value?.messages || []);
    const latestTraining = computed(() => appState.training?.run?.metrics || {});

    const scrollToBottom = async () => {
      await nextTick();
      if (messageBox.value) {
        messageBox.value.scrollTop = messageBox.value.scrollHeight;
      }
      await typesetMath(pageRoot.value);
    };

    const triggerAttachmentPicker = () => {
      fileInput.value?.click();
    };

    const onFileChange = async (event) => {
      const file = event.target.files?.[0];
      if (file) {
        await attachLocalFile(file);
      }
      event.target.value = "";
    };

    const onSend = async () => {
      await sendCurrentMessage();
      await scrollToBottom();
    };

    watch(messages, scrollToBottom, { deep: true });
    watch(
      () => [appState.sources.length, appState.contextPreview],
      async () => {
        await nextTick();
        await typesetMath(pageRoot.value);
      },
      { deep: true }
    );

    onMounted(scrollToBottom);

    return {
      appState,
      clearAttachment,
      currentSession,
      fileInput,
      formatRichText,
      latestTraining,
      messageBox,
      messages,
      onFileChange,
      onSend,
      pageRoot,
      triggerAttachmentPicker,
    };
  },
  template: `
    <section ref="pageRoot" class="page-fill">
      <div class="content-grid">
        <div class="content-main">
          <article class="card full-card">
            <header class="card-header">
              <div>
                <div class="eyebrow">Dialogue Workspace</div>
                <h2 class="card-title">对话工作台</h2>
                <div class="card-subtitle">在这里直接和 Agent 对话、附加本地文本文件，并观察检索证据和上下文。</div>
              </div>
              <div class="topbar-chip">{{ appState.chatStatus }}</div>
            </header>
            <div ref="messageBox" class="messages">
              <div class="messages-inner" v-if="messages.length">
                <div v-for="(message, index) in messages" :key="index" class="message" :class="message.role === 'user' ? 'user' : 'assistant'">
                  <div class="message-card">
                    <div class="message-role">{{ message.role === 'user' ? '用户' : '助手' }}</div>
                    <div class="message-bubble" v-html="formatRichText(message.content)"></div>
                  </div>
                </div>
              </div>
              <div v-else class="messages-inner">
                <div class="empty-state">这里会显示当前会话的完整消息。你可以先问一个强化学习或轨迹相关的问题，我们就从这个页面一路追到证据、训练和策略展示。</div>
              </div>
            </div>
          </article>

          <article class="card composer-card">
            <el-input
              v-model="appState.queryInput"
              type="textarea"
              :rows="4"
              resize="none"
              placeholder="例如：请帮我对比 PPO 与 DQN 在轨迹路径规划中的决策差异。"
              @keydown.ctrl.enter.prevent="onSend"
            />
            <div v-if="appState.composerAttachment" class="attachment-chip-row">
              <div class="attachment-chip">
                <span class="attachment-name">{{ appState.composerAttachment.name }}</span>
                <span class="attachment-meta">已附加 {{ Math.ceil(appState.composerAttachment.size / 1024) }} KB，本次会作为上下文交给 Agent</span>
              </div>
              <el-button text type="danger" @click="clearAttachment">移除</el-button>
            </div>
            <div class="composer-actions">
              <div style="display:flex; gap:10px; align-items:center;">
                <el-button circle @click="triggerAttachmentPicker">+</el-button>
                <input ref="fileInput" type="file" hidden @change="onFileChange" />
                <el-select v-model="appState.topK" style="width: 110px;">
                  <el-option :value="3" label="Top 3" />
                  <el-option :value="5" label="Top 5" />
                  <el-option :value="8" label="Top 8" />
                </el-select>
              </div>
              <el-button type="primary" :loading="appState.sending" @click="onSend">发送</el-button>
            </div>
            <div class="composer-hint">支持 Markdown 表格、LaTeX 公式和文本类附件。快捷键：Ctrl + Enter</div>
          </article>
        </div>

        <aside class="content-side">
          <article class="card full-card">
            <header class="card-header">
              <div>
                <div class="eyebrow">Retrieved Evidence</div>
                <h3 class="card-title side-title">证据与来源</h3>
              </div>
            </header>
            <div class="card-body side-scroll">
              <div v-if="appState.sources.length" class="source-list">
                <div v-for="(source, index) in appState.sources" :key="index" class="source-card">
                  <h4>{{ source.title || source.name || ('来源 ' + (index + 1)) }}</h4>
                  <div class="source-meta">topic: {{ source.topic || '-' }} | score: {{ Number(source.score || 0).toFixed(4) }}</div>
                  <div class="source-path">{{ source.path || source.source_path || '-' }}</div>
                  <div class="source-snippet">{{ source.snippet || source.preview || '当前来源没有摘要。' }}</div>
                </div>
              </div>
              <div v-else class="empty-state">当前回答还没有返回检索来源。等你发出问题后，这里会展示知识库命中的证据。</div>
            </div>
          </article>

          <article class="card full-card">
            <header class="card-header">
              <div>
                <div class="eyebrow">Context</div>
                <h3 class="card-title side-title">上下文预览</h3>
              </div>
            </header>
            <div class="card-body side-scroll">
              <div class="context-box">{{ appState.contextPreview }}</div>
            </div>
          </article>

          <article class="card">
            <div class="card-body">
              <div class="info-grid">
                <div class="info-stat">
                  <div class="label">当前会话</div>
                  <div class="value compact">{{ currentSession?.title || '未命名会话' }}</div>
                </div>
                <div class="info-stat">
                  <div class="label">最近训练 Reward</div>
                  <div class="value">{{ Number(latestTraining.trained_average_reward || 0).toFixed(4) }}</div>
                </div>
              </div>
            </div>
          </article>
        </aside>
      </div>
    </section>
  `,
};
