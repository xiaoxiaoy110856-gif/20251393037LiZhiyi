<script setup>
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { Picture } from "@element-plus/icons-vue";
import {
  appState,
  attachLocalFile,
  clearAttachment,
  generateImageFromPrompt,
  openImageTool,
  sendCurrentMessage,
  sendImageGenerationFromComposer,
} from "@/stores/appStore";
import { formatRichText, typesetMath } from "@/utils/format";

const pageRoot = ref(null);
const messageBox = ref(null);
const fileInput = ref(null);

const currentSession = computed(() => appState.sessions.find((item) => item.id === appState.sessionId) || null);
const messages = computed(() => currentSession.value?.messages || []);
const modelOptions = computed(() => appState.health?.modelOptions || []);
const comfyui = computed(() => appState.health?.comfyui || {});

async function scrollToBottom() {
  // 对话刷新后自动滚动到底部，保证最新回答可见。
  await nextTick();
  if (messageBox.value) {
    messageBox.value.scrollTop = messageBox.value.scrollHeight;
  }
  await typesetMath(pageRoot.value);
}

function triggerAttachmentPicker() {
  // 触发隐藏的文件选择框，用于把本地文件附加到当前对话。
  fileInput.value?.click();
}

async function onFileChange(event) {
  // 用户选择文件后，将文件内容读入 appStore.composerAttachment。
  const file = event.target.files?.[0];
  if (file) await attachLocalFile(file);
  event.target.value = "";
}

async function onSend() {
  // 核心4：聊天页面的发送按钮/快捷键入口，最终调用 sendCurrentMessage。
  await sendCurrentMessage();
  await scrollToBottom();
}

function messageAttachment(message) {
  // 从消息内容或 attachment 字段中解析附件信息，用于在气泡上显示文件卡片。
  if (message.attachment) return message.attachment;
  const match = (message.content || "").match(/\[Attached file: ([^,\]]+),\s*([^,\]]+),\s*(\d+) chars included\]/);
  if (!match) return null;
  return {
    name: match[1],
    sizeLabel: match[2],
    chars: Number(match[3] || 0),
  };
}

// 核心4：渲染气泡前移除轻量附件标记；真实文件正文已经走后端上下文，不需要直接显示。
function messageText(message) {
  return (message.content || "").replace(/\n\n\[Attached file: [^\]]+\]\s*$/m, "");
}

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
</script>

<template>
  <section ref="pageRoot" class="page-fill">
    <div class="content-grid">
      <div class="content-main workspace-main">
        <article class="card full-card workspace-chat-card">
          <header class="card-header">
            <div>
              <div class="eyebrow">Local Assistant</div>
              <h2 class="card-title">对话助手</h2>
              <div class="card-subtitle">连续对话、知识库证据和本地文件分析都从这里开始。</div>
            </div>
            <div class="topbar-chip">{{ appState.chatStatus }}</div>
          </header>

          <div ref="messageBox" class="messages workspace-messages">
            <div v-if="messages.length" class="messages-inner">
              <div
                v-for="(message, index) in messages"
                :key="index"
                class="message"
                :class="message.role === 'user' ? 'user' : 'assistant'"
              >
                <div class="message-card">
                  <div class="message-role">{{ message.role === "user" ? "你" : "助手" }}</div>
                  <div class="message-bubble">
                    <div v-if="messageAttachment(message)" class="message-attachment-card">
                      <div class="message-attachment-icon">FILE</div>
                      <div>
                        <div class="message-attachment-name">{{ messageAttachment(message).name }}</div>
                        <div class="message-attachment-meta">
                          {{ messageAttachment(message).sizeLabel || `${Math.ceil((messageAttachment(message).size || 0) / 1024)} KB` }}
                          · {{ messageAttachment(message).chars }} chars included
                        </div>
                      </div>
                    </div>
                    <div v-html="formatRichText(messageText(message))" />
                  </div>
                </div>
              </div>
            </div>
            <div v-else class="messages-inner">
              <div class="empty-state">
                先发一条消息吧。可以让它解释一个概念、总结本地知识库资料，或者继续追问上一轮回答。
              </div>
            </div>
          </div>
        </article>

        <article class="card composer-card workspace-composer-card">
          <!-- 核心4：主聊天输入框。sendCurrentMessage() 读取的源文本就是这里绑定的 appState.queryInput。 -->
          <el-input
            v-model="appState.queryInput"
            type="textarea"
            :rows="4"
            resize="none"
            placeholder="例如：请结合知识库解释 PPO 为什么能稳定策略更新。"
            @keydown.ctrl.enter.prevent="onSend"
          />
          <div v-if="appState.composerAttachment" class="attachment-chip-row">
            <div class="attachment-chip">
              <span class="attachment-name">{{ appState.composerAttachment.name }}</span>
              <span class="attachment-meta">
                已附加 {{ Math.ceil(appState.composerAttachment.size / 1024) }} KB 文本，发送时会一起分析。
              </span>
            </div>
            <el-button text type="danger" @click="clearAttachment">移除</el-button>
          </div>
          <div v-if="appState.imageToolOpen" class="composer-tool-panel">
            <div class="composer-tool-title">
              <span>Image Generation</span>
              <span class="composer-tool-status">{{ comfyui.ready ? "ComfyUI ready" : "ComfyUI not ready, fallback may be used" }}</span>
            </div>
            <el-input
              v-model="appState.composerImagePrompt"
              type="textarea"
              :rows="2"
              resize="none"
              placeholder="Describe the image to generate, for example: a warm bedroom photo with a wooden bed."
              @keydown.ctrl.enter.prevent="sendImageGenerationFromComposer"
            />
            <div class="composer-tool-actions">
              <span class="composer-hint">{{ appState.imageStatus || "The generated image will appear in this chat." }}</span>
              <el-button type="primary" plain :loading="appState.imageGenerating" @click="sendImageGenerationFromComposer">Generate image</el-button>
            </div>
          </div>
          <div class="composer-actions">
            <div style="display:flex; gap:10px; align-items:center;">
              <el-button circle @click="triggerAttachmentPicker">+</el-button>
              <el-tooltip content="Generate image" placement="top">
                <el-button circle :type="appState.imageToolOpen ? 'primary' : 'default'" @click="openImageTool">
                  <el-icon><Picture /></el-icon>
                </el-button>
              </el-tooltip>
              <input ref="fileInput" type="file" hidden @change="onFileChange" />
              <el-select v-model="appState.topK" style="width: 110px;">
                <el-option :value="3" label="Top 3" />
                <el-option :value="5" label="Top 5" />
                <el-option :value="8" label="Top 8" />
              </el-select>
              <el-select v-model="appState.selectedModelId" style="width: 190px;">
                <el-option
                  v-for="model in modelOptions"
                  :key="model.id"
                  :value="model.id"
                  :label="model.available === false ? `${model.label} (not configured)` : model.label"
                />
              </el-select>
            </div>
            <el-button type="primary" :loading="appState.sending" @click="onSend">发送</el-button>
          </div>
          <div class="composer-hint">支持 Markdown、LaTeX 和表格。快捷键：Ctrl + Enter</div>
        </article>
      </div>

      <aside class="content-side">
        <article class="card full-card">
          <header class="card-header">
            <div>
              <div class="eyebrow">Retrieved Evidence</div>
              <h3 class="card-title side-title">检索证据</h3>
            </div>
          </header>
          <div class="card-body side-scroll">
            <div v-if="appState.sources.length" class="source-list">
              <div v-for="(source, index) in appState.sources" :key="index" class="source-card">
                <h4>{{ source.title || source.name || `证据 ${index + 1}` }}</h4>
                <div class="source-meta">topic: {{ source.topic || "-" }} | score: {{ Number(source.score || 0).toFixed(4) }}</div>
                <div class="source-path">{{ source.path || source.source_path || "-" }}</div>
                <div class="source-snippet">{{ source.snippet || source.preview || "暂无片段预览" }}</div>
              </div>
            </div>
            <div v-else class="empty-state">这里会显示当前回答引用到的文档、论文笔记或本地知识片段。</div>
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
                <div class="value compact">{{ currentSession?.title || "未命名会话" }}</div>
              </div>
              <div class="info-stat">
                <div class="label">知识库文档</div>
                <div class="value">{{ appState.health?.knowledgeDocuments ?? 0 }}</div>
              </div>
            </div>
          </div>
        </article>

        <article class="card full-card">
          <header class="card-header">
            <div>
              <div class="eyebrow">Image Generation</div>
              <h3 class="card-title side-title">图片生成</h3>
              <div class="card-subtitle">
                {{ comfyui.ready ? "ComfyUI 已就绪" : "ComfyUI 未就绪，将生成占位图" }}
              </div>
            </div>
          </header>
          <div class="card-body side-scroll">
            <el-input
              v-model="appState.imagePrompt"
              type="textarea"
              :rows="4"
              resize="none"
              placeholder="描述你想生成的图片，例如：一张强化学习路径规划流程图。"
            />
            <div class="composer-actions image-actions">
              <span class="composer-hint">{{ appState.imageStatus || "未配置真实图片模型时，会先生成 SVG 图片卡片。" }}</span>
              <el-button type="primary" plain :loading="appState.imageGenerating" @click="generateImageFromPrompt">生成</el-button>
            </div>
            <div class="source-meta">
              ComfyUI: {{ comfyui.online ? "在线" : "离线" }} |
              checkpoint: {{ comfyui.checkpoints?.[0] || "未发现模型" }}
            </div>
            <div v-if="appState.generatedImage?.url || appState.generatedImage?.images?.[0]?.url" class="generated-image-wrap">
              <a :href="appState.generatedImage.url || appState.generatedImage.images[0].url" target="_blank" rel="noreferrer">
                <img :src="appState.generatedImage.url || appState.generatedImage.images[0].url" alt="Generated image" />
              </a>
              <div class="source-path">{{ appState.generatedImage.path || appState.generatedImage.images?.[0]?.url }}</div>
            </div>
          </div>
        </article>
      </aside>
    </div>
  </section>
</template>
