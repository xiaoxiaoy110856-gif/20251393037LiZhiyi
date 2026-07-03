const { createApp, nextTick } = Vue;

const LONGQUANYI_QUERY = "龙泉驿区, 成都市, 四川省, 中国";
const NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search";

const TEXT = {
  appTitle: "轨迹强化学习研究 Agent",
  appSubtitle: "面向轨迹、强化学习与知识检索的本地研究工作台",
  workspaceTitle: "对话工作台",
  workspaceSubtitle: "在这里直接对话、挂本地文本附件，并查看证据、上下文和策略痕迹。",
  knowledgeTitle: "知识库",
  knowledgeSubtitle: "查看知识库规模、主题分布和研究焦点，并在需要时重建索引。",
  statusTitle: "系统状态",
  statusSubtitle: "集中查看模型、Agent、RAG、数据库和 Embedding 设备的当前状态。",
  trainingTitle: "训练中心",
  trainingSubtitle: "查看最近一次检索策略训练、训练曲线和代表性 episode。",
  policyTitle: "策略作用展示",
  policySubtitle: "解释为什么先做检索策略学习，以及 DQN / PPO 在 Agent 上分别适合做什么。",
  reposTitle: "仓库工具",
  reposSubtitle: "克隆 GitHub 项目、查看本地仓库，并把它们接到研究工作流里。",
  pathTitle: "路径规划演示",
  pathSubtitle: "先挂上龙泉驿区真实地图底图和地标，后面你给我具体路线点，我们再把 PPO / DQN 动态走法接上去。",
  untitled: "新会话",
  emptySession: "先从一个问题开始吧。你可以问轨迹压缩、PPO 设计、实验路线，或者直接挂一个本地文本文件过来。",
  noSources: "这次没有拿到明确来源，通常意味着问题更偏模型常识，或者当前知识库证据不足。",
  defaultContext: "这里会展示本次回答用到的上下文、证据摘要和策略信息。",
  loadingStatus: "正在生成回答...",
  readyStatus: "系统就绪",
  errorStatus: "请求失败",
  rebuilding: "正在重建知识库索引...",
  rebuildDone: (count) => `知识库已重建完成，当前文档数 ${count}。`,
  noTraining: "还没有可展示的检索策略训练结果。先跑一次 reward sweep 或检索策略训练就会在这里出现。",
  noMap: "当前没有加载到龙泉驿区地图。",
};

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeRegex(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function applyInlineFormatting(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function splitTableRow(line) {
  let trimmed = line.trim();
  if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
  if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
  return trimmed.split("|").map((cell) => cell.trim());
}

function isTableSeparator(line) {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function renderTable(block) {
  const lines = block
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2 || !lines.every((line) => line.includes("|")) || !isTableSeparator(lines[1])) {
    return null;
  }

  const headers = splitTableRow(lines[0]);
  const rows = lines.slice(2).map(splitTableRow);
  const headHtml = headers.map((cell) => `<th>${applyInlineFormatting(cell)}</th>`).join("");
  const bodyHtml = rows
    .map((row) => `<tr>${row.map((cell) => `<td>${applyInlineFormatting(cell)}</td>`).join("")}</tr>`)
    .join("");

  return `<div class="table-wrap"><table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table></div>`;
}

function formatRichText(text) {
  const codeBlocks = [];
  let working = String(text || "");

  working = working.replace(/```([\s\S]*?)```/g, (_, code) => {
    const token = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(`<pre><code>${escapeHtml(code.trim())}</code></pre>`);
    return token;
  });

  const blocks = working
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  let html = blocks
    .map((block) => {
      const tableHtml = renderTable(block);
      if (tableHtml) return tableHtml;

      if (/^[-*]\s+/m.test(block)) {
        const items = block
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => line.replace(/^[-*]\s+/, ""));
        return `<ul>${items.map((item) => `<li>${applyInlineFormatting(item)}</li>`).join("")}</ul>`;
      }

      if (/^\d+\.\s+/m.test(block)) {
        const items = block
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => line.replace(/^\d+\.\s+/, ""));
        return `<ol>${items.map((item) => `<li>${applyInlineFormatting(item)}</li>`).join("")}</ol>`;
      }

      return `<p>${applyInlineFormatting(block).replace(/\n/g, "<br>")}</p>`;
    })
    .join("");

  codeBlocks.forEach((block, index) => {
    const pattern = new RegExp(escapeRegex(`__CODE_BLOCK_${index}__`), "g");
    html = html.replace(pattern, block);
  });

  return html || `<p>${escapeHtml(text || "")}</p>`;
}

async function typesetMath(root) {
  if (window.MathJax?.typesetPromise && root) {
    try {
      await window.MathJax.typesetPromise([root]);
    } catch (error) {
      console.warn("MathJax typeset failed:", error);
    }
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

createApp({
  data() {
    return {
      text: TEXT,
      activePage: "workspace",
      sessions: [],
      sessionId: null,
      sending: false,
      topK: 5,
      queryInput: "",
      chatStatus: TEXT.readyStatus,
      statusBusy: false,
      health: null,
      knowledge: null,
      training: null,
      repos: [],
      mapReady: false,
      mapStatus: "正在加载龙泉驿区地图...",
      mapSource: "",
      mapCenter: [30.56, 104.27],
      mapGeoJson: null,
      mapBoundingBox: null,
      mapLeaflet: null,
      baseTileLayer: null,
      districtBoundaryLayer: null,
      pointLayerGroup: null,
      mapKeyPoints: [
        { name: "东安湖片区", value: [104.2705, 30.6045], description: "适合展示赛事管制、临时封控和策略绕行。" },
        { name: "龙泉主城", value: [104.2756, 30.5565], description: "适合作为中间决策节点，体现策略对主干道的偏好。" },
        { name: "洛带古镇方向", value: [104.3495, 30.6352], description: "适合作为东侧目标点，后面你可以指定是否把它设成终点。" },
        { name: "经开区片区", value: [104.2458, 30.5668], description: "适合作为工业区关键点，后面可接入不同奖励设计。" },
      ],
      cloneForm: {
        repoUrl: "",
        branch: "",
        targetName: "",
      },
      cloneStatus: "",
      cloneLoading: false,
      plannerMode: "DQN",
      composerAttachment: null,
      contextPreview: TEXT.defaultContext,
      sources: [],
    };
  },
  computed: {
    currentSession() {
      return this.sessions.find((item) => item.id === this.sessionId) || null;
    },
    currentMessages() {
      return this.currentSession?.messages || [];
    },
    sessionCountLabel() {
      return `${this.sessions.length} 个会话`;
    },
    statusRows() {
      if (!this.health) return [];
      return [
        { label: "LLM 后端", value: this.health.llmBackend || "-" },
        { label: "模型名称", value: this.health.llmModel || "-" },
        { label: "LLM 状态", value: this.health.llmReady ? "已连接" : "未连接", level: this.health.llmReady ? "ok" : "warn" },
        { label: "Agent", value: this.health.agentEnabled ? "已开启" : "未开启" },
        { label: "RAG", value: this.health.ragEnabled ? "已开启" : "未开启" },
        { label: "Embedding", value: this.health.embeddingModel || "-" },
        { label: "研究焦点", value: this.health.researchFocus || "-" },
        { label: "知识库文档", value: String(this.health.knowledgeDocuments ?? 0) },
        { label: "数据库后端", value: this.health.dbBackend || "-" },
        { label: "数据库状态", value: this.health.dbReady ? "已连接" : "未连接", level: this.health.dbReady ? "ok" : "warn" },
        { label: "数据库详情", value: this.health.dbDetail || "-" },
      ];
    },
    knowledgeTopics() {
      if (!this.knowledge?.topics) return [];
      return Object.entries(this.knowledge.topics)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count);
    },
    trainingMetrics() {
      return this.training?.run?.metrics || {};
    },
    policyMetrics() {
      const metrics = this.trainingMetrics;
      return [
        { label: "Average Reward", trained: Number(metrics.trained_average_reward || 0), baseline: Number(metrics.baseline_average_reward || 0) },
        { label: "Source Hit", trained: Number(metrics.trained_average_source_hit || 0), baseline: Number(metrics.baseline_average_source_hit || 0) },
        { label: "Topic Hit", trained: Number(metrics.trained_average_topic_hit || 0), baseline: Number(metrics.baseline_average_topic_hit || 0) },
        { label: "Point Recall", trained: Number(metrics.trained_average_point_recall || 0), baseline: Number(metrics.baseline_average_point_recall || 0) },
      ];
    },
    representativeEpisode() {
      return this.training?.episode || null;
    },
    baselineEpisode() {
      return this.training?.baselineEpisode || null;
    },
    mapModeDescription() {
      return this.plannerMode === "PPO"
        ? "PPO 更适合后续接入连续代价、速度约束和多步策略更新。等你给定路线点后，我们可以把整条路径按策略阶段逐步播放。"
        : "DQN 更适合先把离散路口决策讲清楚。等你给定起终点和关键节点后，我们可以把每一步的选点过程动态展示出来。";
    },
  },
  watch: {
    activePage() {
      this.$nextTick(() => {
        this.renderPolicyCharts();
        this.renderPathMapChart();
        this.typesetCurrentView();
      });
    },
    training: {
      deep: true,
      handler() {
        this.$nextTick(() => this.renderPolicyCharts());
      },
    },
    currentMessages: {
      deep: true,
      handler() {
        this.$nextTick(() => {
          this.scrollMessagesToBottom();
          this.typesetCurrentView();
        });
      },
    },
    plannerMode() {
      this.$nextTick(() => this.renderPathMapChart());
    },
  },
  methods: {
    async refreshAll() {
      await Promise.all([
        this.refreshHealth(),
        this.refreshSessions(),
        this.refreshTraining(),
        this.refreshKnowledge(),
        this.refreshRepos(),
        this.refreshPathMap(),
      ]);
      await nextTick();
      await this.typesetCurrentView();
      this.renderPolicyCharts();
      this.renderPathMapChart();
    },
    async refreshHealth() {
      this.health = await fetchJson("/api/health");
    },
    async refreshKnowledge() {
      this.knowledge = await fetchJson("/api/knowledge");
    },
    async refreshTraining() {
      this.training = await fetchJson("/api/retrieval-training/latest");
    },
    async refreshSessions() {
      const data = await fetchJson("/api/sessions");
      this.sessions = data.sessions || [];
      if (!this.sessionId && this.sessions.length) this.sessionId = this.sessions[0].id;
    },
    async refreshRepos() {
      const data = await fetchJson("/api/repos");
      this.repos = data.items || [];
    },
    async refreshPathMap() {
      this.mapStatus = "正在加载龙泉驿区地图...";
      this.mapSource = "";
      try {
        const url = `${NOMINATIM_SEARCH_URL}?format=jsonv2&polygon_geojson=1&limit=1&q=${encodeURIComponent(LONGQUANYI_QUERY)}`;
        const response = await fetch(url, {
          headers: {
            Accept: "application/json",
          },
        });
        if (!response.ok) throw new Error(`Nominatim 请求失败: ${response.status}`);
        const results = await response.json();
        if (!Array.isArray(results) || !results.length) throw new Error("没有找到龙泉驿区边界");
        const match = results[0];
        const lat = Number(match.lat);
        const lon = Number(match.lon);
        if (Number.isFinite(lat) && Number.isFinite(lon)) {
          this.mapCenter = [lat, lon];
        }
        this.mapGeoJson = match.geojson || null;
        this.mapBoundingBox = match.boundingbox || null;
        this.mapReady = true;
        this.mapSource = "Nominatim + OpenStreetMap";
        this.mapStatus = "龙泉驿区真实地图已加载";
        await nextTick();
        this.renderLeafletMap(this.mapGeoJson, this.mapBoundingBox);
      } catch (error) {
        console.warn("Longquanyi map load failed:", error);
        this.mapReady = false;
        this.mapGeoJson = null;
        this.mapBoundingBox = null;
        this.mapStatus = `地图加载失败：${error.message}`;
      }
    },
    async createSession() {
      const data = await fetchJson("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: this.text.untitled }),
      });
      this.sessionId = data.session.id;
      this.contextPreview = this.text.defaultContext;
      this.sources = [];
      await this.refreshSessions();
    },
    selectSession(id) {
      this.sessionId = id;
    },
    async sendMessage() {
      const query = this.queryInput.trim();
      if (!query || this.sending) return;

      this.sending = true;
      this.chatStatus = this.text.loadingStatus;
      this.statusBusy = true;

      if (!this.sessionId) await this.createSession();

      const optimistic = this.currentMessages.slice();
      optimistic.push({ role: "user", content: query });
      const session = this.currentSession;
      if (session) session.messages = optimistic;
      this.queryInput = "";

      try {
        const data = await fetchJson("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            session_id: this.sessionId,
            top_k: this.topK,
            attachment_name: this.composerAttachment?.name || "",
            attachment_text: this.composerAttachment?.content || "",
          }),
        });

        this.contextPreview = data.contextPreview || this.text.defaultContext;
        this.sources = data.sources || [];

        const target = this.sessions.find((item) => item.id === data.session.id);
        if (target) {
          target.messages = data.history || [];
          target.title = data.session.title || target.title;
          target.summary = data.session.summary || target.summary;
          target.updated_at = data.session.updated_at || target.updated_at;
        }

        this.chatStatus = this.text.readyStatus;
        await this.refreshSessions();
        await nextTick();
        await this.typesetCurrentView();
        this.scrollMessagesToBottom();
        this.clearComposerAttachment();
      } catch (error) {
        this.chatStatus = this.text.errorStatus;
        this.contextPreview = error.message;
      } finally {
        this.sending = false;
        this.statusBusy = false;
      }
    },
    async rebuildKnowledge() {
      try {
        this.chatStatus = this.text.rebuilding;
        this.statusBusy = true;
        const data = await fetchJson("/api/knowledge/rebuild", { method: "POST" });
        this.contextPreview = this.text.rebuildDone(data.documentCount || data.documents || 0);
        await Promise.all([this.refreshKnowledge(), this.refreshHealth()]);
      } catch (error) {
        this.contextPreview = error.message;
      } finally {
        this.chatStatus = this.text.readyStatus;
        this.statusBusy = false;
      }
    },
    async cloneRepo() {
      if (!this.cloneForm.repoUrl.trim() || this.cloneLoading) return;
      this.cloneLoading = true;
      this.cloneStatus = "正在克隆仓库...";
      try {
        const data = await fetchJson("/api/repos/clone", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            repo_url: this.cloneForm.repoUrl,
            branch: this.cloneForm.branch,
            target_name: this.cloneForm.targetName,
          }),
        });
        this.cloneStatus = `仓库已克隆到 ${data.target}`;
        this.cloneForm.branch = "";
        this.cloneForm.targetName = "";
        await this.refreshRepos();
      } catch (error) {
        this.cloneStatus = error.message;
      } finally {
        this.cloneLoading = false;
      }
    },
    triggerAttachmentPicker() {
      this.$refs.filePicker?.click();
    },
    clearComposerAttachment() {
      this.composerAttachment = null;
      if (this.$refs.filePicker) this.$refs.filePicker.value = "";
    },
    async handleAttachmentChange(event) {
      const file = event.target.files?.[0];
      if (!file) return;
      const lower = file.name.toLowerCase();
      const textLike = [".txt", ".md", ".py", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".csv", ".log", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sh", ".ps1"];
      const matched = textLike.some((suffix) => lower.endsWith(suffix));
      if (!matched) {
        this.contextPreview = `当前附件入口只支持文本类文件，暂不直接处理 ${file.name}。`;
        this.clearComposerAttachment();
        return;
      }
      const content = await file.text();
      const maxChars = 12000;
      this.composerAttachment = {
        name: file.name,
        size: file.size,
        content: content.slice(0, maxChars),
        truncated: content.length > maxChars,
      };
    },
    async typesetCurrentView() {
      await nextTick();
      await typesetMath(document.getElementById("app"));
    },
    renderPolicyCharts() {
      if (this.activePage !== "policy" || !window.echarts || !this.training?.available) return;
      const compareDom = document.getElementById("policy-compare-chart");
      const rewardDom = document.getElementById("policy-reward-chart");
      if (!compareDom || !rewardDom) return;

      const compareChart = window.echarts.getInstanceByDom(compareDom) || window.echarts.init(compareDom);
      const rewardChart = window.echarts.getInstanceByDom(rewardDom) || window.echarts.init(rewardDom);

      compareChart.setOption({
        animation: false,
        tooltip: { trigger: "axis" },
        legend: { top: 0 },
        grid: { left: 40, right: 20, top: 50, bottom: 30 },
        xAxis: {
          type: "category",
          data: this.policyMetrics.map((item) => item.label),
          axisLabel: { color: "#596281" },
        },
        yAxis: {
          type: "value",
          axisLabel: { color: "#596281" },
          splitLine: { lineStyle: { color: "#edf1ff" } },
        },
        series: [
          {
            name: "Baseline",
            type: "bar",
            itemStyle: { color: "#ff9bc5", borderRadius: [8, 8, 0, 0] },
            data: this.policyMetrics.map((item) => Number(item.baseline.toFixed(4))),
          },
          {
            name: "DQN",
            type: "bar",
            itemStyle: { color: "#5b83ff", borderRadius: [8, 8, 0, 0] },
            data: this.policyMetrics.map((item) => Number(item.trained.toFixed(4))),
          },
        ],
      });

      const trace = this.training?.trace || [];
      rewardChart.setOption({
        animation: false,
        tooltip: { trigger: "axis" },
        grid: { left: 40, right: 20, top: 20, bottom: 30 },
        xAxis: {
          type: "category",
          data: trace.map((item, index) => `${index + 1}`),
          axisLabel: { color: "#596281" },
        },
        yAxis: {
          type: "value",
          axisLabel: { color: "#596281" },
          splitLine: { lineStyle: { color: "#edf1ff" } },
        },
        series: [
          {
            name: "Reward",
            type: "line",
            smooth: true,
            symbol: "circle",
            symbolSize: 6,
            lineStyle: { width: 3, color: "#6e7dff" },
            itemStyle: { color: "#ff81ba" },
            areaStyle: { color: "rgba(110, 125, 255, 0.12)" },
            data: trace.map((item) => Number((item.reward ?? item.average_reward ?? 0).toFixed(4))),
          },
        ],
      });
    },
    renderLeafletMap(geojson, boundingBox) {
      if (this.activePage !== "pathdemo" || !window.L) return;
      const mapDom = document.getElementById("longquanyi-map-chart");
      if (!mapDom) return;

      const staleMapInstance =
        !this.mapLeaflet ||
        this.mapLeaflet._container !== mapDom ||
        !mapDom.querySelector(".leaflet-container");

      if (staleMapInstance) {
        if (this.mapLeaflet) {
          this.mapLeaflet.remove();
          this.mapLeaflet = null;
          this.baseTileLayer = null;
          this.districtBoundaryLayer = null;
          this.pointLayerGroup = null;
        }
        mapDom.innerHTML = "";
        this.mapLeaflet = window.L.map(mapDom, {
          zoomControl: true,
          scrollWheelZoom: true,
        }).setView(this.mapCenter, 11);

        this.baseTileLayer = window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 19,
          attribution: "&copy; OpenStreetMap contributors",
        }).addTo(this.mapLeaflet);
      }

      if (this.districtBoundaryLayer) {
        this.mapLeaflet.removeLayer(this.districtBoundaryLayer);
        this.districtBoundaryLayer = null;
      }
      if (this.pointLayerGroup) {
        this.mapLeaflet.removeLayer(this.pointLayerGroup);
        this.pointLayerGroup = null;
      }

      if (geojson) {
        this.districtBoundaryLayer = window.L.geoJSON(geojson, {
          style: {
            color: "#5b83ff",
            weight: 2,
            fillColor: "#eef4ff",
            fillOpacity: 0.18,
          },
        }).addTo(this.mapLeaflet);
        this.mapLeaflet.fitBounds(this.districtBoundaryLayer.getBounds(), { padding: [24, 24] });
      } else if (Array.isArray(boundingBox) && boundingBox.length === 4) {
        const south = Number(boundingBox[0]);
        const north = Number(boundingBox[1]);
        const west = Number(boundingBox[2]);
        const east = Number(boundingBox[3]);
        this.mapLeaflet.fitBounds([[south, west], [north, east]], { padding: [24, 24] });
      } else {
        this.mapLeaflet.setView(this.mapCenter, 11);
      }

      this.pointLayerGroup = window.L.layerGroup().addTo(this.mapLeaflet);
      this.mapKeyPoints.forEach((point) => {
        const marker = window.L.circleMarker([point.value[1], point.value[0]], {
          radius: 7,
          color: this.plannerMode === "PPO" ? "#45c38a" : "#ff81ba",
          weight: 2,
          fillColor: this.plannerMode === "PPO" ? "#45c38a" : "#ff81ba",
          fillOpacity: 0.85,
        });
        marker.bindPopup(`<strong>${escapeHtml(point.name)}</strong><br>${escapeHtml(point.description)}`);
        marker.addTo(this.pointLayerGroup);
      });

      const refreshMapSize = () => {
        if (this.mapLeaflet) {
          this.mapLeaflet.invalidateSize(true);
        }
      };

      this.mapLeaflet.whenReady(() => {
        setTimeout(refreshMapSize, 80);
      });
      requestAnimationFrame(() => {
        setTimeout(refreshMapSize, 160);
      });
    },
    renderPathMapChart() {
      if (this.activePage !== "pathdemo") return;
      this.$nextTick(() => {
        const attemptRender = (retries = 8) => {
          const mapDom = document.getElementById("longquanyi-map-chart");
          if (!mapDom) {
            if (retries > 0) setTimeout(() => attemptRender(retries - 1), 80);
            return;
          }
          this.renderLeafletMap(this.mapGeoJson, this.mapBoundingBox);
        };
        attemptRender();
      });
    },
    playPlanner(mode) {
      this.plannerMode = mode;
      this.$nextTick(() => this.renderPathMapChart());
    },
    scrollMessagesToBottom() {
      const box = this.$refs.messageBox;
      if (box) box.scrollTop = box.scrollHeight;
    },
    formattedMessage(content) {
      return formatRichText(content || "");
    },
    messageRole(role) {
      return role === "user" ? "用户" : "助手";
    },
    handleComposerShortcut(event) {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        this.sendMessage();
      }
    },
    renderTrainingCurve() {
      const trace = this.training?.trace || [];
      if (!trace.length) {
        return `<div class="curve-legend">还没有可用的 reward 轨迹。</div>`;
      }
      const rewards = trace.map((item) => Number(item.reward ?? item.average_reward ?? 0));
      const min = Math.min(...rewards);
      const max = Math.max(...rewards);
      const width = 760;
      const height = 150;
      const padding = 12;
      const range = max - min || 1;
      const stepX = rewards.length > 1 ? (width - padding * 2) / (rewards.length - 1) : 0;
      const points = rewards
        .map((reward, index) => {
          const x = padding + stepX * index;
          const y = height - padding - ((reward - min) / range) * (height - padding * 2);
          return `${x},${y}`;
        })
        .join(" ");

      return `
        <svg class="curve-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <polyline points="${points}" fill="none" stroke="#5b83ff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></polyline>
        </svg>
        <div class="curve-legend">最低 ${min.toFixed(4)} / 最高 ${max.toFixed(4)} / 共 ${rewards.length} 个训练点</div>
      `;
    },
  },
  async mounted() {
    await this.refreshAll();
    this.scrollMessagesToBottom();
  },
  template: `
    <div class="workspace">
      <header class="topbar">
        <div class="topbar-brand">
          <div class="topbar-mark">A</div>
          <div class="topbar-copy">
            <div class="title">{{ text.appTitle }}</div>
            <div class="subtitle">{{ text.appSubtitle }}</div>
          </div>
        </div>
        <nav class="topbar-nav">
          <el-menu mode="horizontal" :default-active="activePage" @select="activePage = $event">
            <el-menu-item index="workspace">对话工作台</el-menu-item>
            <el-menu-item index="knowledge">知识库</el-menu-item>
            <el-menu-item index="status">系统状态</el-menu-item>
            <el-menu-item index="training">训练中心</el-menu-item>
            <el-menu-item index="policy">策略作用展示</el-menu-item>
            <el-menu-item index="repos">仓库工具</el-menu-item>
            <el-menu-item index="pathdemo">路径规划演示</el-menu-item>
          </el-menu>
        </nav>
        <div class="topbar-meta">
          <div class="topbar-chip">RAG {{ health?.ragEnabled ? '已启用' : '未启用' }}</div>
          <div class="topbar-chip">DB {{ health?.dbReady ? '已连接' : '未连接' }}</div>
        </div>
      </header>

      <div class="workspace-body">
        <aside class="sidebar">
          <section class="sidebar-card">
            <h3 class="sidebar-title">研究工作流</h3>
            <p class="sidebar-copy">先把 Agent 本体做完整，再用检索策略学习去优化它。这就是我们当前这条主线。</p>
          </section>

          <section class="sidebar-card session-card">
            <div class="session-toolbar">
              <div>
                <h3 class="sidebar-title">会话导航</h3>
                <div class="session-meta">{{ sessionCountLabel }}</div>
              </div>
              <el-button type="primary" plain size="small" @click="createSession">新会话</el-button>
            </div>
            <el-menu class="session-menu" :default-active="sessionId" @select="selectSession">
              <el-menu-item v-for="session in sessions" :key="session.id" :index="session.id">
                <div class="session-entry">
                  <strong>{{ session.title || text.untitled }}</strong>
                  <span>{{ session.summary || '等待第一条消息...' }}</span>
                </div>
              </el-menu-item>
            </el-menu>
          </section>
        </aside>

        <main class="content-shell">
          <section v-if="activePage === 'workspace'" class="content-grid">
            <div class="content-main">
              <article class="card full-card">
                <header class="card-header">
                  <div>
                    <div class="eyebrow">Conversation Workspace</div>
                    <h2 class="card-title">{{ text.workspaceTitle }}</h2>
                    <div class="card-subtitle">{{ text.workspaceSubtitle }}</div>
                  </div>
                  <div class="topbar-chip">{{ chatStatus }}</div>
                </header>
                <div ref="messageBox" class="messages">
                  <div class="messages-inner" v-if="currentMessages.length">
                    <div v-for="(message, idx) in currentMessages" :key="idx" class="message" :class="message.role">
                      <div class="message-card">
                        <div class="message-role">{{ messageRole(message.role) }}</div>
                        <div class="message-bubble" v-html="formattedMessage(message.content)"></div>
                      </div>
                    </div>
                  </div>
                  <div v-else class="messages-inner">
                    <div class="empty-state">{{ text.emptySession }}</div>
                  </div>
                </div>
                <div class="composer-card">
                  <el-input
                    v-model="queryInput"
                    type="textarea"
                    :rows="4"
                    resize="none"
                    placeholder="问一个轨迹 / 强化学习问题，或让助手基于知识库、仓库与附件给你做分析。"
                    @keydown="handleComposerShortcut"
                  />
                  <div v-if="composerAttachment" class="attachment-chip-row">
                    <div class="attachment-chip">
                      <div class="attachment-name">{{ composerAttachment.name }}</div>
                      <div class="attachment-meta">
                        {{ Math.max(1, Math.round((composerAttachment.content || '').length / 1024)) }} KB 文本
                        <span v-if="composerAttachment.truncated"> · 已截断到 12000 字符</span>
                      </div>
                    </div>
                    <el-button text type="danger" @click="clearComposerAttachment">移除</el-button>
                  </div>
                  <div class="composer-actions">
                    <div style="display:flex;align-items:center;gap:10px;">
                      <input ref="filePicker" type="file" style="display:none" @change="handleAttachmentChange" />
                      <el-button circle @click="triggerAttachmentPicker">+</el-button>
                      <el-input-number v-model="topK" :min="1" :max="10" size="small" />
                    </div>
                    <el-button type="primary" :loading="sending" @click="sendMessage">发送</el-button>
                  </div>
                  <div class="composer-hint">支持公式、真实表格渲染和文本类本地附件。快捷键：Ctrl/Command + Enter 发送。</div>
                </div>
              </article>
            </div>

            <aside class="content-side">
              <article class="card full-card">
                <header class="card-header">
                  <div>
                    <div class="eyebrow">Evidence</div>
                    <h3 class="card-title side-title">检索来源</h3>
                  </div>
                </header>
                <div class="card-body side-scroll">
                  <div v-if="sources.length" class="source-list">
                    <div v-for="(source, idx) in sources" :key="idx" class="source-card">
                      <h4>{{ source.title || source.source || ('来源 ' + (idx + 1)) }}</h4>
                      <div class="source-path">{{ source.path || source.source || '-' }}</div>
                      <div class="source-snippet">{{ source.snippet || '未返回摘要。' }}</div>
                    </div>
                  </div>
                  <div v-else class="empty-state">{{ text.noSources }}</div>
                </div>
              </article>

              <article class="card full-card">
                <header class="card-header">
                  <div>
                    <div class="eyebrow">Context</div>
                    <h3 class="card-title side-title">上下文快照</h3>
                  </div>
                </header>
                <div class="card-body side-scroll">
                  <div class="context-box">{{ contextPreview }}</div>
                </div>
              </article>

              <article class="card full-card">
                <header class="card-header">
                  <div>
                    <div class="eyebrow">Quick Training</div>
                    <h3 class="card-title side-title">训练摘要</h3>
                  </div>
                </header>
                <div class="card-body">
                  <div v-if="training?.available" class="training-summary">
                    <div><strong>最近训练</strong> {{ training.run?.name || '-' }}</div>
                    <div><strong>Reward 增益</strong> {{ Number(trainingMetrics.reward_gain_vs_baseline || 0).toFixed(4) }}</div>
                    <div><strong>Source Hit</strong> {{ Number(trainingMetrics.trained_average_source_hit || 0).toFixed(4) }}</div>
                    <div><strong>Topic Hit</strong> {{ Number(trainingMetrics.trained_average_topic_hit || 0).toFixed(4) }}</div>
                  </div>
                  <div v-else class="training-empty">{{ text.noTraining }}</div>
                </div>
              </article>
            </aside>
          </section>

          <section v-else-if="activePage === 'knowledge'" class="page-fill page-single">
            <article class="card full-card">
              <header class="card-header">
                <div>
                  <div class="eyebrow">Knowledge Base</div>
                  <h2 class="card-title">{{ text.knowledgeTitle }}</h2>
                  <div class="card-subtitle">{{ text.knowledgeSubtitle }}</div>
                </div>
                <el-button type="primary" plain @click="rebuildKnowledge">重建知识库</el-button>
              </header>
              <div class="card-body page-scroll">
                <div class="info-stat-grid" v-if="knowledge">
                  <div class="info-stat">
                    <div class="label">知识库标题</div>
                    <div class="value compact">{{ knowledge.title || '-' }}</div>
                  </div>
                  <div class="info-stat">
                    <div class="label">文档数</div>
                    <div class="value">{{ knowledge.documentCount || 0 }}</div>
                  </div>
                  <div class="info-stat">
                    <div class="label">Chunk 数</div>
                    <div class="value">{{ knowledge.chunkCount || 0 }}</div>
                  </div>
                  <div class="info-stat">
                    <div class="label">研究焦点</div>
                    <div class="value compact">{{ knowledge.researchFocus || '-' }}</div>
                  </div>
                </div>
                <div class="section-block">
                  <h3 class="section-title">主题分布</h3>
                  <div class="knowledge-topic-list">
                    <div v-for="topic in knowledgeTopics" :key="topic.name" class="topic-card">
                      <div class="name">{{ topic.name }}</div>
                      <div class="count">{{ topic.count }}</div>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          </section>

          <section v-else-if="activePage === 'status'" class="page-fill page-single">
            <article class="card full-card">
              <header class="card-header">
                <div>
                  <div class="eyebrow">System Status</div>
                  <h2 class="card-title">{{ text.statusTitle }}</h2>
                  <div class="card-subtitle">{{ text.statusSubtitle }}</div>
                </div>
                <el-button type="primary" plain @click="refreshHealth">刷新状态</el-button>
              </header>
              <div class="card-body page-scroll">
                <div class="health-list">
                  <div v-for="row in statusRows" :key="row.label" class="health-row">
                    <span>{{ row.label }}</span>
                    <strong :class="row.level || ''">{{ row.value }}</strong>
                  </div>
                </div>
              </div>
            </article>
          </section>

          <section v-else-if="activePage === 'training'" class="page-fill page-single">
            <article class="card full-card">
              <header class="card-header">
                <div>
                  <div class="eyebrow">Training Center</div>
                  <h2 class="card-title">{{ text.trainingTitle }}</h2>
                  <div class="card-subtitle">{{ text.trainingSubtitle }}</div>
                </div>
                <el-button type="primary" plain @click="refreshTraining">刷新训练</el-button>
              </header>
              <div class="card-body page-scroll" v-if="training?.available">
                <div class="training-summary">
                  <div><strong>训练名</strong> {{ training.run?.name || '-' }}</div>
                  <div><strong>状态</strong> {{ training.run?.status || '-' }}</div>
                  <div><strong>更新时间</strong> {{ training.run?.updatedAt || '-' }}</div>
                </div>
                <div class="training-metric-grid">
                  <div class="training-metric">
                    <div class="label">训练后 Average Reward</div>
                    <div class="value">{{ Number(trainingMetrics.trained_average_reward || 0).toFixed(4) }}</div>
                  </div>
                  <div class="training-metric">
                    <div class="label">Baseline Average Reward</div>
                    <div class="value">{{ Number(trainingMetrics.baseline_average_reward || 0).toFixed(4) }}</div>
                  </div>
                  <div class="training-metric">
                    <div class="label">训练后 Source Hit</div>
                    <div class="value">{{ Number(trainingMetrics.trained_average_source_hit || 0).toFixed(4) }}</div>
                  </div>
                  <div class="training-metric">
                    <div class="label">Baseline Source Hit</div>
                    <div class="value">{{ Number(trainingMetrics.baseline_average_source_hit || 0).toFixed(4) }}</div>
                  </div>
                </div>
                <div class="curve-card">
                  <h3>Reward 曲线</h3>
                  <div class="curve-svg-shell" v-html="renderTrainingCurve()"></div>
                </div>
                <div v-if="representativeEpisode" class="episode-card">
                  <h3>代表 episode</h3>
                  <div class="episode-query">{{ representativeEpisode.query }}</div>
                  <div class="episode-grid">
                    <div class="episode-row"><span>DQN 动作</span><b>{{ representativeEpisode.chosen_action }}</b></div>
                    <div class="episode-row"><span>DQN Reward</span><b>{{ Number(representativeEpisode.reward || 0).toFixed(4) }}</b></div>
                    <div class="episode-row"><span>Baseline 动作</span><b>{{ baselineEpisode?.chosen_action || '-' }}</b></div>
                    <div class="episode-row"><span>Baseline Reward</span><b>{{ Number(baselineEpisode?.reward || 0).toFixed(4) }}</b></div>
                  </div>
                </div>
              </div>
              <div class="card-body" v-else>
                <div class="training-empty">{{ text.noTraining }}</div>
              </div>
            </article>
          </section>

          <section v-else-if="activePage === 'policy'" class="page-fill page-single">
            <article class="card full-card">
              <header class="card-header">
                <div>
                  <div class="eyebrow">Policy Impact</div>
                  <h2 class="card-title">{{ text.policyTitle }}</h2>
                  <div class="card-subtitle">{{ text.policySubtitle }}</div>
                </div>
                <el-button type="primary" plain @click="refreshTraining">刷新结果</el-button>
              </header>
              <div class="card-body page-scroll" v-if="training?.available">
                <div class="info-stat-grid">
                  <div class="info-stat">
                    <div class="label">当前已落地</div>
                    <div class="value compact">DQN 检索策略学习</div>
                  </div>
                  <div class="info-stat">
                    <div class="label">下一步最适合</div>
                    <div class="value compact">PPO 多步工具决策 / 连续策略优化</div>
                  </div>
                </div>
                <div class="policy-explainer-grid">
                  <div class="policy-note">
                    <h3>为什么先做检索策略学习</h3>
                    <p>它最贴近当前 Agent 本体，而且状态、动作、奖励都很清楚。状态是用户问题和上下文；动作是 query rewrite、主题偏好和证据选择；奖励直接来自 source hit、topic hit 和 point recall。</p>
                  </div>
                  <div class="policy-note">
                    <h3>DQN 现在实际在优化什么</h3>
                    <p>DQN 不是直接优化整段回答，而是在学“这个 Agent 该怎么检索更合适”。这让强化学习的效果可以被直接量化，也更容易在工作台里展示清楚。</p>
                  </div>
                </div>
                <div class="chart-card">
                  <h3>Baseline vs DQN</h3>
                  <div id="policy-compare-chart" class="echart-box"></div>
                </div>
                <div class="chart-card">
                  <h3>DQN Reward 轨迹</h3>
                  <div id="policy-reward-chart" class="echart-box"></div>
                </div>
              </div>
              <div class="card-body" v-else>
                <div class="training-empty">{{ text.noTraining }}</div>
              </div>
            </article>
          </section>

          <section v-else-if="activePage === 'repos'" class="page-fill page-single">
            <article class="card full-card">
              <header class="card-header">
                <div>
                  <div class="eyebrow">Repository Tools</div>
                  <h2 class="card-title">{{ text.reposTitle }}</h2>
                  <div class="card-subtitle">{{ text.reposSubtitle }}</div>
                </div>
                <el-button type="primary" plain @click="refreshRepos">刷新仓库</el-button>
              </header>
              <div class="card-body page-scroll">
                <div class="info-stat-grid">
                  <div class="info-stat">
                    <div class="label">本地项目数量</div>
                    <div class="value">{{ repos.length }}</div>
                  </div>
                  <div class="info-stat">
                    <div class="label">仓库目录</div>
                    <div class="value compact">trl/repos</div>
                  </div>
                </div>
                <div class="section-block">
                  <h3 class="section-title">GitHub 克隆</h3>
                  <div class="form-grid">
                    <el-input v-model="cloneForm.repoUrl" placeholder="输入 GitHub 仓库地址，例如 https://github.com/user/repo.git" />
                    <div class="inline-form">
                      <el-input v-model="cloneForm.branch" placeholder="分支名（可选）" />
                      <el-input v-model="cloneForm.targetName" placeholder="本地目录名（可选）" />
                      <el-button type="primary" :loading="cloneLoading" @click="cloneRepo">开始克隆</el-button>
                    </div>
                    <div class="status-note">{{ cloneStatus || '输入 GitHub 仓库地址后，可以直接把项目克隆到 trl/repos，并作为后续知识检索和项目分析入口。' }}</div>
                  </div>
                </div>
                <div class="section-block">
                  <h3 class="section-title">本地仓库管理</h3>
                  <div v-if="repos.length" class="repo-list">
                    <div v-for="repo in repos" :key="repo.path" class="repo-card">
                      <div class="repo-name">{{ repo.name }}</div>
                      <div class="repo-path">{{ repo.path }}</div>
                    </div>
                  </div>
                  <div v-else class="empty-state">还没有本地仓库。先在上面克隆一个项目，我们就能继续做代码分析、README 阅读和知识接入。</div>
                </div>
              </div>
            </article>
          </section>

          <section v-else-if="activePage === 'pathdemo'" class="page-fill page-single">
            <article class="card full-card">
              <header class="card-header">
                <div>
                  <div class="eyebrow">RL Path Planning</div>
                  <h2 class="card-title">{{ text.pathTitle }}</h2>
                  <div class="card-subtitle">{{ text.pathSubtitle }}</div>
                </div>
                <div style="display:flex;gap:10px;align-items:center;">
                  <el-button type="primary" plain @click="playPlanner('DQN')">切到 DQN 视角</el-button>
                  <el-button type="success" plain @click="playPlanner('PPO')">切到 PPO 视角</el-button>
                </div>
              </header>
              <div class="card-body page-scroll">
                <div class="policy-explainer-grid">
                  <div class="policy-note">
                    <h3>当前这一步我们先做什么</h3>
                    <p>先把龙泉驿区真实行政区划 GeoJSON 底图挂到 ECharts 上。等你告诉我具体路线、起终点和关键节点后，我们再把 PPO / DQN 的逐步路径动画加进去。</p>
                  </div>
                  <div class="policy-note">
                    <h3>为什么这页和检索策略学习要并存</h3>
                    <p>路径规划页负责让人直观看懂 RL 如何在地图上决策；检索策略学习页负责解释 RL 如何真实提升当前 Agent 的知识问答与检索质量。一个偏可视化，一个偏本体优化。</p>
                  </div>
                </div>

                <div class="path-demo-layout">
                  <div class="map-panel">
                    <div id="longquanyi-map-chart" class="echart-box map-echart"></div>
                  </div>

                  <div class="map-side">
                    <div class="analysis-card">
                      <div class="analysis-label">当前视角</div>
                      <div class="analysis-value">{{ plannerMode }}</div>
                    </div>
                    <div class="analysis-card">
                      <div class="analysis-label">地图状态</div>
                      <div class="analysis-value compact">{{ mapStatus }}</div>
                    </div>
                    <div class="analysis-card">
                      <div class="analysis-label">地图来源</div>
                      <div class="analysis-value compact">{{ mapSource || '尚未成功加载' }}</div>
                    </div>
                    <div class="policy-note">
                      <h3>关键点</h3>
                      <p v-for="point in mapKeyPoints" :key="point.name"><strong>{{ point.name }}</strong>：{{ point.description }}</p>
                    </div>
                    <div class="policy-note">
                      <h3>当前说明</h3>
                      <p>{{ mapModeDescription }}</p>
                    </div>
                    <div class="policy-note">
                      <h3>图例</h3>
                      <p><span class="legend-line dqn"></span> DQN 视角占位路径</p>
                      <p><span class="legend-line ppo"></span> PPO 视角占位路径</p>
                      <p><span class="legend-dot start"></span> 关键点标记</p>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          </section>
        </main>
      </div>
    </div>
  `,
}).use(ElementPlus).mount("#app");
