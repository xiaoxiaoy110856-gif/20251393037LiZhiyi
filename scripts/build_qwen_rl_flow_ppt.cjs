const fs = require("fs");
const fsp = fs.promises;
const path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const JSZip = require("jszip");
const SHAPE = new pptxgen().ShapeType;

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, "docs", "ppt");
const PREVIEW_DIR = path.join(OUT_DIR, "previews");
const ASSET_DIR = path.join(OUT_DIR, "assets");
const PPTX_PATH = path.join(OUT_DIR, "qwen_rl_agent_flow.pptx");
const INSPECTION_PATH = path.join(OUT_DIR, "qwen_rl_agent_flow_inspection.json");
const GENERATED_IMAGES_DIR = path.join(ROOT, "outputs", "generated_images");
const W = 13.333;
const H = 7.5;
const PX_W = 1600;
const PX_H = 900;

const C = {
  bg: "FFF7FB",
  bg2: "FDF2F8",
  panel: "FFFFFF",
  panel2: "FFE5F1",
  cyan: "EC4899",
  green: "10B981",
  amber: "F59E0B",
  red: "E11D48",
  blue: "14B8A6",
  text: "17312B",
  muted: "66736E",
  faint: "EFD6E2",
  line: "A8DCCB",
};

const methods = [
  { name: "PPO", reward: 0.0599, source: 0.1019, topic: 0.0972, color: C.blue },
  { name: "DPO", reward: 0.09, source: 0.1388, topic: 0.0856, color: C.cyan },
  { name: "ORPO", reward: -0.0913, source: -0.0973, topic: -0.0208, color: C.red },
  { name: "LinUCB", reward: 0.1852, source: 0.3425, topic: 0.0903, color: C.green, best: true },
  { name: "Dueling DDQN", reward: 0.1702, source: 0.3287, topic: 0.081, color: C.amber },
];

const percent = (v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}pp`;
const esc = (s) =>
  String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

function addBg(slide) {
  slide.background = { color: C.bg };
  slide.addShape(SHAPE.rect, {
    x: 0,
    y: 0,
    w: W,
    h: H,
    fill: { color: C.bg },
    line: { color: C.bg },
  });
  slide.addShape(SHAPE.rect, {
    x: 0,
    y: 0,
    w: W,
    h: 0.1,
    fill: { color: C.cyan },
    line: { color: C.cyan },
    transparency: 15,
  });
}

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    margin: opts.margin ?? 0.05,
    fontFace: opts.fontFace || "Microsoft YaHei",
    fontSize: opts.size || 16,
    color: opts.color || C.text,
    bold: !!opts.bold,
    breakLine: false,
    fit: "shrink",
    valign: opts.valign || "mid",
    align: opts.align || "left",
    transparency: opts.transparency || 0,
    rotate: opts.rotate || 0,
    charSpace: 0,
  });
}

function addTitle(slide, title, subtitle) {
  addText(slide, title, 0.58, 0.34, 8.8, 0.45, { size: 22, bold: true });
  if (subtitle) addText(slide, subtitle, 0.62, 0.86, 9.5, 0.3, { size: 9.5, color: C.muted });
  slide.addShape(SHAPE.line, {
    x: 0.6,
    y: 1.2,
    w: 12.1,
    h: 0,
    line: { color: C.faint, width: 1 },
  });
}

function addNode(slide, x, y, w, h, title, body, opts = {}) {
  const shape = opts.round ? SHAPE.roundRect : SHAPE.rect;
  slide.addShape(shape, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: opts.fill || C.panel, transparency: opts.transparency ?? 0 },
    line: { color: opts.line || C.line, width: opts.lineWidth || 1.2 },
  });
  addText(slide, title, x + 0.16, y + 0.08, w - 0.32, 0.25, {
    size: opts.titleSize || 11,
    bold: true,
    color: opts.titleColor || C.text,
  });
  if (body) {
    addText(slide, body, x + 0.16, y + 0.42, w - 0.32, h - 0.52, {
      size: opts.bodySize || 8.3,
      color: opts.bodyColor || C.muted,
      valign: "top",
    });
  }
}

function addArrow(slide, x1, y1, x2, y2, color = C.cyan, width = 1.6) {
  slide.addShape(SHAPE.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: { color, width, beginArrowType: "none", endArrowType: "triangle" },
  });
}

function assetPath(name) {
  return path.join(ASSET_DIR, name);
}

function addVisual(slide, name, x, y, w, h) {
  slide.addImage({ path: assetPath(name), x, y, w, h });
}

function addMiniNote(slide, title, body, x, y, w, color = C.cyan) {
  addText(slide, title, x, y, w, 0.25, { size: 11.5, bold: true, color });
  addText(slide, body, x, y + 0.32, w, 0.42, { size: 8.6, color: C.muted, valign: "top" });
}

function addPill(slide, text, x, y, w, color = C.cyan) {
  slide.addShape(SHAPE.roundRect, {
    x,
    y,
    w,
    h: 0.34,
    rectRadius: 0.08,
    fill: { color: C.panel2 },
    line: { color, width: 1 },
  });
  addText(slide, text, x + 0.1, y + 0.04, w - 0.2, 0.24, { size: 8.5, color: C.text, align: "center" });
}

function addMetric(slide, label, value, x, y, w, color) {
  addText(slide, label, x, y, w, 0.22, { size: 8.5, color: C.muted });
  addText(slide, value, x, y + 0.24, w, 0.42, { size: 18, bold: true, color });
}

function addBar(slide, label, value, x, y, w, color, min = -0.1, max = 0.35) {
  const zero = x + ((0 - min) / (max - min)) * w;
  const end = x + ((value - min) / (max - min)) * w;
  slide.addShape(SHAPE.line, {
    x,
    y: y + 0.17,
    w,
    h: 0,
    line: { color: C.faint, width: 8, transparency: 35 },
  });
  slide.addShape(SHAPE.line, {
    x: zero,
    y: y + 0.17,
    w: end - zero,
    h: 0,
    line: { color, width: 8 },
  });
  slide.addShape(SHAPE.line, {
    x: zero,
    y: y - 0.04,
    w: 0,
    h: 0.42,
    line: { color: C.muted, width: 1 },
  });
  addText(slide, label, x - 1.1, y + 0.02, 1.0, 0.22, { size: 8.2, color: C.text, align: "right" });
  addText(slide, percent(value), x + w + 0.12, y + 0.02, 1.0, 0.22, { size: 8.2, color });
}

function slide1(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addText(slide, "Qwen 驱动的", 0.72, 1.3, 4.8, 0.6, { size: 24, color: C.cyan, bold: true });
  addText(slide, "RL-RAG Agent 流程", 0.68, 1.92, 10.6, 0.82, { size: 42, bold: true });
  addText(slide, "用强化学习训练检索策略，让本地大模型找得更准、答得更有依据", 0.76, 2.88, 8.8, 0.36, {
    size: 14,
    color: C.muted,
  });
  addText(slide, "Qwen 生成", 0.82, 4.1, 1.6, 0.32, { size: 12, bold: true, color: C.text });
  addText(slide, "RL 选策略", 3.02, 4.1, 1.6, 0.32, { size: 12, bold: true, color: C.green });
  addText(slide, "RAG 找证据", 5.2, 4.1, 1.9, 0.32, { size: 12, bold: true, color: C.amber });
  addArrow(slide, 2.3, 4.25, 2.88, 4.25);
  addArrow(slide, 4.46, 4.25, 5.06, 4.25);
  addMetric(slide, "当前最优策略", "LinUCB", 8.9, 1.3, 2.0, C.green);
  addMetric(slide, "Source Hit 提升", "+34.25pp", 8.9, 2.16, 2.1, C.green);
  addMetric(slide, "Topic Hit 提升", "+9.03pp", 8.9, 3.02, 2.1, C.cyan);
  addText(slide, "2026.05 · 轨迹 / 强化学习 / 论文 RAG", 0.76, 6.78, 6.2, 0.25, { size: 8.5, color: C.muted });
}

function slide2(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "端到端流程图", "从用户问题进入 Qwen，到 RL 检索策略选择，再回到 Qwen 生成答案");
  const nodes = [
    ["用户问题", "Vue 前端输入\n历史对话进入后端", 0.62, 1.78],
    ["Qwen Planner", "Ollama 本地 Qwen3.5\n决定 final 或 tool_use", 2.58, 1.78],
    ["工具调用", "search_project_docs\n传入 query 与上下文", 4.72, 1.78],
    ["RL 检索策略", "LinUCB / DDQN / DPO\n选择 retrieval action", 6.86, 1.78],
    ["论文知识库", "约 200 PDF\n203 docs / 13,555 chunks", 9.1, 1.78],
    ["Qwen 答案", "带来源、主题和证据\n输出给用户", 11.18, 1.78],
  ];
  nodes.forEach(([t, b, x, y], idx) => {
    addNode(slide, x, y, idx === 5 ? 1.55 : 1.75, 1.05, t, b, {
      fill: idx === 3 ? "E7F8F0" : C.panel,
      line: idx === 3 ? C.green : C.line,
      titleColor: idx === 3 ? C.green : C.text,
      bodySize: 7.4,
    });
  });
  for (let i = 0; i < nodes.length - 1; i += 1) {
    const x1 = nodes[i][2] + (i === 5 ? 1.55 : 1.75);
    addArrow(slide, x1 + 0.07, 2.3, nodes[i + 1][2] - 0.1, 2.3, i === 2 ? C.green : C.cyan);
  }
  addNode(
    slide,
    1.08,
    4.35,
    11.15,
    1.2,
    "关键闭环",
    "Qwen 不直接盲答；遇到项目资料问题时先调用检索工具。RL 只负责选择“怎么检索”，RAG 返回证据后，Qwen 再负责组织语言和推理回答。",
    { fill: "FFF0F6", line: C.cyan, titleColor: C.cyan, bodySize: 11 }
  );
}

function slide3(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "Qwen 的角色", "Qwen 是 Agent 的规划器和最终表达层；RL 不替代 Qwen，只优化检索动作");
  addNode(slide, 0.78, 1.65, 3.05, 1.18, "1. 理解问题", "读取用户问题、历史对话和系统规则，判断是否需要项目知识。", {
    line: C.cyan,
    titleColor: C.cyan,
  });
  addNode(slide, 5.08, 1.65, 3.05, 1.18, "2. 发起工具", "输出 JSON tool_use：search_project_docs(query)。", {
    line: C.green,
    titleColor: C.green,
  });
  addNode(slide, 9.38, 1.65, 3.05, 1.18, "3. 生成答案", "读取 observation，综合来源、主题、证据，生成最终回答。", {
    line: C.amber,
    titleColor: C.amber,
  });
  addArrow(slide, 3.92, 2.24, 4.92, 2.24);
  addArrow(slide, 8.22, 2.24, 9.22, 2.24, C.green);
  addNode(
    slide,
    1.0,
    4.0,
    5.15,
    1.25,
    "工具调用格式",
    '{ "tool_use": { "name": "search_project_docs", "arguments": { "query": "..." } } }',
    { fill: "FFFFFF", line: C.line, titleColor: C.text, bodySize: 10.5 }
  );
  addNode(
    slide,
    7.15,
    4.0,
    4.95,
    1.25,
    "最终输出要求",
    "答案必须尽量引用检索证据；当证据不足时说明限制，避免把 Qwen 的猜测包装成结论。",
    { fill: "FFFFFF", line: C.line, titleColor: C.text, bodySize: 10.5 }
  );
}

function slide4(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "RL 检索策略层", "把一次检索视为离散动作选择：根据 query 特征，选择最可能命中证据的 action");
  addNode(slide, 0.7, 1.55, 2.65, 3.65, "Query Features", "关键词\n任务意图\n轨迹子领域\n是否偏论文 / benchmark\n是否偏 reward / policy", {
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 11,
  });
  addNode(slide, 5.05, 2.12, 2.7, 2.5, "Policy", "LinUCB 优先\nDueling DDQN 对照\nDPO / PPO / ORPO 对照", {
    fill: "E7F8F0",
    line: C.green,
    titleColor: C.green,
    bodySize: 11,
  });
  addNode(slide, 9.22, 1.55, 3.08, 3.65, "Retrieval Actions", "baseline\nrl_focus / reward_focus\ntrajectory_focus\npaper_focus\ncompression_focus\nplanning_focus\nsimilarity_focus / broad_search", {
    line: C.amber,
    titleColor: C.amber,
    bodySize: 9.5,
  });
  addArrow(slide, 3.55, 3.32, 4.82, 3.32, C.cyan);
  addArrow(slide, 7.92, 3.32, 9.0, 3.32, C.green);
  addNode(
    slide,
    1.12,
    5.88,
    10.9,
    0.78,
    "动作的含义",
    "action 不改变 Qwen 参数，而是改写检索 query、top_k 和主题权重；收益体现在 Source Hit / Topic Hit / Point Recall 上。",
    { fill: "FFF0F6", line: C.faint, titleColor: C.text, bodySize: 10.2 }
  );
}

function slide5(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "训练逻辑", "用已有题库和本地论文库离线评估每个 action，再训练策略选择器");
  const steps = [
    ["评测题库", "36 个代表性问题\n覆盖 RL / 轨迹 / 检索", 0.76, 1.66],
    ["动作枚举", "对每题跑 9 个 retrieval action\n记录检索结果", 3.2, 1.66],
    ["奖励计算", "reward = Source Hit\n+ Topic Hit + Point Recall", 5.75, 1.66],
    ["策略训练", "PPO / DPO / ORPO\nLinUCB / Dueling DDQN", 8.24, 1.66],
    ["上线选择", "优先 LinUCB\n保留 DDQN 作对照", 10.78, 1.66],
  ];
  steps.forEach(([t, b, x, y], idx) =>
    addNode(slide, x, y, idx === 4 ? 1.72 : 1.88, 1.26, t, b, {
      fill: idx === 3 ? "E7F8F0" : C.panel,
      line: idx === 3 ? C.green : C.line,
      titleColor: idx === 3 ? C.green : C.text,
      bodySize: 8.3,
    })
  );
  for (let i = 0; i < steps.length - 1; i += 1) {
    addArrow(slide, steps[i][2] + 1.95, 2.29, steps[i + 1][2] - 0.1, 2.29, i === 2 ? C.green : C.cyan);
  }
  addNode(slide, 1.0, 4.4, 3.25, 1.1, "Source Hit", "检索结果是否命中目标论文/来源", {
    line: C.green,
    titleColor: C.green,
  });
  addNode(slide, 5.02, 4.4, 3.25, 1.1, "Topic Hit", "检索内容是否覆盖目标主题", {
    line: C.cyan,
    titleColor: C.cyan,
  });
  addNode(slide, 9.04, 4.4, 3.25, 1.1, "Point Recall", "关键知识点是否被召回", {
    line: C.amber,
    titleColor: C.amber,
  });
  addText(slide, "训练目标：不是让模型“说得更漂亮”，而是让 Agent 更稳定地找到正确证据。", 1.1, 6.15, 10.7, 0.35, {
    size: 13,
    color: C.text,
    bold: true,
    align: "center",
  });
}

function slide6(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "效果对比", "相对各自 baseline 的提升：Source Hit 和 Topic Hit 是最能体现 RL 检索收益的指标");
  addText(slide, "Source Hit Gain", 2.4, 1.46, 1.6, 0.24, { size: 9, color: C.green, bold: true });
  addText(slide, "Topic Hit Gain", 7.05, 1.46, 1.6, 0.24, { size: 9, color: C.cyan, bold: true });
  methods.forEach((m, i) => {
    const y = 1.92 + i * 0.74;
    addText(slide, m.name, 0.8, y - 0.01, 1.25, 0.25, { size: 9.3, color: m.best ? C.green : C.text, bold: !!m.best });
    addBar(slide, "", m.source, 2.25, y, 3.6, m.best ? C.green : m.color);
    addBar(slide, "", m.topic, 6.95, y, 3.6, C.cyan);
    addText(slide, `Reward ${percent(m.reward)}`, 11.08, y + 0.02, 1.35, 0.2, {
      size: 7.8,
      color: m.reward >= 0 ? C.text : C.red,
      align: "right",
    });
  });
  addNode(
    slide,
    0.82,
    6.02,
    11.66,
    0.68,
    "结论",
    "LinUCB 与 Dueling DDQN 在 Source Hit 上提升最大，说明 RL 策略主要增强了“找对来源”的能力；ORPO 表现弱，适合作为反例说明偏好式目标并不一定适合单步检索选择。",
    { fill: "FFF0F6", line: C.green, titleColor: C.green, bodySize: 9.5 }
  );
}

function slide7(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "是否合理", "从 Qwen 到 RL-RAG 的分工是清楚的：生成归生成，检索策略归策略");
  addNode(slide, 0.82, 1.55, 3.5, 3.92, "为什么合理", "检索动作是离散选择。\n每次问题只需选择一次 action。\n结果可用 Source Hit / Topic Hit 直接打分。\n这天然适合 Contextual Bandit / DQN。", {
    line: C.green,
    titleColor: C.green,
    bodySize: 10.5,
  });
  addNode(slide, 4.92, 1.55, 3.5, 3.92, "它提升了什么", "更容易命中正确论文。\n更容易覆盖目标主题。\n回答更有证据支撑。\nQwen 的最终表达仍保留。", {
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 10.5,
  });
  addNode(slide, 9.02, 1.55, 3.5, 3.92, "边界在哪里", "当前没有微调 Qwen 本体。\n提升来自检索策略与证据质量。\n后续可加入神经 reranker 或 reward model。", {
    line: C.amber,
    titleColor: C.amber,
    bodySize: 10.5,
  });
  slide.addShape(SHAPE.rect, {
    x: 0.92,
    y: 6.0,
    w: 11.48,
    h: 0.82,
    fill: { color: "FFF0F6" },
    line: { color: C.green, width: 1.2 },
  });
  addText(slide, "推荐上线：Qwen + RAG + LinUCB；\nDDQN 作为神经网络策略对照，PPO/DPO/ORPO 作为实验组。", 1.08, 6.1, 11.16, 0.58, {
    size: 12.2,
    bold: true,
    align: "center",
  });
}

function slide8(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "Qwen 到 Agent 的实现流程", "Qwen 负责规划和表达，工具能力由 Agent 统一调度");
  const topY = 1.52;
  const nodes = [
    ["前端", "问题、附件、图片入口", 0.62, topY, 1.82],
    ["聊天接口", "/api/chat", 2.75, topY, 1.6],
    ["上下文", "摘要 + 近期消息 + RAG", 4.64, topY, 1.78],
    ["Agent 循环", "Qwen 决策 final/tool", 6.72, topY, 1.68],
    ["工具执行", "参数校验与执行", 8.68, topY, 1.78],
    ["结果回填", "工具结果返回 Qwen", 10.78, topY, 1.9],
  ];
  nodes.forEach(([t, b, x, y, w], idx) => {
    addNode(slide, x, y, w, 1.08, t, b, {
      fill: idx === 3 ? "E7F8F0" : C.panel,
      line: idx === 3 ? C.green : idx === 4 ? C.amber : C.line,
      titleColor: idx === 3 ? C.green : idx === 4 ? C.amber : C.text,
      bodySize: 7.5,
    });
  });
  for (let i = 0; i < nodes.length - 1; i += 1) {
    addArrow(slide, nodes[i][2] + nodes[i][4] + 0.06, topY + 0.55, nodes[i + 1][2] - 0.08, topY + 0.55, i >= 2 ? C.green : C.cyan);
  }
  addNode(slide, 0.95, 3.48, 3.35, 1.35, "Qwen/Ollama", "本地模型负责规划、选择工具、整合证据", {
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 9.5,
  });
  addNode(slide, 5.05, 3.48, 3.35, 1.35, "工具族", "RAG 检索、文件读写、图片生成、轨迹算法", {
    line: C.green,
    titleColor: C.green,
    bodySize: 9.5,
  });
  addNode(slide, 9.15, 3.48, 3.35, 1.35, "最终返回", "答案、证据来源、工具轨迹、图片路径", {
    line: C.amber,
    titleColor: C.amber,
    bodySize: 9.5,
  });
  addArrow(slide, 4.28, 4.15, 4.86, 4.15, C.cyan);
  addArrow(slide, 8.48, 4.15, 9.0, 4.15, C.green);
  addNode(
    slide,
    1.04,
    5.82,
    11.22,
    0.72,
    "实现原则",
    "Qwen 不直接访问系统；所有本地能力都通过可审计工具执行，再由 Qwen 组织成用户可读答案。",
    { fill: "FFF0F6", line: C.faint, titleColor: C.text, bodySize: 10.2 }
  );
}

function slide9(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "读取 / 分析文件功能", "读取、搜索、索引、修改提案和受保护写入拆开处理");
  addNode(slide, 0.76, 1.56, 2.55, 1.18, "前端入口", "上传附件\n读取文件\n编辑提案", {
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 8.8,
  });
  addNode(slide, 3.86, 1.56, 2.55, 1.18, "路径校验", "只允许 workspace 内\n拦截敏感文件", {
    line: C.green,
    titleColor: C.green,
    bodySize: 8.8,
  });
  addNode(slide, 6.96, 1.56, 2.55, 1.18, "读取与检索", "读文件\n全文搜索\n生成索引", {
    line: C.amber,
    titleColor: C.amber,
    bodySize: 8.8,
  });
  addNode(slide, 10.06, 1.56, 2.55, 1.18, "Qwen 分析", "文件内容进入上下文\n回答可带来源", {
    line: C.line,
    titleColor: C.text,
    bodySize: 8.8,
  });
  addArrow(slide, 3.42, 2.16, 3.74, 2.16);
  addArrow(slide, 6.52, 2.16, 6.84, 2.16, C.green);
  addArrow(slide, 9.62, 2.16, 9.94, 2.16, C.amber);
  addNode(slide, 0.86, 4.0, 3.55, 1.5, "修改提案", "先生成差异和新内容，前端展示 diff，用户确认后再写入。", {
    fill: "FFFFFF",
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 10,
  });
  addNode(slide, 4.88, 4.0, 3.55, 1.5, "写入保护", "写入前校验哈希，确认文件未被改动，并自动保留备份。", {
    fill: "FFFFFF",
    line: C.green,
    titleColor: C.green,
    bodySize: 10,
  });
  addNode(slide, 8.9, 4.0, 3.55, 1.5, "可追踪结果", "返回路径、差异、备份路径和新哈希，便于回看。", {
    fill: "FFFFFF",
    line: C.amber,
    titleColor: C.amber,
    bodySize: 10,
  });
}

function slide10(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "沙盒权限机制", "同一窗口同一 scope 只确认一次，后端保留硬边界");
  addNode(slide, 0.76, 1.48, 3.0, 3.8, "前端授权缓存", "ensureSandboxApproval(scope)\n\n确认结果存在 sessionStorage\n\nread_file、edit、clone 等操作按 scope 授权。", {
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 9.6,
  });
  addNode(slide, 4.24, 1.48, 3.0, 3.8, "后端安全边界", "resolve_safe_path\n\n阻止路径穿越和 symlink 逃逸\n忽略构建/依赖目录\n拦截 .env、token、key、pem 等敏感文件。", {
    line: C.green,
    titleColor: C.green,
    bodySize: 9.6,
  });
  addNode(slide, 7.72, 1.48, 3.0, 3.8, "写入限制", "只允许写入指定根目录\n\n写入前校验 hash\n写入时自动备份\n异常统一返回给前端。", {
    line: C.amber,
    titleColor: C.amber,
    bodySize: 9.6,
  });
  addNode(slide, 11.14, 1.48, 1.45, 3.8, "效果", "用户体验像 Codex：\n\n第一次确认\n同窗口复用\n越界仍拦截", {
    fill: "E7F8F0",
    line: C.green,
    titleColor: C.green,
    bodySize: 9,
  });
  addText(slide, "权限不是只靠前端确认；真正的安全约束在后端路径解析、敏感文件规则和写入根目录中。", 1.0, 6.2, 11.2, 0.33, {
    size: 12.6,
    bold: true,
    align: "center",
    color: C.text,
  });
}

function slide11(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "上下文压缩", "旧消息压成摘要和滚动状态，运行时只装配最相关上下文");
  const y = 1.72;
  const nodes = [
    ["原始消息", "保留全量原文", 0.78, y, 1.78],
    ["Token 统计", "超过阈值触发", 3.0, y, 1.78],
    ["分段摘要", "旧消息压缩", 5.22, y, 2.05],
    ["滚动状态", "项目目标/偏好/事实", 7.76, y, 2.05],
    ["运行时装配", "状态 + 相关 + 近期", 10.28, y, 2.1],
  ];
  nodes.forEach(([t, b, x, yy, w], idx) =>
    addNode(slide, x, yy, w, 1.22, t, b, {
      fill: idx >= 2 ? "E7F8F0" : C.panel,
      line: idx >= 2 ? C.green : C.line,
      titleColor: idx >= 2 ? C.green : C.text,
      bodySize: 8.2,
    })
  );
  for (let i = 0; i < nodes.length - 1; i += 1) addArrow(slide, nodes[i][2] + nodes[i][4] + 0.06, y + 0.62, nodes[i + 1][2] - 0.08, y + 0.62, i >= 1 ? C.green : C.cyan);
  addNode(slide, 1.04, 4.22, 3.3, 1.3, "压缩产物", "摘要段\n滚动状态\n装配日志", {
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 10,
  });
  addNode(slide, 5.0, 4.22, 3.3, 1.3, "选择策略", "先放滚动状态\n再检索相关摘要\n最后拼近期窗口", {
    line: C.green,
    titleColor: C.green,
    bodySize: 10,
  });
  addNode(slide, 8.96, 4.22, 3.3, 1.3, "降级模式", "数据库不可用时\n保留近期消息\n保证聊天不断路", {
    line: C.amber,
    titleColor: C.amber,
    bodySize: 10,
  });
  addText(slide, "关键点：压缩不会删除原始消息；它只是给 Qwen 构造一个更短、更相关的输入视图。", 1.0, 6.18, 11.2, 0.33, {
    size: 12.6,
    bold: true,
    align: "center",
  });
}

function slide12(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "图片生成功能实现", "图片请求进入高质量生成链路：计划、执行、评分、重试");
  const nodes = [
    ["图片意图", "识别生成请求", 0.72, 1.52],
    ["结构化计划", "正向词 + 负向词", 3.0, 1.52],
    ["ComfyUI 执行", "生成候选图片", 5.4, 1.52],
    ["质量评分", "尺寸、伪影、水印", 7.8, 1.52],
    ["重试修复", "低分自动修复", 10.2, 1.52],
  ];
  nodes.forEach(([t, b, x, y], idx) =>
    addNode(slide, x, y, 1.92, 1.22, t, b, {
      fill: idx === 3 ? "E7F8F0" : C.panel,
      line: idx === 3 ? C.green : C.line,
      titleColor: idx === 3 ? C.green : C.text,
      bodySize: 7.8,
    })
  );
  for (let i = 0; i < nodes.length - 1; i += 1) addArrow(slide, nodes[i][2] + 2.0, 2.14, nodes[i + 1][2] - 0.08, 2.14, i >= 2 ? C.green : C.cyan);
  addNode(slide, 0.92, 4.05, 3.55, 1.45, "Preset 能力", "产品、人物、海报、Logo、室内等场景都有默认参数和质量门槛。", {
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 10,
  });
  addNode(slide, 4.88, 4.05, 3.55, 1.45, "返回内容", "返回最终图、候选图、生成计划、质量报告和重试记录。", {
    line: C.green,
    titleColor: C.green,
    bodySize: 10,
  });
  addNode(slide, 8.84, 4.05, 3.55, 1.45, "前端展示", "生成图片保存到本地路径，聊天区和图片工具都可直接展示。", {
    line: C.amber,
    titleColor: C.amber,
    bodySize: 10,
  });
}

function slide13(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "知识库、地图与算法融合", "RAG 提供论文证据，地图提供轨迹场景，RL 策略选择检索和压缩动作");
  addNode(slide, 0.72, 1.45, 3.55, 3.95, "知识库 / RAG", "论文 PDF 切块\n主题自动推断\n向量检索 + 词法兜底\nRL 改写检索动作", {
    line: C.cyan,
    titleColor: C.cyan,
    bodySize: 9.2,
  });
  addNode(slide, 4.88, 1.45, 3.55, 3.95, "地图 / 轨迹", "Leaflet + OSM 展示\nOSRM 生成真实路线\n支持预设和手动起终点\n轨迹保存到 MySQL", {
    line: C.green,
    titleColor: C.green,
    bodySize: 9.2,
  });
  addNode(slide, 9.04, 1.45, 3.55, 3.95, "S3/MLSimp/RLTS", "S3 保留端点和骨架\nMLSimp 偏代表点\nRLTS 关注路口变化\nDQN/PPO 学习动作选择", {
    line: C.amber,
    titleColor: C.amber,
    bodySize: 9.2,
  });
  addNode(
    slide,
    1.0,
    6.0,
    11.25,
    0.75,
    "融合方式",
    "问题进入 RAG，地图产生轨迹样本，RL 策略选择 action，最后由 Qwen 解释证据和算法结果。",
    { fill: "FFF0F6", line: C.green, titleColor: C.green, bodySize: 10 }
  );
}

function visualFrame(inner) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760">
  <defs>
    <filter id="glow"><feGaussianBlur stdDeviation="7" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <marker id="arrow" markerWidth="14" markerHeight="14" refX="10" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,8 L11,4 z" fill="#${C.cyan}"/></marker>
    <linearGradient id="panel" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FFE5F1"/><stop offset="1" stop-color="#FFF7FB"/></linearGradient>
    <linearGradient id="green" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#${C.green}"/><stop offset="1" stop-color="#${C.cyan}"/></linearGradient>
  </defs>
  <rect width="1200" height="760" rx="34" fill="#FFF7FB"/>
  <rect x="18" y="18" width="1164" height="724" rx="30" fill="#FFF0F6" stroke="#EC4899" stroke-opacity=".45" stroke-width="3"/>
  ${inner}
</svg>`;
}

function vText(text, x, y, size = 28, color = C.text, weight = 700) {
  return `<text x="${x}" y="${y}" font-family="Microsoft YaHei, Segoe UI, sans-serif" font-size="${size}" fill="#${color}" font-weight="${weight}">${esc(text)}</text>`;
}

function vNode(x, y, w, h, label, color = C.cyan) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="18" fill="#FFFFFF" stroke="#${color}" stroke-width="3"/>
  ${vText(label, x + 24, y + h / 2 + 10, 30, color, 700)}`;
}

function visualAgentGraphSvg() {
  const lines = [
    `<path d="M600 380 C430 250 290 260 210 170" stroke="#${C.cyan}" stroke-width="6" fill="none" opacity=".75"/>`,
    `<path d="M600 380 C780 250 910 260 1000 165" stroke="#${C.green}" stroke-width="6" fill="none" opacity=".75"/>`,
    `<path d="M600 380 C430 510 300 510 205 610" stroke="#${C.blue}" stroke-width="6" fill="none" opacity=".75"/>`,
    `<path d="M600 380 C780 520 920 515 1005 610" stroke="#${C.amber}" stroke-width="6" fill="none" opacity=".75"/>`,
    `<circle cx="600" cy="380" r="118" fill="#E7F8F0" stroke="#${C.green}" stroke-width="5" filter="url(#glow)"/>`,
    vText("Qwen", 542, 368, 42, C.text, 800),
    vText("Agent Core", 497, 417, 34, C.green, 800),
    vNode(105, 98, 220, 112, "RAG", C.cyan),
    vNode(880, 98, 220, 112, "Policy", C.green),
    vNode(105, 552, 220, 112, "Files", C.blue),
    vNode(880, 552, 220, 112, "Map", C.amber),
    `<circle cx="600" cy="380" r="198" fill="none" stroke="#${C.faint}" stroke-width="3" stroke-dasharray="14 16"/>`,
  ];
  return visualFrame(lines.join(""));
}

function visualFileSandboxSvg() {
  const inner = [
    `<rect x="90" y="105" width="420" height="555" rx="26" fill="#FFFFFF" stroke="#${C.cyan}" stroke-width="3"/>`,
    vText("PROJECT", 132, 165, 30, C.cyan, 800),
    ...["PROJECT_OVERVIEW.md", "backend/agent_loop.py", "frontend/appStore.js", "kb/raw/papers"].map((t, i) =>
      `<rect x="132" y="${210 + i * 75}" width="315" height="48" rx="12" fill="#FDF2F8" stroke="#${i === 0 ? C.green : C.faint}" stroke-width="2"/>${vText(t, 152, 242 + i * 75, 20, i === 0 ? C.green : C.muted, 650)}`
    ),
    `<rect x="585" y="118" width="520" height="360" rx="24" fill="#FDF2F8" stroke="#${C.green}" stroke-width="3"/>`,
    vText("Read + Analyze", 625, 178, 34, C.green, 800),
    `<path d="M632 240 H1030 M632 295 H980 M632 350 H1048 M632 405 H930" stroke="#${C.line}" stroke-width="14" stroke-linecap="round"/>`,
    `<path d="M512 382 C555 382 560 382 585 382" stroke="#${C.cyan}" stroke-width="8" fill="none" marker-end="url(#arrow)"/>`,
    `<path d="M760 565 l82 -62 82 62 v82 c0 52 -164 52 -164 0z" fill="#E7F8F0" stroke="#${C.green}" stroke-width="5"/>`,
    `<rect x="805" y="572" width="74" height="64" rx="10" fill="#${C.green}" opacity=".9"/>`,
    `<path d="M823 572 v-28 c0 -42 38 -58 56 -26" stroke="#${C.green}" stroke-width="14" fill="none" stroke-linecap="round"/>`,
  ];
  return visualFrame(inner.join(""));
}

function visualSandboxLockSvg() {
  const inner = [
    `<path d="M600 95 l365 122 v220 c0 170 -110 270 -365 325 C345 707 235 607 235 437 V217z" fill="#E7F8F0" stroke="#${C.green}" stroke-width="6" filter="url(#glow)"/>`,
    `<rect x="465" y="338" width="270" height="210" rx="24" fill="#FFF7FB" stroke="#${C.cyan}" stroke-width="5"/>`,
    `<path d="M505 338 v-68 c0 -126 190 -126 190 0v68" stroke="#${C.cyan}" stroke-width="28" fill="none" stroke-linecap="round"/>`,
    `<circle cx="600" cy="440" r="28" fill="#${C.green}"/>`,
    `<path d="M600 462 v48" stroke="#${C.green}" stroke-width="16" stroke-linecap="round"/>`,
    vText("Once per scope", 98, 165, 31, C.cyan, 800),
    vText("Server boundary", 835, 165, 31, C.green, 800),
    vText("Hash + backup", 850, 640, 31, C.amber, 800),
    `<path d="M245 175 H95 V475 H245" stroke="#${C.cyan}" stroke-width="4" fill="none" stroke-dasharray="12 12"/>`,
    `<path d="M958 175 H1105 V475 H958" stroke="#${C.green}" stroke-width="4" fill="none" stroke-dasharray="12 12"/>`,
  ];
  return visualFrame(inner.join(""));
}

function visualContextSvg() {
  const cards = Array.from({ length: 10 }, (_, i) => {
    const x = 90 + (i % 5) * 112;
    const y = 110 + Math.floor(i / 5) * 92;
    return `<rect x="${x}" y="${y}" width="82" height="58" rx="12" fill="#FFFFFF" stroke="#${C.line}" stroke-width="2"/>
    <path d="M${x + 16} ${y + 24} h48 M${x + 16} ${y + 39} h35" stroke="#${C.muted}" stroke-width="5" stroke-linecap="round"/>`;
  });
  const inner = [
    ...cards,
    `<path d="M642 210 C710 260 710 390 642 445 L555 520 H430 L555 445 C605 390 605 260 555 210z" fill="#E7F8F0" stroke="#${C.green}" stroke-width="4"/>`,
    vText("compress", 456, 382, 28, C.green, 800),
    `<rect x="735" y="118" width="350" height="165" rx="24" fill="#FFFFFF" stroke="#${C.cyan}" stroke-width="3"/>`,
    vText("Segment Summary", 780, 183, 30, C.cyan, 800),
    `<path d="M780 220 h245 M780 250 h190" stroke="#${C.line}" stroke-width="10" stroke-linecap="round"/>`,
    `<rect x="735" y="365" width="350" height="185" rx="24" fill="#E7F8F0" stroke="#${C.green}" stroke-width="3"/>`,
    vText("Rolling State", 795, 435, 32, C.green, 800),
    `<path d="M780 476 h242 M780 507 h170" stroke="#${C.line}" stroke-width="10" stroke-linecap="round"/>`,
    `<path d="M1085 457 C1130 457 1140 457 1160 457" stroke="#${C.green}" stroke-width="6" marker-end="url(#arrow)"/>`,
  ];
  return visualFrame(inner.join(""));
}

function findGeneratedSample() {
  const candidates = [
    "comfy_20260514_193355_f27167ea.png",
    "comfy_20260514_193121_569c0411.png",
    "comfy_20260514_192953_4028771e.png",
    "comfy_20260514_192942_149f406e.png",
  ].map((name) => path.join(GENERATED_IMAGES_DIR, name));
  return candidates.find((file) => fs.existsSync(file)) || "";
}

function dataUri(file) {
  if (!file || !fs.existsSync(file)) return "";
  const ext = path.extname(file).toLowerCase().replace(".", "") || "png";
  return `data:image/${ext === "jpg" ? "jpeg" : ext};base64,${fs.readFileSync(file).toString("base64")}`;
}

function visualImageGenerationSvg() {
  const sample = dataUri(findGeneratedSample());
  const imageBlock = sample
    ? `<clipPath id="clip"><rect x="80" y="108" width="470" height="470" rx="28"/></clipPath>
       <image href="${sample}" x="80" y="108" width="470" height="470" preserveAspectRatio="xMidYMid slice" clip-path="url(#clip)"/>
       <rect x="80" y="108" width="470" height="470" rx="28" fill="none" stroke="#${C.cyan}" stroke-width="4"/>`
    : `<rect x="80" y="108" width="470" height="470" rx="28" fill="#FFFFFF" stroke="#${C.cyan}" stroke-width="4"/>`;
  const inner = [
    imageBlock,
    vText("ComfyUI sample", 118, 635, 31, C.cyan, 800),
    `<rect x="645" y="115" width="405" height="110" rx="24" fill="#FFFFFF" stroke="#${C.cyan}" stroke-width="3"/>`,
    `<rect x="710" y="318" width="405" height="110" rx="24" fill="#E7F8F0" stroke="#${C.green}" stroke-width="3"/>`,
    `<rect x="645" y="520" width="405" height="110" rx="24" fill="#FFFFFF" stroke="#${C.amber}" stroke-width="3"/>`,
    vText("Prompt Plan", 695, 183, 31, C.cyan, 800),
    vText("Quality Gate", 770, 386, 31, C.green, 800),
    vText("Retry / Best", 700, 588, 31, C.amber, 800),
    `<path d="M848 225 C848 270 913 270 913 318" stroke="#${C.cyan}" stroke-width="7" fill="none" marker-end="url(#arrow)"/>`,
    `<path d="M913 428 C913 475 848 475 848 520" stroke="#${C.green}" stroke-width="7" fill="none" marker-end="url(#arrow)"/>`,
  ];
  return visualFrame(inner.join(""));
}

function visualKnowledgeMapSvg() {
  const route = "M150 575 C260 455 320 505 405 388 S590 248 710 330 S870 420 1028 205";
  const inner = [
    `<path d="M80 145 H1120 M80 290 H1120 M80 435 H1120 M235 80 V680 M410 80 V680 M585 80 V680 M760 80 V680 M935 80 V680" stroke="#${C.faint}" stroke-width="2" opacity=".75"/>`,
    `<path d="${route}" stroke="#${C.cyan}" stroke-width="12" fill="none" stroke-linecap="round"/>`,
    `<path d="${route}" stroke="#ffffff" stroke-width="3" fill="none" stroke-linecap="round" opacity=".55"/>`,
    ...[
      [150, 575, C.green, "S3"],
      [405, 388, C.amber, "DQN"],
      [710, 330, C.green, "RLTS"],
      [1028, 205, C.amber, "PPO"],
    ].map(([x, y, color, label]) => `<circle cx="${x}" cy="${y}" r="22" fill="#${color}" stroke="#FFF7FB" stroke-width="6"/><text x="${x - 24}" y="${y - 34}" font-family="Microsoft YaHei" font-size="24" fill="#${color}" font-weight="800">${label}</text>`),
    `<rect x="94" y="90" width="260" height="92" rx="18" fill="#FFFFFF" stroke="#${C.cyan}" stroke-width="3"/>`,
    vText("Paper RAG", 125, 148, 30, C.cyan, 800),
    `<rect x="820" y="560" width="295" height="92" rx="18" fill="#E7F8F0" stroke="#${C.green}" stroke-width="3"/>`,
    vText("Trajectory Policy", 850, 618, 30, C.green, 800),
  ];
  return visualFrame(inner.join(""));
}

async function writeVisual(name, svg) {
  await sharp(Buffer.from(svg)).png().toFile(assetPath(name));
}

async function ensureVisualAssets() {
  await fsp.mkdir(ASSET_DIR, { recursive: true });
  await writeVisual("visual_agent_graph.png", visualAgentGraphSvg());
  await writeVisual("visual_file_sandbox.png", visualFileSandboxSvg());
  await writeVisual("visual_sandbox_lock.png", visualSandboxLockSvg());
  await writeVisual("visual_context_compression.png", visualContextSvg());
  await writeVisual("visual_image_generation.png", visualImageGenerationSvg());
  await writeVisual("visual_knowledge_map.png", visualKnowledgeMapSvg());
}

function slide8Visual(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "Qwen 到 Agent 的实现流程", "把 Qwen 放在规划层，工具执行和证据回填交给 Agent");
  addVisual(slide, "visual_agent_graph.png", 0.65, 1.45, 6.35, 4.05);
  addMiniNote(slide, "1. Qwen 规划", "判断回答、检索、读文件、生成图片或进入轨迹算法。", 7.55, 1.68, 4.65, C.cyan);
  addMiniNote(slide, "2. 工具注册表执行", "所有工具都有统一 schema、参数校验和结构化 observation。", 7.55, 2.74, 4.65, C.green);
  addMiniNote(slide, "3. 前端呈现", "答案、来源、工具轨迹、图片路径统一回到页面。", 7.55, 3.8, 4.65, C.amber);
  addNode(slide, 7.35, 5.35, 4.9, 0.72, "实现原则", "Qwen 不直接访问系统，本地能力都走可审计工具。", {
    fill: "FFF0F6",
    line: C.faint,
    titleColor: C.text,
    bodySize: 9.2,
  });
}

function slide9Visual(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "读取 / 分析文件功能", "文件能力拆成读取、搜索、索引、修改提案和受保护写入");
  addVisual(slide, "visual_file_sandbox.png", 0.62, 1.45, 6.25, 4.05);
  addMiniNote(slide, "读取路径", "前端选择文件或路径，后端只允许 workspace 内访问。", 7.42, 1.68, 4.75, C.cyan);
  addMiniNote(slide, "分析路径", "文件内容进入 Qwen 上下文，可结合 RAG 和工具轨迹回答。", 7.42, 2.74, 4.75, C.green);
  addMiniNote(slide, "修改路径", "先展示 diff，再校验哈希写入，并保留备份。", 7.42, 3.8, 4.75, C.amber);
  addNode(slide, 7.22, 5.36, 5.0, 0.72, "结果", "用户看到的是“可读、可改、可回滚”的文件操作流程。", {
    fill: "FFF0F6",
    line: C.faint,
    titleColor: C.text,
    bodySize: 9.2,
  });
}

function slide10Visual(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "沙盒权限机制", "同一窗口同一 scope 只确认一次，真正的边界在后端");
  addVisual(slide, "visual_sandbox_lock.png", 0.72, 1.42, 6.0, 4.1);
  addMiniNote(slide, "前端授权缓存", "sessionStorage 记录当前窗口已确认的 scope。", 7.25, 1.6, 4.85, C.cyan);
  addMiniNote(slide, "后端硬边界", "路径穿越、symlink 逃逸、敏感文件都会被拦截。", 7.25, 2.68, 4.85, C.green);
  addMiniNote(slide, "写入保护", "只允许指定写根，写前 hash 校验，写入时自动备份。", 7.25, 3.76, 4.85, C.amber);
  addText(slide, "这样前端体验接近 Codex，但权限不是只靠弹窗。", 7.26, 5.32, 4.8, 0.38, {
    size: 12,
    bold: true,
    color: C.text,
  });
}

function slide11Visual(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "上下文压缩", "旧消息压成摘要和滚动状态，运行时只装配最相关上下文");
  addVisual(slide, "visual_context_compression.png", 0.76, 1.38, 11.85, 3.65);
  addMiniNote(slide, "压缩产物", "摘要段、滚动状态和装配日志都保留。", 1.0, 5.5, 3.35, C.cyan);
  addMiniNote(slide, "装配策略", "优先状态，再取相关旧摘要，最后拼近期消息。", 4.9, 5.5, 3.45, C.green);
  addMiniNote(slide, "降级模式", "数据库不可用时保留近期消息，保证聊天不断路。", 8.85, 5.5, 3.25, C.amber);
}

function slide12Visual(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "图片生成功能实现", "后端已调用生成接口；ComfyUI 离线时走 fallback，在线时返回真实 PNG");
  addVisual(slide, "visual_image_generation.png", 0.72, 1.35, 6.35, 4.35);
  addMiniNote(slide, "本次调用", "已直接调用 backend.image.service.generate_image 生成 3 个示例图卡。", 7.42, 1.58, 4.75, C.cyan);
  addMiniNote(slide, "真实样例", "PPT 中左侧图片来自项目已有 ComfyUI PNG 输出。", 7.42, 2.7, 4.75, C.green);
  addMiniNote(slide, "质量链路", "Prompt 重写、候选生成、质量评分和低分重试组成闭环。", 7.42, 3.82, 4.75, C.amber);
  addNode(slide, 7.22, 5.36, 5.0, 0.72, "落地方式", "前端只拿 /generated-images/... 路径即可渲染。", {
    fill: "FFF0F6",
    line: C.faint,
    titleColor: C.text,
    bodySize: 9.2,
  });
}

function slide13Visual(pptx) {
  const slide = pptx.addSlide();
  addBg(slide);
  addTitle(slide, "知识库、地图与算法融合", "RAG 提供证据，地图产生轨迹样本，RL 策略选择 action");
  addVisual(slide, "visual_knowledge_map.png", 0.68, 1.35, 6.65, 4.35);
  addMiniNote(slide, "知识库", "论文 PDF 切块后进入向量检索，并保留词法检索兜底。", 7.75, 1.54, 4.45, C.cyan);
  addMiniNote(slide, "地图轨迹", "Leaflet/OSRM 生成真实路线，轨迹结果可保存到 MySQL。", 7.75, 2.66, 4.45, C.green);
  addMiniNote(slide, "S3 / MLSimp / RLTS", "三类轨迹压缩方法在同一条 route_geometry 上对照展示。", 7.75, 3.78, 4.45, C.amber);
  addNode(slide, 7.52, 5.34, 4.95, 0.76, "融合闭环", "问题进 RAG，轨迹进地图，策略进 RL，解释统一交给 Qwen。", {
    fill: "FFF0F6",
    line: C.green,
    titleColor: C.green,
    bodySize: 9.2,
  });
}

function wrapLine(text, max = 18) {
  const out = [];
  let cur = "";
  for (const ch of String(text)) {
    cur += ch;
    if (cur.length >= max || ch === "\n") {
      out.push(cur.replace(/\n/g, ""));
      cur = "";
    }
  }
  if (cur) out.push(cur);
  return out;
}

function svgText(text, x, y, size = 28, color = C.text, weight = 400, max = 42) {
  const lines = wrapLine(text, max);
  return `<text x="${x}" y="${y}" font-family="Microsoft YaHei, Arial" font-size="${size}" fill="#${color}" font-weight="${weight}">${lines
    .map((line, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : size * 1.25}">${esc(line)}</tspan>`)
    .join("")}</text>`;
}

function svgNode(x, y, w, h, title, body, stroke = C.line, fill = C.panel, titleColor = C.text) {
  return [
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="#${fill}" stroke="#${stroke}" stroke-width="2"/>`,
    svgText(title, x + 20, y + 38, 24, titleColor, 700, Math.max(8, Math.floor(w / 22))),
    body ? svgText(body, x + 20, y + 78, 18, C.muted, 400, Math.max(9, Math.floor(w / 18))) : "",
  ].join("");
}

function svgArrow(x1, y1, x2, y2, color = C.cyan) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#${color}" stroke-width="4" marker-end="url(#arrow)"/>`;
}

function svgHeader(title, subtitle) {
  return [
    `<rect width="${PX_W}" height="${PX_H}" fill="#${C.bg}"/>`,
    `<rect width="${PX_W}" height="12" fill="#${C.cyan}"/>`,
    svgText(title, 72, 84, 34, C.text, 700),
    subtitle ? svgText(subtitle, 76, 124, 18, C.muted, 400, 88) : "",
    `<line x1="72" y1="148" x2="1526" y2="148" stroke="#${C.faint}" stroke-width="2"/>`,
  ].join("");
}

function svgShell(content) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${PX_W}" height="${PX_H}" viewBox="0 0 ${PX_W} ${PX_H}">
  <defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#${C.cyan}"/></marker></defs>
  ${content}
  </svg>`;
}

function previewSvgs() {
  const s = [];
  s.push(
    svgShell([
      `<rect width="${PX_W}" height="${PX_H}" fill="#${C.bg}"/>`,
      `<rect width="${PX_W}" height="12" fill="#${C.cyan}"/>`,
      svgText("Qwen 驱动的", 92, 230, 42, C.cyan, 700),
      svgText("RL-RAG Agent 流程", 88, 315, 68, C.text, 700),
      svgText("用强化学习训练检索策略，让本地大模型找得更准、答得更有依据", 96, 394, 26, C.muted, 400, 60),
      svgText("当前最优策略  LinUCB", 1080, 230, 30, C.green, 700),
      svgText("Source Hit +34.25pp", 1080, 305, 28, C.green, 700),
      svgText("Topic Hit +9.03pp", 1080, 374, 28, C.cyan, 700),
      svgText("Qwen 生成  →  RL 选策略  →  RAG 找证据", 100, 565, 30, C.text, 700),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("端到端流程图", "从用户问题进入 Qwen，到 RL 检索策略选择，再回到 Qwen 生成答案"),
      svgNode(76, 230, 190, 126, "用户问题", "Vue 前端输入", C.line),
      svgNode(318, 230, 210, 126, "Qwen Planner", "决定 final / tool_use", C.cyan),
      svgNode(580, 230, 210, 126, "工具调用", "search_project_docs", C.line),
      svgNode(842, 230, 220, 126, "RL 检索策略", "选择 retrieval action", C.green, "E7F8F0", C.green),
      svgNode(1120, 230, 205, 126, "论文知识库", "约 200 PDF", C.amber),
      svgNode(1380, 230, 150, 126, "Qwen 答案", "证据生成", C.line),
      svgArrow(266, 292, 318, 292),
      svgArrow(528, 292, 580, 292),
      svgArrow(790, 292, 842, 292, C.green),
      svgArrow(1062, 292, 1120, 292),
      svgArrow(1325, 292, 1380, 292),
      svgNode(140, 560, 1320, 120, "关键闭环", "Qwen 调工具；RL 决定怎么检索；RAG 返回证据；Qwen 再组织最终答案。", C.cyan, "FFF0F6", C.cyan),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("Qwen 的角色", "Qwen 是 Agent 的规划器和最终表达层；RL 不替代 Qwen，只优化检索动作"),
      svgNode(110, 230, 360, 150, "1. 理解问题", "判断是否需要项目知识", C.cyan, C.panel, C.cyan),
      svgNode(620, 230, 360, 150, "2. 发起工具", "输出 JSON tool_use", C.green, C.panel, C.green),
      svgNode(1130, 230, 360, 150, "3. 生成答案", "读取证据并回答", C.amber, C.panel, C.amber),
      svgArrow(470, 305, 620, 305),
      svgArrow(980, 305, 1130, 305),
      svgNode(120, 550, 620, 130, "工具调用格式", '{ "tool_use": { "name": "search_project_docs" } }', C.line, "FFFFFF"),
      svgNode(860, 550, 600, 130, "最终输出要求", "引用检索证据；证据不足时说明限制。", C.line, "FFFFFF"),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("RL 检索策略层", "把一次检索视为离散动作选择：根据 query 特征，选择最可能命中证据的 action"),
      svgNode(90, 220, 320, 370, "Query Features", "关键词 / 任务意图 / 子领域 / 论文偏好 / reward 偏好", C.cyan, C.panel, C.cyan),
      svgNode(610, 290, 340, 240, "Policy", "LinUCB 优先；Dueling DDQN 对照；DPO/PPO/ORPO 实验", C.green, "E7F8F0", C.green),
      svgNode(1120, 220, 360, 370, "Retrieval Actions", "baseline / paper_focus / compression_focus / planning_focus / broad_search ...", C.amber, C.panel, C.amber),
      svgArrow(410, 405, 610, 405),
      svgArrow(950, 405, 1120, 405, C.green),
      svgNode(140, 720, 1320, 82, "动作的含义", "action 改写检索 query、top_k 和主题权重，不改变 Qwen 参数。", C.faint, "FFF0F6"),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("训练逻辑", "用已有题库和本地论文库离线评估每个 action，再训练策略选择器"),
      svgNode(90, 220, 220, 150, "评测题库", "36 个问题", C.line),
      svgNode(380, 220, 240, 150, "动作枚举", "每题 9 个 action", C.line),
      svgNode(690, 220, 240, 150, "奖励计算", "Source/Topic/Recall", C.line),
      svgNode(1000, 220, 250, 150, "策略训练", "PPO/DPO/ORPO/LinUCB/DDQN", C.green, "E7F8F0", C.green),
      svgNode(1320, 220, 190, 150, "上线选择", "LinUCB 优先", C.amber),
      svgArrow(310, 295, 380, 295),
      svgArrow(620, 295, 690, 295),
      svgArrow(930, 295, 1000, 295, C.green),
      svgArrow(1250, 295, 1320, 295),
      svgNode(120, 540, 360, 135, "Source Hit", "是否命中目标论文/来源", C.green, C.panel, C.green),
      svgNode(620, 540, 360, 135, "Topic Hit", "是否覆盖目标主题", C.cyan, C.panel, C.cyan),
      svgNode(1120, 540, 360, 135, "Point Recall", "关键知识点是否召回", C.amber, C.panel, C.amber),
    ].join(""))
  );
  const barSvg = [];
  barSvg.push(svgHeader("效果对比", "相对各自 baseline 的提升：Source Hit 和 Topic Hit 是最能体现 RL 检索收益的指标"));
  barSvg.push(svgText("Source Hit Gain", 340, 210, 24, C.green, 700));
  barSvg.push(svgText("Topic Hit Gain", 920, 210, 24, C.cyan, 700));
  const min = -0.1;
  const max = 0.35;
  methods.forEach((m, i) => {
    const y = 270 + i * 78;
    const sx = 350;
    const tx = 930;
    const bw = 390;
    const z = (x) => ((x - min) / (max - min)) * bw;
    const sourceEnd = sx + z(m.source);
    const topicEnd = tx + z(m.topic);
    const zero1 = sx + z(0);
    const zero2 = tx + z(0);
    barSvg.push(svgText(m.name, 90, y + 8, 22, m.best ? C.green : C.text, m.best ? 700 : 400));
    barSvg.push(`<line x1="${sx}" y1="${y + 12}" x2="${sx + bw}" y2="${y + 12}" stroke="#${C.faint}" stroke-width="13"/>`);
    barSvg.push(`<line x1="${zero1}" y1="${y + 12}" x2="${sourceEnd}" y2="${y + 12}" stroke="#${m.best ? C.green : m.color}" stroke-width="13"/>`);
    barSvg.push(`<line x1="${tx}" y1="${y + 12}" x2="${tx + bw}" y2="${y + 12}" stroke="#${C.faint}" stroke-width="13"/>`);
    barSvg.push(`<line x1="${zero2}" y1="${y + 12}" x2="${topicEnd}" y2="${y + 12}" stroke="#${C.cyan}" stroke-width="13"/>`);
    barSvg.push(svgText(percent(m.source), sx + bw + 20, y + 18, 18, m.source >= 0 ? C.green : C.red));
    barSvg.push(svgText(percent(m.topic), tx + bw + 20, y + 18, 18, m.topic >= 0 ? C.cyan : C.red));
  });
  barSvg.push(svgNode(100, 720, 1400, 90, "结论", "LinUCB 与 Dueling DDQN 在 Source Hit 上提升最大；ORPO 是偏好式目标不适合当前单步检索任务的反例。", C.green, "FFF0F6", C.green));
  s.push(svgShell(barSvg.join("")));
  s.push(
    svgShell([
      svgHeader("是否合理", "从 Qwen 到 RL-RAG 的分工是清楚的：生成归生成，检索策略归策略"),
      svgNode(100, 220, 390, 360, "为什么合理", "检索动作是离散选择；每次问题只需一次 action；奖励可直接衡量。", C.green, C.panel, C.green),
      svgNode(605, 220, 390, 360, "它提升了什么", "更容易命中正确论文；更容易覆盖目标主题；回答更有证据。", C.cyan, C.panel, C.cyan),
      svgNode(1110, 220, 390, 360, "边界在哪里", "没有微调 Qwen 本体；提升来自检索策略与证据质量。", C.amber, C.panel, C.amber),
      `<rect x="170" y="735" width="1260" height="100" rx="12" fill="#FFF0F6" stroke="#${C.green}" stroke-width="2"/>`,
      svgText("推荐上线：Qwen + RAG + LinUCB；", 430, 775, 28, C.text, 700, 40),
      svgText("DDQN 作为神经网络策略对照，PPO/DPO/ORPO 作为实验组。", 330, 812, 26, C.text, 700, 58),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("Qwen 到 Agent 的实现流程", "前端请求进入 Agent 后，由 Qwen 决策是否调用工具，工具结果再回填给 Qwen 生成最终回答"),
      svgNode(80, 235, 190, 125, "前端", "问题、附件、按钮", C.line),
      svgNode(320, 235, 190, 125, "聊天接口", "/api/chat", C.cyan),
      svgNode(560, 235, 210, 125, "上下文", "记忆 + RAG", C.line),
      svgNode(820, 235, 190, 125, "Agent 循环", "Qwen 选工具", C.green, "E7F8F0", C.green),
      svgNode(1060, 235, 190, 125, "工具执行", "校验参数", C.amber),
      svgNode(1300, 235, 190, 125, "结果回填", "工具结果", C.line),
      svgArrow(270, 298, 320, 298),
      svgArrow(510, 298, 560, 298),
      svgArrow(770, 298, 820, 298, C.green),
      svgArrow(1010, 298, 1060, 298, C.green),
      svgArrow(1250, 298, 1300, 298),
      svgNode(130, 555, 360, 130, "Qwen / Ollama", "负责规划和最终回答", C.cyan, C.panel, C.cyan),
      svgNode(620, 555, 360, 130, "工具族", "RAG、文件、图片、轨迹", C.green, C.panel, C.green),
      svgNode(1110, 555, 360, 130, "前端展示", "答案、来源、工具轨迹", C.amber, C.panel, C.amber),
      svgArrow(490, 620, 620, 620, C.cyan),
      svgArrow(980, 620, 1110, 620, C.green),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("读取 / 分析文件功能实现", "文件能力分成读取、搜索、索引、修改提案、受保护写入五个环节"),
      svgNode(95, 220, 300, 150, "前端入口", "上传、读取、编辑提案", C.cyan, C.panel, C.cyan),
      svgNode(465, 220, 300, 150, "路径校验", "只允许 workspace 内", C.green, C.panel, C.green),
      svgNode(835, 220, 300, 150, "读取与检索", "读文件、全文搜索、索引", C.amber, C.panel, C.amber),
      svgNode(1205, 220, 300, 150, "Qwen 分析", "文件进入上下文", C.line),
      svgArrow(395, 295, 465, 295),
      svgArrow(765, 295, 835, 295, C.green),
      svgArrow(1135, 295, 1205, 295, C.amber),
      svgNode(180, 555, 360, 130, "修改提案", "先生成差异，不直接写", C.cyan, "FFFFFF", C.cyan),
      svgNode(620, 555, 360, 130, "写入保护", "哈希校验 + 自动备份", C.green, "FFFFFF", C.green),
      svgNode(1060, 555, 360, 130, "可追踪结果", "路径、差异、备份、哈希", C.amber, "FFFFFF", C.amber),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("沙盒权限机制", "前端同一窗口同一 scope 只确认一次，后端再做不可绕过的路径与敏感文件保护"),
      svgNode(130, 230, 330, 300, "前端授权缓存", "同一窗口同一 scope\n只确认一次", C.cyan, C.panel, C.cyan),
      svgNode(520, 230, 330, 300, "后端安全边界", "禁止越界路径\n拦截敏感文件", C.green, C.panel, C.green),
      svgNode(910, 230, 330, 300, "写入限制", "限定写入目录\n写前校验并备份", C.amber, C.panel, C.amber),
      svgNode(1300, 230, 170, 300, "效果", "体验接近 Codex\n边界落在服务端", C.green, "E7F8F0", C.green),
      svgArrow(460, 382, 520, 382, C.cyan),
      svgArrow(850, 382, 910, 382, C.green),
      svgArrow(1240, 382, 1300, 382, C.amber),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("上下文压缩", "长对话不直接塞满 Qwen，而是把旧消息压成摘要和滚动状态，再按相关性装配"),
      svgNode(90, 250, 190, 130, "原始消息", "对话记录", C.line),
      svgNode(350, 250, 190, 130, "Token 统计", "达到阈值触发压缩", C.cyan),
      svgNode(610, 250, 220, 130, "分段摘要", "旧消息压缩", C.green, "E7F8F0", C.green),
      svgNode(900, 250, 220, 130, "滚动状态", "用户偏好、任务目标、结论", C.green, "E7F8F0", C.green),
      svgNode(1190, 250, 240, 130, "运行时装配", "状态 + 相关 + 近期", C.amber),
      svgArrow(280, 315, 350, 315),
      svgArrow(540, 315, 610, 315, C.green),
      svgArrow(830, 315, 900, 315, C.green),
      svgArrow(1120, 315, 1190, 315, C.amber),
      svgNode(170, 585, 350, 120, "压缩产物", "摘要段、滚动状态、装配日志", C.cyan, C.panel, C.cyan),
      svgNode(625, 585, 350, 120, "选择策略", "保留近期消息，检索相关旧摘要", C.green, C.panel, C.green),
      svgNode(1080, 585, 350, 120, "降级模式", "MySQL 不可用时保留近期", C.amber, C.panel, C.amber),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("图片生成功能实现", "Agent 可识别图片意图，生成结构化计划，调用 ComfyUI，并用质量控制循环筛选结果"),
      svgNode(90, 245, 210, 130, "图片意图", "识别生成请求", C.line),
      svgNode(380, 245, 220, 130, "结构化计划", "正向词 + 负向词", C.cyan),
      svgNode(680, 245, 220, 130, "ComfyUI 执行", "工作流生成图片", C.green, "E7F8F0", C.green),
      svgNode(980, 245, 220, 130, "质量评分", "尺寸、伪影、水印", C.green, "E7F8F0", C.green),
      svgNode(1280, 245, 220, 130, "重试修复", "低分时自动修复", C.amber),
      svgArrow(300, 310, 380, 310),
      svgArrow(600, 310, 680, 310, C.cyan),
      svgArrow(900, 310, 980, 310, C.green),
      svgArrow(1200, 310, 1280, 310, C.amber),
      svgNode(180, 580, 360, 120, "Preset 能力", "产品、海报、Logo、室内等", C.cyan, C.panel, C.cyan),
      svgNode(620, 580, 360, 120, "返回内容", "最终图、候选图、质量报告", C.green, C.panel, C.green),
      svgNode(1060, 580, 360, 120, "前端展示", "图片路径直接渲染", C.amber, C.panel, C.amber),
    ].join(""))
  );
  s.push(
    svgShell([
      svgHeader("知识库、地图与 S3 / MLSimp / RLTS 融合", "RAG 负责论文知识，地图负责轨迹场景，RL 策略负责选择检索和压缩动作"),
      svgNode(115, 225, 380, 300, "知识库 / RAG", "论文 PDF 切块\n向量检索 + 词法兜底\nRL 改写检索动作", C.cyan, C.panel, C.cyan),
      svgNode(610, 225, 380, 300, "地图 / 轨迹", "Leaflet + OSM\nOSRM 生成真实路线\n轨迹存入 MySQL", C.green, C.panel, C.green),
      svgNode(1105, 225, 380, 300, "S3/MLSimp/RLTS", "S3 保骨架\nMLSimp 取代表点\nRLTS 关注路口变化", C.amber, C.panel, C.amber),
      svgNode(230, 625, 1140, 90, "融合方式", "问题进 RAG，地图产轨迹样本，RL 策略选 action，最后由 Qwen 解释证据和算法结果", C.green, "FFF0F6", C.green),
    ].join(""))
  );
  return s;
}

function previewAssetImage(name, x, y, w, h) {
  const file = assetPath(name);
  if (!fs.existsSync(file)) return "";
  const uri = `data:image/png;base64,${fs.readFileSync(file).toString("base64")}`;
  return `<image href="${uri}" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet"/>`;
}

function visualPreview(title, subtitle, asset, notes) {
  const parts = [
    svgHeader(title, subtitle),
    previewAssetImage(asset, 82, 170, 760, 520),
  ];
  notes.forEach((note, i) => {
    const y = 235 + i * 145;
    const color = [C.cyan, C.green, C.amber][i % 3];
    parts.push(svgText(note[0], 930, y, 30, color, 700, 24));
    parts.push(svgText(note[1], 930, y + 48, 22, C.muted, 400, 34));
  });
  return svgShell(parts.join(""));
}

function visualPreviewSvgs() {
  return [
    visualPreview("Qwen 到 Agent 的实现流程", "Qwen 负责规划，Agent 负责工具调度和证据回填", "visual_agent_graph.png", [
      ["Qwen 规划", "判断回答、检索、文件、图片和轨迹动作"],
      ["工具执行", "统一 schema、参数校验、结构化结果"],
      ["前端展示", "答案、来源、工具轨迹和图片路径"],
    ]),
    visualPreview("读取 / 分析文件功能实现", "受控读取、搜索、索引、修改提案和写入保护", "visual_file_sandbox.png", [
      ["读取路径", "文件只能从 workspace 安全边界内进入"],
      ["分析路径", "内容进入 Qwen 上下文并可带来源"],
      ["修改路径", "先看 diff，再 hash 校验和备份"],
    ]),
    visualPreview("沙盒权限机制", "同一窗口同一 scope 只确认一次，后端保留硬边界", "visual_sandbox_lock.png", [
      ["前端缓存", "sessionStorage 记录确认过的 scope"],
      ["后端边界", "路径穿越、敏感文件、越权写入都拦截"],
      ["写入保护", "指定写根、hash 校验、自动备份"],
    ]),
    visualPreview("上下文压缩", "旧消息压缩成摘要和滚动状态，运行时按相关性装配", "visual_context_compression.png", [
      ["压缩产物", "摘要段、滚动状态、装配日志"],
      ["装配策略", "状态优先，再取相关摘要和近期消息"],
      ["降级模式", "数据库不可用时保留近期消息"],
    ]),
    visualPreview("图片生成功能实现", "后端生成接口已调用，ComfyUI 在线时返回真实 PNG", "visual_image_generation.png", [
      ["本次调用", "已调用后端生成 3 个示例图卡"],
      ["真实样例", "使用项目已有 ComfyUI PNG 输出"],
      ["质量链路", "Prompt、候选、评分、重试形成闭环"],
    ]),
    visualPreview("知识库、地图与 S3 / MLSimp / RLTS 融合", "RAG 提供证据，地图产生轨迹样本，RL 策略选择 action", "visual_knowledge_map.png", [
      ["知识库", "论文切块、向量检索、词法兜底"],
      ["地图轨迹", "Leaflet/OSRM 路线保存到 MySQL"],
      ["算法对照", "S3、MLSimp、RLTS 同场比较"],
    ]),
  ];
}

async function renderPreviews() {
  await fsp.mkdir(PREVIEW_DIR, { recursive: true });
  const svgs = previewSvgs();
  visualPreviewSvgs().forEach((svg, i) => {
    svgs[7 + i] = svg;
  });
  const pngs = [];
  for (let i = 0; i < svgs.length; i += 1) {
    const file = path.join(PREVIEW_DIR, `slide_${String(i + 1).padStart(2, "0")}.png`);
    await sharp(Buffer.from(svgs[i])).png().toFile(file);
    pngs.push(file);
  }
  const thumbs = await Promise.all(
    pngs.map(async (file, i) => ({
      input: await sharp(file).resize(640, 360).png().toBuffer(),
      left: (i % 2) * 640,
      top: Math.floor(i / 2) * 360,
    }))
  );
  const montage = path.join(PREVIEW_DIR, "montage.png");
  await sharp({
    create: {
      width: 1280,
      height: Math.ceil(pngs.length / 2) * 360,
      channels: 4,
      background: `#${C.bg}`,
    },
  })
    .composite(thumbs)
    .png()
    .toFile(montage);
  return { pngs, montage };
}

async function inspectDeck(expectedSlides) {
  const buf = await fsp.readFile(PPTX_PATH);
  const zip = await JSZip.loadAsync(buf);
  const slideFiles = Object.keys(zip.files).filter((name) => /^ppt\/slides\/slide\d+\.xml$/.test(name));
  const xml = (await Promise.all(slideFiles.map((name) => zip.files[name].async("string")))).join("\n");
  const rels = Object.keys(zip.files).filter((name) => /^ppt\/slides\/_rels\/slide\d+\.xml\.rels$/.test(name));
  const inspection = {
    pptx: path.relative(ROOT, PPTX_PATH),
    slideCount: slideFiles.length,
    relCount: rels.length,
    bytes: buf.length,
    hasTodoText: /TODO|Lorem|placeholder/i.test(xml),
    hasPinkGreenTheme: xml.includes(C.bg) && xml.includes(C.cyan) && xml.includes(C.green),
    hasRecommendationText: /PPO\/DPO\/ORPO/.test(xml),
    generatedAt: new Date().toISOString(),
  };
  await fsp.writeFile(INSPECTION_PATH, `${JSON.stringify(inspection, null, 2)}\n`, "utf8");
  if (inspection.slideCount !== expectedSlides || inspection.hasTodoText || !inspection.hasPinkGreenTheme || !inspection.hasRecommendationText) {
    throw new Error(`Deck inspection failed: ${JSON.stringify(inspection)}`);
  }
  return inspection;
}

async function main() {
  await fsp.mkdir(OUT_DIR, { recursive: true });
  await ensureVisualAssets();
  const pptx = new pptxgen();
  pptx.defineLayout({ name: "QWEN_WIDE", width: W, height: H });
  pptx.layout = "QWEN_WIDE";
  pptx.author = "Codex";
  pptx.company = "TRL Agent";
  pptx.subject = "Qwen + RL-RAG Agent process";
  pptx.title = "Qwen 驱动的 RL-RAG Agent 流程";
  pptx.lang = "zh-CN";
  pptx.theme = {
    headFontFace: "Microsoft YaHei",
    bodyFontFace: "Microsoft YaHei",
    lang: "zh-CN",
  };

  [slide1, slide2, slide3, slide4, slide5, slide6, slide7, slide8Visual, slide9Visual, slide10Visual, slide11Visual, slide12Visual, slide13Visual].forEach((fn) => fn(pptx));
  await pptx.writeFile({ fileName: PPTX_PATH });
  const previews = await renderPreviews();
  const inspection = await inspectDeck(13);
  console.log(JSON.stringify({ pptx: PPTX_PATH, previews, inspection }, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
