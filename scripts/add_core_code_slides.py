from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from scripts.add_linucb_explanation_slides import (
    A_NS,
    APP_NS,
    CT_NS,
    PKG_REL_NS,
    P_NS,
    R_NS,
    SLIDE_H_IN,
    SLIDE_W_IN,
    THEME,
    add,
    add_xfrm,
    emu,
    no_line,
    qn,
    rgb,
    shape,
    slide_relationship_xml,
    solid_fill,
    textbox,
)


PPTX_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("PPTX_PATH", r"C:\Users\Lenovo\Desktop\新建文件夹\软件过程管理.pptx"))
PREVIEW_DIR = Path(os.environ.get("PREVIEW_DIR", "outputs/ppt_previews/core_code_slides"))
PNG_PREVIEW_DIR = PREVIEW_DIR / "png"

CODE_BG = "12312A"
CODE_FG = "EAFBF0"
CODE_DIM = "9ED9B8"


def text_body_font(
    parent: ET.Element,
    text: str,
    font_size: int,
    color: str,
    *,
    font_family: str = "Microsoft YaHei",
    bold: bool = False,
    align: str = "l",
    anchor: str = "t",
    margin: int = 50000,
    autofit: bool = True,
) -> None:
    tx = add(parent, "p:txBody")
    body_pr = add(
        tx,
        "a:bodyPr",
        {
            "wrap": "square",
            "rtlCol": "0",
            "anchor": anchor,
            "lIns": str(margin),
            "rIns": str(margin),
            "tIns": str(margin),
            "bIns": str(margin),
        },
    )
    if autofit:
        add(body_pr, "a:normAutofit", {"fontScale": "76000", "lnSpcReduction": "18000"})
    add(tx, "a:lstStyle")
    for raw_line in text.splitlines() or [""]:
        p = add(tx, "a:p")
        ppr = add(p, "a:pPr", {"algn": {"l": "l", "c": "ctr", "r": "r"}.get(align, align)})
        ln_spc = add(ppr, "a:lnSpc")
        add(ln_spc, "a:spcPct", {"val": "88000"})
        r = add(p, "a:r")
        rpr_attrs = {"lang": "zh-CN", "sz": str(font_size * 100)}
        if bold:
            rpr_attrs["b"] = "1"
        rpr = add(r, "a:rPr", rpr_attrs)
        solid_fill(rpr, color)
        add(rpr, "a:latin", {"typeface": font_family})
        add(rpr, "a:ea", {"typeface": font_family})
        add(rpr, "a:cs", {"typeface": font_family})
        t = add(r, "a:t")
        t.text = raw_line


def text_shape_font(
    sp_tree: ET.Element,
    sid: int,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    font_size: int = 12,
    color: str = "17312B",
    font_family: str = "Microsoft YaHei",
    bold: bool = False,
    align: str = "l",
    anchor: str = "t",
) -> int:
    sp = add(sp_tree, "p:sp")
    nv = add(sp, "p:nvSpPr")
    add(nv, "p:cNvPr", {"id": str(sid), "name": name})
    add(nv, "p:cNvSpPr", {"txBox": "1"})
    add(nv, "p:nvPr")
    sp_pr = add(sp, "p:spPr")
    add_xfrm(sp_pr, x, y, w, h)
    geom = add(sp_pr, "a:prstGeom", {"prst": "rect"})
    add(geom, "a:avLst")
    add(sp_pr, "a:noFill")
    no_line(sp_pr)
    text_body_font(
        sp,
        text,
        font_size,
        color,
        font_family=font_family,
        bold=bold,
        align=align,
        anchor=anchor,
    )
    return sid + 1


def build_code_slide(spec: dict[str, Any]) -> bytes:
    sld = ET.Element(qn("p:sld"))
    c_sld = add(sld, "p:cSld")
    sp_tree = add(c_sld, "p:spTree")
    nv_grp = add(sp_tree, "p:nvGrpSpPr")
    add(nv_grp, "p:cNvPr", {"id": "1", "name": ""})
    add(nv_grp, "p:cNvGrpSpPr")
    add(nv_grp, "p:nvPr")
    grp_pr = add(sp_tree, "p:grpSpPr")
    xfrm = add(grp_pr, "a:xfrm")
    add(xfrm, "a:off", {"x": "0", "y": "0"})
    add(xfrm, "a:ext", {"cx": "0", "cy": "0"})
    add(xfrm, "a:chOff", {"x": "0", "y": "0"})
    add(xfrm, "a:chExt", {"cx": "0", "cy": "0"})

    sid = 2
    sid = shape(sp_tree, sid, "background", 0, 0, SLIDE_W_IN, SLIDE_H_IN, fill=THEME["bg"], prst="rect")
    sid = shape(sp_tree, sid, "top-green-ribbon", 0, 0, SLIDE_W_IN, 0.16, fill=THEME["green"], prst="rect")
    sid = shape(sp_tree, sid, "top-pink-ribbon", 0, 0.16, SLIDE_W_IN, 0.07, fill=THEME["pink"], prst="rect")
    sid = textbox(sp_tree, sid, "slide-title", 0.58, 0.38, 8.9, 0.58, spec["title"], font_size=25, color=THEME["ink"], bold=True, align="l")
    sid = textbox(sp_tree, sid, "slide-subtitle", 0.62, 0.95, 9.5, 0.40, spec["subtitle"], font_size=11, color=THEME["muted"], align="l")
    sid = shape(sp_tree, sid, "code-tag", 10.25, 0.46, 2.45, 0.40, spec["tag"], fill=THEME["green2"], outline=THEME["green"], font_size=11, color=THEME["line"], bold=True)

    sid = shape(sp_tree, sid, "code-panel", 0.65, 1.55, 8.15, 5.35, fill=CODE_BG, outline=THEME["green"], prst="roundRect")
    sid = shape(sp_tree, sid, "code-accent", 0.65, 1.55, 0.12, 5.35, fill=THEME["pink"], prst="rect")
    sid = text_shape_font(
        sp_tree,
        sid,
        "code-text",
        0.86,
        1.72,
        7.73,
        5.02,
        spec["code"],
        font_size=10,
        color=CODE_FG,
        font_family="Consolas",
        align="l",
        anchor="t",
    )

    sid = shape(sp_tree, sid, "explain-panel", 9.06, 1.55, 3.58, 5.35, fill="FFFFFF", outline=THEME["green"], prst="roundRect")
    sid = shape(sp_tree, sid, "explain-title", 9.27, 1.78, 3.12, 0.42, "这段代码在做什么", fill=THEME["green"], outline=THEME["line"], font_size=13, color="FFFFFF", bold=True)
    sid = text_shape_font(
        sp_tree,
        sid,
        "explain-text",
        9.28,
        2.32,
        3.10,
        3.55,
        "\n".join(f"{idx}. {line}" for idx, line in enumerate(spec["explain"], 1)),
        font_size=13,
        color=THEME["ink"],
        font_family="Microsoft YaHei",
        align="l",
        anchor="t",
    )
    sid = shape(sp_tree, sid, "source-label", 9.28, 6.05, 3.10, 0.48, spec["source"], fill=THEME["pink2"], outline=THEME["pink"], font_size=9, color=THEME["ink"], bold=True)
    sid = textbox(sp_tree, sid, "footer", 0.70, 6.98, 11.8, 0.28, "说明：本页只保留答辩需要讲清楚的核心代码，完整实现见项目源码。", font_size=9, color=THEME["muted"], align="c")

    clr = add(sld, "p:clrMapOvr")
    add(clr, "a:masterClrMapping")
    return ET.tostring(sld, encoding="utf-8", xml_declaration=True)


def get_specs() -> list[dict[str, Any]]:
    return [
        {
            "anchor": "训练逻辑",
            "title": "核心代码：强化学习训练参数",
            "subtitle": "PPO/DPO/ORPO/LinUCB/DDQN 共用同一批检索样本，reward 用 Source Hit、Topic Hit、Point Recall 计算。",
            "tag": "RL 训练",
            "source": "backend/retrieval_rl_env.py · scripts/train_retrieval_policy_ppo.py",
            "code": """# backend/retrieval_rl_env.py
ACTIONS = (
    RetrievalAction("baseline", top_k=4),
    RetrievalAction("rl_focus", top_k=4,
        query_suffixes=("reinforcement learning policy optimization",)),
    RetrievalAction("trajectory_focus", top_k=4,
        query_suffixes=("trajectory planning trajectory optimization",)),
    RetrievalAction("paper_focus", top_k=5,
        query_suffixes=("paper method benchmark ablation",)),
)

# scripts/train_retrieval_policy_ppo.py
parser.add_argument("--epochs", type=int, default=0)  # 本项目设为 60
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--source-weight", type=float, default=0.5)
parser.add_argument("--topic-weight", type=float, default=0.3)
parser.add_argument("--point-weight", type=float, default=0.2)

reward = 0.5 * source_hit + 0.3 * topic_hit + 0.2 * point_recall""",
            "explain": [
                "一个问题就是一个检索 episode。",
                "动作不是直接生成答案，而是选择检索方式。",
                "Source Hit 权重最高，先保证命中正确论文。",
                "60 Epoch 的 PPO 与 LinUCB、DDQN 共用同一奖励口径。",
            ],
        },
        {
            "anchor": "Qwen 到 Agent 的实现流程",
            "title": "核心代码：Qwen 到 Agent 主链路",
            "subtitle": "前端请求进入 /api/chat 后，后端先装配上下文，再决定走 Agent 工具链或普通 Qwen 回答。",
            "tag": "Qwen 后端",
            "source": "backend/llm_service.py · backend/app.py",
            "code": """# backend/llm_service.py
def run_messages(messages, query="", context_block="", model_id=None):
    option = resolve_model_option(model_id)
    backend = str(option.get("backend") or get_llm_backend())
    model_value = str(option.get("model") or "")

    if backend == "hf":
        return _chat_hf(messages, model_value)

    return _chat_ollama(messages, model_value or None)

# backend/app.py
def chat_payload(query, session_id, top_k, attachment_text="", model_id=None):
    session = get_session(session_id)
    ConversationCompressor(model_id=model_id).maybe_compress(session["id"])
    history = ContextAssembler().assemble(session["id"], query)["messages"]

    if agent_enabled() and not attachment_text:
        agent_result = agent_chat(query, history=history, top_k=top_k)
        return {"answer": agent_result["answer"],
                "toolTraces": agent_result.get("tool_traces", [])}

    return {"answer": chat_reply(query, history=history,
                                 context_block=attachment_text)}""",
            "explain": [
                "run_messages 是所有 Qwen 调用的统一入口。",
                "chat_payload 串起会话、压缩上下文和 Agent。",
                "Agent 开启时，Qwen 先规划是否调用工具。",
                "没有工具需求时，直接走普通 Qwen 对话。",
            ],
        },
        {
            "anchor": "读取 / 分析文件功能",
            "title": "核心代码：对话框读取文字与文件上下文",
            "subtitle": "用户输入框文字、附件名和附件正文都会进入同一个 /api/chat payload，后端再把文件内容作为隐藏上下文交给模型。",
            "tag": "文件读取",
            "source": "frontend/src/stores/appStore.js · backend/app.py",
            "code": """// frontend/src/stores/appStore.js
export async function sendCurrentMessage() {
  const query = appState.queryInput.trim()
  if (!query || appState.sending) return

  const payload = {
    query,
    session_id: appState.sessionId,
    top_k: appState.topK,
    model_id: appState.selectedModelId,
    attachment_name: appState.composerAttachment?.name || "",
    attachment_text: appState.composerAttachment?.text || "",
  }

  const data = await api.chat(payload)
  upsertSession({ ...data.session, messages: data.history || [] })
}

# backend/app.py
if attachment_text.strip():
    label = attachment_name.strip() or "attached_file"
    visible_query = f"{query}\\n\\n[Attached file: {label}]"
    file_context = f"\\n\\n[Attached file: {label}]\\n{attachment_text.strip()}" """,
            "explain": [
                "v-model 读取聊天框文字，sendCurrentMessage 统一提交。",
                "附件正文不显示成一大段，而是作为 file_context。",
                "Qwen 能分析文件内容，聊天记录只保留可读标签。",
                "这就是读取/分析本地文件的前后端连接点。",
            ],
        },
        {
            "anchor": "沙盒权限机制",
            "title": "核心代码：沙盒授权与路径硬边界",
            "subtitle": "前端做同一窗口同一 scope 的一次性确认，后端用真实路径解析防止越权读取或写入。",
            "tag": "沙盒权限",
            "source": "frontend/src/stores/appStore.js · backend/workspace_security.py",
            "code": """// frontend/src/stores/appStore.js
async function ensureSandboxApproval(scope, description) {
  if (appState.sandboxApprovals[scope]) return true

  const ok = window.confirm(
    `${description}\\n\\nAllow this operation for this window?`
  )
  if (!ok) return false

  appState.sandboxApprovals[scope] = true
  saveSandboxApprovals(appState.sandboxApprovals)
  return true
}

# backend/workspace_security.py
def resolve_safe_path(relative_path: str = ".") -> Path:
    root = workspace_root()
    candidate = Path(relative_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.expanduser().resolve()

    if not (resolved == root or root in resolved.parents):
        raise ValueError(f"Path is outside workspace root: {relative_path}")
    return resolved""",
            "explain": [
                "前端缓存授权，避免每次同类命令都重复弹窗。",
                "scope 是权限粒度，例如 read_file 或 write_file。",
                "后端不信任前端，所有路径都重新 resolve。",
                "../ 和符号链接逃逸都会被挡在 workspace 外。",
            ],
        },
        {
            "anchor": "上下文压缩",
            "title": "核心代码：上下文压缩与重组",
            "subtitle": "旧消息先由 Qwen 压成结构化摘要，当前轮再按 rolling state、相关摘要、近期消息重新装配。",
            "tag": "上下文压缩",
            "source": "backend/conversation_compressor.py · backend/context_assembler.py",
            "code": """# backend/conversation_compressor.py
class ConversationCompressor:
    def maybe_compress(self, conversation_id: str):
        if not context_compression_enabled() or not self.repository.mysql_available():
            return {"compressed": False, "reason": "disabled_or_no_mysql"}
        return self._compress(conversation_id)

    def _summarize_segment(self, segment):
        transcript = "\\n".join(
            f"[message_id={m['id']} role={m['role']}]\\n{m.get('content', '')}"
            for m in segment
        )
        raw = run_messages([
            {"role": "system", "content": COMPRESSION_PROMPT},
            {"role": "user", "content": transcript},
        ], query="Compress this conversation segment.",
           model_id=self.model_id)
        return _extract_json(raw)

# backend/context_assembler.py
relevant = self.retriever.retrieve(current_user_message, summaries, messages)
assembled.append({"role": "system",
                  "content": "[Conversation rolling state]\\n" + json.dumps(state)})""",
            "explain": [
                "压缩发生在历史变长之后，不影响当前问题。",
                "摘要由 Qwen 生成，再保存到数据库。",
                "装配时只拿相关摘要，避免上下文无限膨胀。",
                "数据库不可用时会降级为最近消息裁剪。",
            ],
        },
        {
            "anchor": "图片生成功能实现",
            "title": "核心代码：图片生成工具链",
            "subtitle": "Agent 先判断用户是否真的要生成图片，再调用高级生成工具：Prompt 改写、候选图生成、质量评分、择优返回。",
            "tag": "图片生成",
            "source": "backend/agent_loop.py · backend/image_quality.py",
            "code": """# backend/agent_loop.py
def should_generate_image(query: str) -> bool:
    return bool(IMAGE_ACTION_RE.search(query or "")) \
        and not bool(IMAGE_DISCUSSION_RE.search(query or ""))

if should_generate_image(query):
    tool_input = {
        "prompt": query,
        "quality_mode": "balanced",
        "allow_retry": True,
    }
    tool_output = execute_tool("generate_image_advanced", tool_input)
    return {"answer": "已生成图片。", "tool_traces": [tool_output]}

# backend/image_quality.py
class ImageGenerationQualityController:
    def generate(self, prompt, quality_mode="balanced", allow_retry=True, **kw):
        plan = self.rewriter.rewrite(prompt, quality_mode=quality_mode)
        artifacts = self.runner.generate(plan)
        reports = [self.critic.evaluate_image(a, plan) for a in artifacts]
        best_artifact, best_report = sorted(
            zip(artifacts, reports),
            key=lambda item: item[1].score,
            reverse=True,
        )[0]
        return {"final_image": best_artifact, "quality_report": best_report.to_dict()}""",
            "explain": [
                "should_generate_image 防止把普通讨论误判成生图请求。",
                "generate_image_advanced 是 Agent 可调用工具。",
                "质量链路会先改写 prompt，再批量生成候选图。",
                "最终返回分数最高的图片和质量报告。",
            ],
        },
        {
            "anchor": "知识库、地图与算法融合",
            "title": "核心代码：知识库检索与线上 RL 策略",
            "subtitle": "Agent 的 search_project_docs 工具不是固定检索，而是先让训练好的策略选择 retrieval action，再查本地论文知识库。",
            "tag": "RAG + RL",
            "source": "backend/retrieval_policy.py · backend/tools.py",
            "code": """# backend/retrieval_policy.py
def choose_retrieval_action(query: str, requested_top_k: int = 4):
    if not retrieval_policy_enabled():
        action = ACTIONS[0]
        return {"algorithm": "baseline",
                "retrieval_query": compose_retrieval_query(query, action),
                "top_k": requested_top_k}

    for path in _candidate_policy_paths():
        loaded = _load_policy(str(path))
        if not loaded:
            continue
        algorithm, policy = loaded
        state = features_for_query(query)
        action_index, raw_scores = policy.choose(state)
        action = ACTIONS[action_index]
        return {"algorithm": algorithm, "action": action.name,
                "retrieval_query": compose_retrieval_query(query, action),
                "top_k": action.top_k, "scores": raw_scores}

# backend/tools.py
def search_project_docs(query: str, top_k: int = 4):
    policy = choose_retrieval_action(query, requested_top_k=top_k)
    results = search_knowledge(policy["retrieval_query"], top_k=policy["top_k"])
    return {"policy": policy, "results": results, "count": len(results)}""",
            "explain": [
                "线上先尝试加载 LinUCB/DDQN/DPO/ORPO/PPO 策略。",
                "策略输入是 query 特征，输出是检索动作。",
                "检索动作会改写 query 或调整 top_k。",
                "返回 policy 字段，前端可以解释 RL 做了什么。",
            ],
        },
        {
            "anchor": "知识库、地图与算法融合",
            "title": "核心代码：前端地图 API 与轨迹生成",
            "subtitle": "地图页用 OSRM 获取真实道路路线，再构造 DQN 决策点、PPO 阶段和压缩算法展示数据。",
            "tag": "地图 API",
            "source": "frontend/src/api.js · frontend/src/views/PathDemoView.vue",
            "code": """// frontend/src/api.js
export const api = {
  trajectories: (limit = 20) =>
    fetchJson(`/api/trajectories?limit=${encodeURIComponent(limit)}`),
  saveTrajectory: (payload) =>
    fetchJson("/api/trajectories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
}

// frontend/src/views/PathDemoView.vue
async function fetchRoadRoute() {
  const { start, end } = effectiveEndpoints.value
  const url = `${OSRM_URL}/${start[1]},${start[0]};${end[1]},${end[0]}`
    + "?overview=full&geometries=geojson&steps=true"
  const payload = await (await fetch(url, { cache: "no-store" })).json()
  return payload.routes?.[0]
}

async function generateTrajectory() {
  const route = await fetchRoadRoute()
  routeGeometry.value = route.geometry.coordinates.map(([lng, lat]) => [lat, lng])
  dqnDecisionPoints.value = buildDqnDecisionPoints(route.legs?.[0]?.steps || [])
  ppoStages.value = buildPpoStages(route.legs?.[0]?.steps || [])
  buildCompressionStrategies()
  await saveCurrentTrajectory()
}""",
            "explain": [
                "OSRM 提供真实道路路线，不是前端随手画线。",
                "DQN 展示离散转向决策点。",
                "PPO 展示分阶段路径策略。",
                "生成结果会保存到后端数据库，方便复现实验。",
            ],
        },
        {
            "anchor": "知识库、地图与算法融合",
            "title": "核心代码：S3/RLTS/MLsimp 轨迹压缩",
            "subtitle": "同一条 route_geometry 上并列构造三类压缩结果，用于地图页对照展示。",
            "tag": "轨迹压缩",
            "source": "frontend/src/views/PathDemoView.vue",
            "code": """function buildCompressionStrategies() {
  const total = routeGeometry.value.length
  if (!total) return

  const turnIndices = dqnDecisionPoints.value.map((item) => item.routeIndex)
  const uniqueSorted = (values) =>
    Array.from(new Set(values.filter((v) => v >= 0 && v < total)))
      .sort((a, b) => a - b)

  const s3 = uniqueSorted([
    0, total - 1,
    ...turnIndices,
    ...Array.from({ length: 8 }, (_, i) => Math.round((i * (total - 1)) / 7)),
  ])

  const rlts = uniqueSorted([
    0, total - 1,
    ...turnIndices.flatMap((i) => [i - 1, i, i + 1]),
    ...Array.from({ length: 10 }, (_, i) => Math.round((i * (total - 1)) / 9)),
  ])

  const mlsimp = uniqueSorted([
    0, total - 1,
    ...turnIndices.filter((_, i) => i % 2 === 0),
    ...Array.from({ length: 6 }, (_, i) => Math.round((i * (total - 1)) / 5)),
  ])
}""",
            "explain": [
                "三种算法使用同一条路线，保证对比公平。",
                "S3 保留端点、转向点和均匀骨架点。",
                "RLTS 更重视路口附近的局部行为变化。",
                "MLsimp 更偏端点和代表性长段变化点。",
            ],
        },
    ]


def slide_target_to_path(target: str) -> str:
    return target[1:] if target.startswith("/") else f"ppt/{target}"


def parse_ordered_slides(entries: dict[str, bytes]) -> list[dict[str, Any]]:
    ns = {"p": P_NS, "a": A_NS, "r": R_NS}
    pres = ET.fromstring(entries["ppt/presentation.xml"])
    rels = ET.fromstring(entries["ppt/_rels/presentation.xml.rels"])
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    records: list[dict[str, Any]] = []
    for slide_id in pres.find("p:sldIdLst", ns).findall("p:sldId", ns):
        rid = slide_id.attrib.get(f"{{{R_NS}}}id")
        target = relmap.get(rid, "")
        path = slide_target_to_path(target)
        title = ""
        num = None
        match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", path)
        if match:
            num = int(match.group(1))
        if path in entries:
            root = ET.fromstring(entries[path])
            texts = [t.text.strip() for t in root.findall(".//a:t", ns) if t.text and t.text.strip()]
            title = texts[0] if texts else ""
        records.append({"element": slide_id, "rid": rid, "target": target, "path": path, "title": title, "num": num})
    return records


def find_layout_target(entries: dict[str, bytes], preferred_slide_num: int = 57) -> str:
    rel_name = f"ppt/slides/_rels/slide{preferred_slide_num}.xml.rels"
    layout_target = "../slideLayouts/slideLayout1.xml"
    if rel_name in entries:
        rel_root = ET.fromstring(entries[rel_name])
        for rel in rel_root:
            if rel.attrib.get("Type") == f"{R_NS}/slideLayout":
                return rel.attrib.get("Target", layout_target)
    return layout_target


def next_numbers(entries: dict[str, bytes], pres_rels: ET.Element, sld_ids: list[ET.Element]) -> tuple[int, int, int]:
    slide_nums = []
    for name in entries:
        match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
        if match:
            slide_nums.append(int(match.group(1)))
    rid_nums = [
        int(rel.attrib["Id"][3:])
        for rel in pres_rels
        if rel.attrib.get("Id", "").startswith("rId") and rel.attrib["Id"][3:].isdigit()
    ]
    max_slide_id = max(int(el.attrib["id"]) for el in sld_ids)
    return max(slide_nums) + 1, max(rid_nums) + 1, max_slide_id + 1


def existing_title_map(entries: dict[str, bytes]) -> dict[str, int]:
    return {record["title"]: int(record["num"]) for record in parse_ordered_slides(entries) if record["title"] and record["num"]}


def add_content_type(content_types: ET.Element, slide_num: int) -> None:
    ET.SubElement(
        content_types,
        f"{{{CT_NS}}}Override",
        {
            "PartName": f"/ppt/slides/slide{slide_num}.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
        },
    )


def render_png_preview(spec: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1920, 1080
    img = Image.new("RGB", (width, height), f"#{THEME['bg']}")
    draw = ImageDraw.Draw(img)

    def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
        candidates = []
        if mono:
            candidates.extend([r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\CascadiaMono.ttf"])
        if bold:
            candidates.extend([r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"])
        candidates.extend([r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\arial.ttf"])
        for candidate in candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default()

    draw.rectangle([0, 0, width, 23], fill=f"#{THEME['green']}")
    draw.rectangle([0, 23, width, 33], fill=f"#{THEME['pink']}")
    draw.text((88, 55), spec["title"], fill=f"#{THEME['ink']}", font=font(40, bold=True))
    draw.text((90, 132), spec["subtitle"], fill=f"#{THEME['muted']}", font=font(18))
    draw.rounded_rectangle([1475, 66, 1818, 123], radius=24, fill=f"#{THEME['green2']}", outline=f"#{THEME['green']}", width=3)
    draw.text((1515, 82), spec["tag"], fill=f"#{THEME['line']}", font=font(20, bold=True))

    code_box = [94, 223, 1267, 994]
    draw.rounded_rectangle(code_box, radius=28, fill=f"#{CODE_BG}", outline=f"#{THEME['green']}", width=4)
    draw.rectangle([94, 223, 112, 994], fill=f"#{THEME['pink']}")
    code_font = font(20, mono=True)
    y = 248
    for line in spec["code"].splitlines():
        draw.text((132, y), line[:96], fill=f"#{CODE_FG}", font=code_font)
        y += 28
        if y > 965:
            break

    exp_box = [1305, 223, 1818, 994]
    draw.rounded_rectangle(exp_box, radius=28, fill="#FFFFFF", outline=f"#{THEME['green']}", width=4)
    draw.rounded_rectangle([1337, 256, 1780, 315], radius=24, fill=f"#{THEME['green']}", outline=f"#{THEME['line']}", width=2)
    draw.text((1392, 272), "这段代码在做什么", fill="#FFFFFF", font=font(22, bold=True))
    y = 350
    body_font = font(23)
    for idx, line in enumerate(spec["explain"], 1):
        wrapped = textwrap.wrap(f"{idx}. {line}", width=21, break_long_words=False)
        for part in wrapped:
            draw.text((1345, y), part, fill=f"#{THEME['ink']}", font=body_font)
            y += 35
        y += 12
    draw.rounded_rectangle([1337, 872, 1780, 941], radius=20, fill=f"#{THEME['pink2']}", outline=f"#{THEME['pink']}", width=3)
    source_lines = textwrap.wrap(spec["source"], width=35, break_long_words=False)
    sy = 884
    for line in source_lines[:2]:
        draw.text((1360, sy), line, fill=f"#{THEME['ink']}", font=font(15, bold=True))
        sy += 24

    draw.text((548, 1016), "说明：本页只保留答辩需要讲清楚的核心代码，完整实现见项目源码。", fill=f"#{THEME['muted']}", font=font(17))
    img.save(path)


def modify_pptx() -> tuple[Path, Path, list[Path]]:
    if not PPTX_PATH.exists():
        raise FileNotFoundError(PPTX_PATH)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    PNG_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup = PPTX_PATH.with_name(f"{PPTX_PATH.stem}_backup_before_core_code_slides_{timestamp}{PPTX_PATH.suffix}")
    shutil.copy2(PPTX_PATH, backup)

    with zipfile.ZipFile(PPTX_PATH, "r") as zin:
        entries = {name: zin.read(name) for name in zin.namelist()}

    pres = ET.fromstring(entries["ppt/presentation.xml"])
    pres_rels = ET.fromstring(entries["ppt/_rels/presentation.xml.rels"])
    content_types = ET.fromstring(entries["[Content_Types].xml"])
    sld_id_lst = pres.find(f"{{{P_NS}}}sldIdLst")
    original_sld_ids = list(sld_id_lst)
    records = parse_ordered_slides(entries)
    title_to_num = existing_title_map(entries)
    specs = get_specs()
    layout_target = find_layout_target(entries)
    next_slide_num, next_rid, next_slide_id = next_numbers(entries, pres_rels, original_sld_ids)

    preview_paths: list[Path] = []
    missing_specs = [spec for spec in specs if spec["title"] not in title_to_num]

    # Replace existing code slides when rerun, so the operation is idempotent.
    for spec in specs:
        if spec["title"] in title_to_num:
            slide_num = title_to_num[spec["title"]]
            entries[f"ppt/slides/slide{slide_num}.xml"] = build_code_slide(spec)
            entries[f"ppt/slides/_rels/slide{slide_num}.xml.rels"] = slide_relationship_xml(layout_target)
            preview = PNG_PREVIEW_DIR / f"slide{slide_num}_{re.sub(r'[^A-Za-z0-9]+', '_', spec['title'])[:36]}.png"
            render_png_preview(spec, preview)
            preview_paths.append(preview)

    if missing_specs:
        specs_by_anchor: dict[str, list[dict[str, Any]]] = {}
        for spec in missing_specs:
            specs_by_anchor.setdefault(spec["anchor"], []).append(spec)

        children_by_rid = {record["rid"]: record for record in records}
        new_children: list[ET.Element] = []
        inserted_titles: set[str] = set()

        def create_slide(spec: dict[str, Any]) -> ET.Element:
            nonlocal next_slide_num, next_rid, next_slide_id
            slide_num = next_slide_num
            rid = f"rId{next_rid}"
            slide_id = next_slide_id
            next_slide_num += 1
            next_rid += 1
            next_slide_id += 1

            entries[f"ppt/slides/slide{slide_num}.xml"] = build_code_slide(spec)
            entries[f"ppt/slides/_rels/slide{slide_num}.xml.rels"] = slide_relationship_xml(layout_target)
            ET.SubElement(
                pres_rels,
                f"{{{PKG_REL_NS}}}Relationship",
                {
                    "Id": rid,
                    "Type": f"{R_NS}/slide",
                    "Target": f"slides/slide{slide_num}.xml",
                },
            )
            add_content_type(content_types, slide_num)
            preview = PNG_PREVIEW_DIR / f"slide{slide_num}_{re.sub(r'[^A-Za-z0-9]+', '_', spec['title'])[:36]}.png"
            render_png_preview(spec, preview)
            preview_paths.append(preview)
            new_sld = ET.Element(f"{{{P_NS}}}sldId", {"id": str(slide_id)})
            new_sld.set(f"{{{R_NS}}}id", rid)
            inserted_titles.add(spec["title"])
            return new_sld

        for child in original_sld_ids:
            new_children.append(child)
            child_rid = child.attrib.get(f"{{{R_NS}}}id")
            record = children_by_rid.get(child_rid)
            if not record:
                continue
            for anchor, anchor_specs in specs_by_anchor.items():
                if anchor and anchor in record["title"]:
                    for spec in anchor_specs:
                        if spec["title"] not in inserted_titles:
                            new_children.append(create_slide(spec))

        # If the corresponding feature page is missing, append the code page at the end.
        for spec in missing_specs:
            if spec["title"] not in inserted_titles:
                new_children.append(create_slide(spec))

        for child in list(sld_id_lst):
            sld_id_lst.remove(child)
        for child in new_children:
            sld_id_lst.append(child)

    entries["ppt/presentation.xml"] = ET.tostring(pres, encoding="utf-8", xml_declaration=True)
    entries["ppt/_rels/presentation.xml.rels"] = ET.tostring(pres_rels, encoding="utf-8", xml_declaration=True)
    entries["[Content_Types].xml"] = ET.tostring(content_types, encoding="utf-8", xml_declaration=True)
    if "docProps/app.xml" in entries:
        app = ET.fromstring(entries["docProps/app.xml"])
        slides_el = app.find(f"{{{APP_NS}}}Slides")
        if slides_el is not None:
            slides_el.text = str(len(original_sld_ids) + len(missing_specs))
            entries["docProps/app.xml"] = ET.tostring(app, encoding="utf-8", xml_declaration=True)

    temp = PPTX_PATH.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)
    with zipfile.ZipFile(temp, "r") as zcheck:
        bad = zcheck.testzip()
        if bad:
            raise RuntimeError(f"Corrupt pptx entry: {bad}")
    temp.replace(PPTX_PATH)
    return PPTX_PATH, backup, preview_paths


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    deck, backup, previews = modify_pptx()
    print(f"UPDATED={deck}")
    print(f"BACKUP={backup}")
    print("PREVIEWS=")
    for preview in previews:
        print(preview)
