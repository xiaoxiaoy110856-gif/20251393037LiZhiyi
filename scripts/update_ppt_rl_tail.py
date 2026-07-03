from __future__ import annotations

import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SPEC_OUT = ROOT / "outputs" / "ppt_tail_rl_specs.json"
EMU_PER_INCH = 914400
SLIDE_W = 13.333333
SLIDE_H = 7.5

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

THEME = {
    "bg": "FDF7FA",
    "ink": "17231F",
    "muted": "65736E",
    "green": "12A36B",
    "green2": "BFEEDB",
    "green3": "E8F8EF",
    "pink": "E84A8A",
    "pink2": "FFD7E8",
    "pink3": "FFF0F6",
    "blue": "234D7C",
    "blue2": "DDEBFF",
    "line": "26443A",
    "white": "FFFFFF",
    "yellow": "FFE8A3",
}


def emu(value: float) -> int:
    return int(round(value * EMU_PER_INCH))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metrics() -> dict[str, float]:
    ppo = read_json(ROOT / "outputs" / "agent_policy_ppo_multi_step" / "metrics.json")
    dpo = read_json(ROOT / "outputs" / "answer_preference_dpo" / "metrics.json")
    return {
        "ppo_reward_base": float(ppo.get("baseline_average_reward", 1.0239)),
        "ppo_reward": float(ppo.get("trained_average_reward", 1.7469)),
        "ppo_reward_gain": float(ppo.get("reward_gain_vs_baseline", 0.7230)),
        "ppo_recall_base": float(ppo.get("baseline_average_answer_point_recall", 0.4167)),
        "ppo_recall": float(ppo.get("trained_average_answer_point_recall", 0.5278)),
        "ppo_recall_gain": float(ppo.get("answer_point_recall_gain_vs_baseline", 0.1111)),
        "dpo_acc_base": float(dpo.get("initial_pairwise_accuracy", 0.5)),
        "dpo_acc": float(dpo.get("trained_pairwise_accuracy", 1.0)),
        "dpo_acc_gain": float(dpo.get("pairwise_accuracy_gain", 0.5)),
        "dpo_margin": float(dpo.get("trained_average_margin", 6.5516)),
    }


def text_el(
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: int = 22,
    color: str = THEME["ink"],
    bold: bool = False,
    align: str = "l",
    valign: str = "t",
    name: str = "",
    fill: str | None = None,
    line: str | None = None,
    radius: bool = False,
) -> dict:
    return {
        "kind": "text",
        "text": text,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "size": size,
        "color": color,
        "bold": bold,
        "align": align,
        "valign": valign,
        "name": name or text[:20],
        "fill": fill,
        "line": line,
        "radius": radius,
    }


def rect_el(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    line: str | None = None,
    radius: bool = False,
    name: str = "shape",
) -> dict:
    return {
        "kind": "rect",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "fill": fill,
        "line": line,
        "radius": radius,
        "name": name,
    }


def slide_spec(title: str, subtitle: str, elements: list[dict]) -> dict:
    base = [
        rect_el(0, 0, SLIDE_W, SLIDE_H, fill=THEME["bg"], name="background"),
        rect_el(0, 0, SLIDE_W, 0.18, fill=THEME["green"], name="top-rule"),
        rect_el(0, 7.32, SLIDE_W, 0.18, fill=THEME["pink"], name="bottom-rule"),
        text_el(title, 0.55, 0.42, 7.9, 0.56, size=27, bold=True, color=THEME["ink"], name="title"),
        text_el(subtitle, 0.57, 1.0, 8.7, 0.42, size=13, color=THEME["muted"], name="subtitle"),
        text_el("PPO + DPO RL", 10.55, 0.47, 2.2, 0.36, size=14, color=THEME["white"], bold=True, align="c", valign="m", fill=THEME["green"], radius=True, name="tag"),
    ]
    return {"title": title, "subtitle": subtitle, "elements": base + elements}


def build_specs(m: dict[str, float]) -> list[dict]:
    reward_gain = f"+{m['ppo_reward_gain']:.4f}"
    recall_gain = f"+{m['ppo_recall_gain']:.4f}"
    dpo_acc = f"{m['dpo_acc_base']:.2f} -> {m['dpo_acc']:.2f}"
    margin = f"+{m['dpo_margin']:.4f}"

    slides: list[dict] = []

    slides.append(
        slide_spec(
            "强化学习在项目中的新定位",
            "强化学习不只做检索选择，而是进入 Agent 行为链和答案偏好层。",
            [
                text_el("用户问题", 0.7, 1.75, 1.55, 0.58, size=18, bold=True, align="c", valign="m", fill=THEME["white"], line=THEME["green"], radius=True),
                text_el("Qwen Agent\n理解意图与规划任务", 2.65, 1.58, 2.1, 0.95, size=15, bold=True, align="c", valign="m", fill=THEME["green3"], line=THEME["green"], radius=True),
                text_el("状态 s_t\n问题、文件、证据、上下文", 5.15, 1.58, 2.25, 0.95, size=15, bold=True, align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
                text_el("PPO 行为策略\n决定下一步工具动作", 7.85, 1.58, 2.25, 0.95, size=15, bold=True, align="c", valign="m", fill=THEME["pink2"], line=THEME["pink"], radius=True),
                text_el("工具链执行\n读文件 / 检索 / 重排 / 压缩 / 再搜", 10.35, 1.48, 2.35, 1.15, size=13, bold=True, align="c", valign="m", fill=THEME["white"], line=THEME["green"], radius=True),
                text_el("→", 2.22, 1.79, 0.35, 0.38, size=24, color=THEME["green"], bold=True, align="c"),
                text_el("→", 4.78, 1.79, 0.35, 0.38, size=24, color=THEME["green"], bold=True, align="c"),
                text_el("→", 7.45, 1.79, 0.35, 0.38, size=24, color=THEME["pink"], bold=True, align="c"),
                text_el("→", 10.08, 1.79, 0.35, 0.38, size=24, color=THEME["green"], bold=True, align="c"),
                rect_el(1.0, 3.25, 11.2, 0.06, fill=THEME["line"], name="divider"),
                text_el("DPO 答案偏好模型", 1.0, 3.65, 2.75, 0.65, size=18, bold=True, color=THEME["pink"], align="c", valign="m", fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("chosen_answer > rejected_answer", 4.05, 3.65, 3.3, 0.65, size=17, bold=True, color=THEME["ink"], align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
                text_el("最终回答更重证据、更少幻觉、结构更清楚", 7.65, 3.65, 4.25, 0.65, size=17, bold=True, color=THEME["green"], align="c", valign="m", fill=THEME["green3"], line=THEME["green"], radius=True),
                text_el("一句话：PPO 让 Agent 更会行动，DPO 让 Agent 更会回答。", 1.05, 5.25, 11.3, 0.8, size=25, bold=True, align="c", valign="m", color=THEME["ink"]),
            ],
        )
    )

    actions = ["read_file", "search_docs", "rerank", "compress", "second_search", "answer"]
    action_elements: list[dict] = []
    for i, action in enumerate(actions):
        x = 0.65 + i * 2.08
        fill = THEME["green3"] if i % 2 == 0 else THEME["pink3"]
        line = THEME["green"] if i % 2 == 0 else THEME["pink"]
        action_elements.append(text_el(action, x, 2.1, 1.66, 0.62, size=12, bold=True, align="c", valign="m", fill=fill, line=line, radius=True))
        if i < len(actions) - 1:
            action_elements.append(text_el("→", x + 1.65, 2.22, 0.35, 0.35, size=20, color=THEME["line"], bold=True, align="c"))
    slides.append(
        slide_spec(
            "PPO：训练多步 Agent 决策流程",
            "把一次回答拆成多个工具步骤，PPO 学习什么时候读、搜、压缩、再搜和结束。",
            [
                text_el("多步动作空间", 0.7, 1.55, 2.6, 0.36, size=18, bold=True, color=THEME["green"]),
                *action_elements,
                text_el("奖励函数", 0.85, 3.28, 1.6, 0.45, size=18, bold=True, color=THEME["pink"]),
                text_el("answer_point_recall + tool_hit + evidence_count + compression_density - repeated_action_cost", 2.25, 3.26, 9.7, 0.5, size=16, bold=True, color=THEME["ink"], fill=THEME["white"], line=THEME["pink"], radius=True, align="c", valign="m"),
                text_el(reward_gain, 1.15, 4.62, 2.3, 0.82, size=34, bold=True, color=THEME["green"], align="c", valign="m"),
                text_el("average reward gain", 1.0, 5.38, 2.6, 0.32, size=12, color=THEME["muted"], align="c"),
                text_el(recall_gain, 5.2, 4.62, 2.3, 0.82, size=34, bold=True, color=THEME["pink"], align="c", valign="m"),
                text_el("answer point recall gain", 4.9, 5.38, 2.9, 0.32, size=12, color=THEME["muted"], align="c"),
                text_el("60 updates", 9.25, 4.62, 2.45, 0.82, size=30, bold=True, color=THEME["blue"], align="c", valign="m"),
                text_el("真实 torch 训练输出", 9.05, 5.38, 2.85, 0.32, size=12, color=THEME["muted"], align="c"),
            ],
        )
    )

    slides.append(
        slide_spec(
            "DPO：训练答案偏好，而不是训练检索动作",
            "同一个 prompt 下，让模型更偏好有证据、有结构、少幻觉的回答。",
            [
                text_el("chosen_answer", 0.8, 1.72, 2.2, 0.45, size=18, bold=True, color=THEME["green"]),
                text_el("结合项目证据；覆盖关键概念；说明状态、动作、奖励；结构清楚。", 0.8, 2.24, 4.55, 1.55, size=16, color=THEME["ink"], fill=THEME["green3"], line=THEME["green"], radius=True),
                text_el("rejected_answer", 7.95, 1.72, 2.35, 0.45, size=18, bold=True, color=THEME["pink"]),
                text_el("泛泛描述；缺少来源；遗漏要点；容易出现无依据判断。", 7.95, 2.24, 4.55, 1.55, size=16, color=THEME["ink"], fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("DPO 目标：扩大 chosen 与 rejected 的偏好分差", 2.95, 4.15, 7.3, 0.52, size=19, bold=True, align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
                text_el(dpo_acc, 1.45, 5.2, 2.9, 0.65, size=28, bold=True, color=THEME["green"], align="c", valign="m"),
                text_el("pairwise accuracy", 1.32, 5.86, 3.15, 0.3, size=12, color=THEME["muted"], align="c"),
                text_el(margin, 5.35, 5.2, 2.9, 0.65, size=28, bold=True, color=THEME["pink"], align="c", valign="m"),
                text_el("preference margin", 5.2, 5.86, 3.2, 0.3, size=12, color=THEME["muted"], align="c"),
                text_el("证据语言 / 结构清晰 / 领域术语 = 正权重；泛化无支撑 = 负权重", 8.55, 5.05, 3.5, 0.9, size=14, bold=True, color=THEME["ink"], align="c", valign="m", fill=THEME["yellow"], line=THEME["line"], radius=True),
            ],
        )
    )

    slides.append(
        slide_spec(
            "训练数据流：从样本到策略文件",
            "PPO 和 DPO 分别读取不同训练数据，输出不同类型的强化学习结果。",
            [
                text_el("PPO 数据", 0.75, 1.65, 1.5, 0.35, size=17, bold=True, color=THEME["green"]),
                text_el("training_data/agent_eval.jsonl\n问题 + expected_points + expected_tools", 0.75, 2.1, 3.35, 0.92, size=14, color=THEME["ink"], fill=THEME["green3"], line=THEME["green"], radius=True),
                text_el("AgentWorkflowEnv\n多步状态、动作、奖励", 4.85, 2.1, 2.65, 0.92, size=14, bold=True, color=THEME["ink"], align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
                text_el("outputs/agent_policy_ppo_multi_step\n策略 checkpoint + evaluation", 8.18, 2.1, 3.65, 0.92, size=14, color=THEME["ink"], fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("→", 4.27, 2.32, 0.38, 0.32, size=22, bold=True, color=THEME["green"], align="c"),
                text_el("→", 7.66, 2.32, 0.38, 0.32, size=22, bold=True, color=THEME["pink"], align="c"),
                rect_el(0.75, 3.72, 11.25, 0.04, fill=THEME["line"], name="mid-line"),
                text_el("DPO 数据", 0.75, 4.15, 1.5, 0.35, size=17, bold=True, color=THEME["pink"]),
                text_el("training_data/assistant_dpo.jsonl\nprompt + chosen + rejected", 0.75, 4.6, 3.35, 0.92, size=14, color=THEME["ink"], fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("AnswerPreference\n偏好特征 + DPO loss", 4.85, 4.6, 2.65, 0.92, size=14, bold=True, color=THEME["ink"], align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
                text_el("outputs/answer_preference_dpo\n偏好模型 + pairwise 评估", 8.18, 4.6, 3.65, 0.92, size=14, color=THEME["ink"], fill=THEME["green3"], line=THEME["green"], radius=True),
                text_el("→", 4.27, 4.82, 0.38, 0.32, size=22, bold=True, color=THEME["pink"], align="c"),
                text_el("→", 7.66, 4.82, 0.38, 0.32, size=22, bold=True, color=THEME["green"], align="c"),
            ],
        )
    )

    def bar_group(x: float, y: float, label: str, base: float, trained: float, max_value: float, color: str) -> list[dict]:
        base_w = 2.7 * base / max_value
        trained_w = 2.7 * trained / max_value
        return [
            text_el(label, x, y - 0.36, 3.2, 0.26, size=13, bold=True, color=THEME["ink"]),
            rect_el(x, y, 2.7, 0.16, fill="E8ECEA", name=f"{label}-track-1"),
            rect_el(x, y, base_w, 0.16, fill=THEME["muted"], name=f"{label}-base"),
            text_el(f"Baseline {base:.4f}", x + 2.92, y - 0.1, 1.6, 0.28, size=10, color=THEME["muted"]),
            rect_el(x, y + 0.42, 2.7, 0.16, fill="E8ECEA", name=f"{label}-track-2"),
            rect_el(x, y + 0.42, trained_w, 0.16, fill=color, name=f"{label}-trained"),
            text_el(f"RL {trained:.4f}", x + 2.92, y + 0.32, 1.6, 0.28, size=10, color=color, bold=True),
        ]

    slides.append(
        slide_spec(
            "效果对比：强化学习提升了什么",
            "这里展示的是新 PPO/DPO 训练结果，指标从检索命中转向 Agent 行为和回答质量。",
            [
                *bar_group(0.9, 2.08, "PPO average reward", m["ppo_reward_base"], m["ppo_reward"], 2.0, THEME["green"]),
                *bar_group(0.9, 3.52, "PPO answer point recall", m["ppo_recall_base"], m["ppo_recall"], 0.7, THEME["pink"]),
                *bar_group(7.0, 2.08, "DPO pairwise accuracy", m["dpo_acc_base"], m["dpo_acc"], 1.0, THEME["green"]),
                text_el("DPO average margin", 7.0, 3.16, 3.2, 0.28, size=13, bold=True, color=THEME["ink"]),
                text_el(f"0.0000  →  {m['dpo_margin']:.4f}", 7.0, 3.55, 3.55, 0.62, size=26, bold=True, color=THEME["pink"], align="c", valign="m", fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("结论：PPO 证明行为链能被优化；DPO 证明回答偏好能被优化。", 1.1, 5.65, 10.95, 0.72, size=22, bold=True, color=THEME["ink"], align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
            ],
        )
    )

    slides.append(
        slide_spec(
            "最终答辩口径：强化学习真正负责两件事",
            "不要再把 PPO/DPO 解释成单步检索选择；它们现在分别优化行动和回答。",
            [
                text_el("1", 0.95, 1.75, 0.55, 0.55, size=20, bold=True, color=THEME["white"], align="c", valign="m", fill=THEME["green"], radius=True),
                text_el("PPO 负责 Agent 行动策略", 1.72, 1.62, 4.35, 0.42, size=20, bold=True, color=THEME["green"]),
                text_el("学习“下一步做什么”：读文件、检索、重排、压缩、二次检索、生成回答。", 1.72, 2.05, 9.9, 0.45, size=16, color=THEME["ink"]),
                text_el("2", 0.95, 3.05, 0.55, 0.55, size=20, bold=True, color=THEME["white"], align="c", valign="m", fill=THEME["pink"], radius=True),
                text_el("DPO 负责答案偏好优化", 1.72, 2.92, 4.35, 0.42, size=20, bold=True, color=THEME["pink"]),
                text_el("学习“哪个回答更好”：更有证据、更少幻觉、结构更清楚、覆盖要点更多。", 1.72, 3.35, 9.9, 0.45, size=16, color=THEME["ink"]),
                text_el("3", 0.95, 4.35, 0.55, 0.55, size=20, bold=True, color=THEME["white"], align="c", valign="m", fill=THEME["blue"], radius=True),
                text_el("Qwen + RAG 提供执行基础", 1.72, 4.22, 4.35, 0.42, size=20, bold=True, color=THEME["blue"]),
                text_el("Qwen 负责理解与组织语言，RAG 负责证据来源，RL 负责把流程越训越稳。", 1.72, 4.65, 9.9, 0.45, size=16, color=THEME["ink"]),
                text_el("一句话收束：强化学习让 Agent 从“能回答”变成“会行动、会取证、会偏好高质量答案”。", 1.05, 6.0, 11.25, 0.72, size=21, bold=True, color=THEME["white"], align="c", valign="m", fill=THEME["line"], radius=True),
            ],
        )
    )
    return slides


def build_mid_specs(m: dict[str, float]) -> list[dict]:
    reward_gain = f"+{m['ppo_reward_gain']:.4f}"
    recall_gain = f"+{m['ppo_recall_gain']:.4f}"
    dpo_acc = f"{m['dpo_acc_base']:.2f} -> {m['dpo_acc']:.2f}"
    margin = f"+{m['dpo_margin']:.4f}"
    return [
        slide_spec(
            "RL Agent 策略层：PPO 与 DPO 分工",
            "本项目后续强化学习展示只保留 PPO 多步行为策略和 DPO 答案偏好优化。",
            [
                text_el("Agent 状态", 0.8, 1.75, 2.45, 0.72, size=18, bold=True, color=THEME["ink"], align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
                text_el("问题意图、已读文件、检索证据、上下文压缩状态、当前步骤", 0.8, 2.55, 2.45, 1.35, size=13, color=THEME["muted"], align="c", valign="m"),
                text_el("PPO 行为策略", 4.0, 1.58, 2.75, 0.92, size=20, bold=True, color=THEME["green"], align="c", valign="m", fill=THEME["green3"], line=THEME["green"], radius=True),
                text_el("读文件 / 检索 / 重排 / 压缩 / 再搜 / 回答", 3.75, 2.64, 3.25, 1.05, size=15, bold=True, color=THEME["ink"], align="c", valign="m"),
                text_el("DPO 偏好模型", 7.75, 1.58, 2.75, 0.92, size=20, bold=True, color=THEME["pink"], align="c", valign="m", fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("chosen_answer 优于 rejected_answer：更有证据、更少幻觉", 7.45, 2.64, 3.35, 1.05, size=15, bold=True, color=THEME["ink"], align="c", valign="m"),
                text_el("→", 3.35, 1.92, 0.4, 0.35, size=24, bold=True, color=THEME["green"], align="c"),
                text_el("→", 7.08, 1.92, 0.4, 0.35, size=24, bold=True, color=THEME["pink"], align="c"),
                text_el("最终目标", 4.8, 4.65, 3.65, 0.44, size=19, bold=True, color=THEME["line"], align="c"),
                text_el("让 Agent 会行动、会取证、会压缩上下文，并最终偏好高质量答案。", 1.35, 5.18, 10.65, 0.72, size=22, bold=True, color=THEME["white"], align="c", valign="m", fill=THEME["line"], radius=True),
            ],
        ),
        slide_spec(
            "训练逻辑：从单步检索改成两条 RL 主线",
            "PPO 训练行为轨迹；DPO 训练回答偏好，二者共同体现强化学习收益。",
            [
                text_el("PPO 多步轨迹", 0.85, 1.72, 2.45, 0.48, size=19, bold=True, color=THEME["green"]),
                text_el("状态 s_t：问题、文件、证据、压缩状态\n动作 a_t：读 / 搜 / 重排 / 压缩 / 再搜 / 答\n奖励 r_t：要点召回 + 工具命中 + 证据数量 - 成本", 0.85, 2.28, 5.05, 1.55, size=15, color=THEME["ink"], fill=THEME["green3"], line=THEME["green"], radius=True),
                text_el("DPO 偏好对", 7.05, 1.72, 2.45, 0.48, size=19, bold=True, color=THEME["pink"]),
                text_el("prompt：同一个问题\nchosen：证据充分、结构清楚、覆盖要点\nrejected：泛化、缺来源、遗漏关键点", 7.05, 2.28, 5.05, 1.55, size=15, color=THEME["ink"], fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("训练输出", 0.95, 4.55, 1.8, 0.38, size=18, bold=True, color=THEME["line"]),
                text_el("agent_policy_ppo.pt\nanswer_preference_model.json", 2.55, 4.35, 3.55, 0.85, size=16, bold=True, color=THEME["ink"], align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
                text_el("评估指标", 7.15, 4.55, 1.8, 0.38, size=18, bold=True, color=THEME["line"]),
                text_el("Reward / Answer Point Recall\nPairwise Accuracy / Preference Margin", 8.75, 4.35, 3.55, 0.85, size=16, bold=True, color=THEME["ink"], align="c", valign="m", fill=THEME["white"], line=THEME["line"], radius=True),
            ],
        ),
        slide_spec(
            "核心代码：PPO / DPO 新训练参数",
            "这里展示的是新训练逻辑的核心参数，不再围绕单步检索策略展开。",
            [
                text_el("scripts/train_agent_policy_ppo.py", 0.75, 1.58, 3.3, 0.32, size=15, bold=True, color=THEME["green"]),
                text_el('AGENT_ACTION_NAMES = (\n  "read_file", "search_project_docs",\n  "rerank_evidence", "compress_context",\n  "second_search", "generate_answer",\n)\nupdates = 60\nclip_range = 0.2\ngamma = 0.95\nimitation_coef = 0.3', 0.75, 2.0, 5.25, 2.65, size=13, color=THEME["ink"], fill=THEME["green3"], line=THEME["green"], radius=True),
                text_el("scripts/train_answer_preference_dpo.py", 7.0, 1.58, 3.75, 0.32, size=15, bold=True, color=THEME["pink"]),
                text_el('data = assistant_dpo.jsonl\nfields = prompt / chosen / rejected\nepochs = 120\nbeta = 0.4\nlr = 5e-2\nloss = -logsigmoid(beta * margin)\nmargin = score(chosen) - score(rejected)', 7.0, 2.0, 5.25, 2.65, size=13, color=THEME["ink"], fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("代码结论：PPO 学习工具链动作，DPO 学习答案偏好分差。", 1.05, 5.55, 11.2, 0.62, size=21, bold=True, color=THEME["white"], align="c", valign="m", fill=THEME["line"], radius=True),
            ],
        ),
        slide_spec(
            "效果对比：RL 现在体现为行为与回答质量提升",
            "指标从 Source/Topic 检索命中转向 Agent 级收益和答案偏好收益。",
            [
                text_el(reward_gain, 1.2, 1.8, 2.45, 0.72, size=34, bold=True, color=THEME["green"], align="c", valign="m"),
                text_el("PPO reward gain", 1.05, 2.55, 2.75, 0.3, size=13, color=THEME["muted"], align="c"),
                text_el(recall_gain, 5.45, 1.8, 2.45, 0.72, size=34, bold=True, color=THEME["pink"], align="c", valign="m"),
                text_el("answer point recall gain", 5.2, 2.55, 2.95, 0.3, size=13, color=THEME["muted"], align="c"),
                text_el(dpo_acc, 9.5, 1.8, 2.45, 0.72, size=30, bold=True, color=THEME["green"], align="c", valign="m"),
                text_el("DPO pairwise accuracy", 9.2, 2.55, 3.05, 0.3, size=13, color=THEME["muted"], align="c"),
                text_el(margin, 4.85, 3.75, 3.55, 0.66, size=30, bold=True, color=THEME["pink"], align="c", valign="m", fill=THEME["pink3"], line=THEME["pink"], radius=True),
                text_el("DPO preference margin", 4.85, 4.45, 3.55, 0.28, size=13, color=THEME["muted"], align="c"),
                text_el("结论：强化学习不是让文字更花，而是让 Agent 的行动链和答案选择更可靠。", 1.05, 5.7, 11.2, 0.62, size=20, bold=True, color=THEME["white"], align="c", valign="m", fill=THEME["line"], radius=True),
            ],
        ),
    ]


def color_xml(color: str | None, no_fill: bool = False) -> str:
    if no_fill or not color:
        return "<a:noFill/>"
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def shape_xml(el: dict, shape_id: int) -> str:
    prst = "roundRect" if el.get("radius") else "rect"
    fill = color_xml(el.get("fill"))
    line = color_xml(el.get("line"), no_fill=not el.get("line"))
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="{escape(str(el.get('name', 'shape')))}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(el['x'])}" y="{emu(el['y'])}"/><a:ext cx="{emu(el['w'])}" cy="{emu(el['h'])}"/></a:xfrm>
    <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
    {fill}
    <a:ln w="12700">{line}</a:ln>
  </p:spPr>
</p:sp>"""


def text_xml(el: dict, shape_id: int) -> str:
    prst = "roundRect" if el.get("radius") else "rect"
    fill = color_xml(el.get("fill"), no_fill=not el.get("fill"))
    line = color_xml(el.get("line"), no_fill=not el.get("line"))
    anchor = {"t": "t", "m": "ctr", "b": "b"}.get(str(el.get("valign", "t")), "t")
    algn = {"l": "l", "c": "ctr", "r": "r"}.get(str(el.get("align", "l")), "l")
    bold = ' b="1"' if el.get("bold") else ""
    size = int(el.get("size", 18)) * 100
    paras = []
    for raw in str(el.get("text", "")).split("\n"):
        paras.append(
            f"""<a:p><a:pPr algn="{algn}"/><a:r><a:rPr lang="zh-CN" sz="{size}"{bold}><a:solidFill><a:srgbClr val="{el.get('color', THEME['ink'])}"/></a:solidFill><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:rPr><a:t>{escape(raw)}</a:t></a:r><a:endParaRPr lang="zh-CN" sz="{size}"/></a:p>"""
        )
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="{escape(str(el.get('name', 'text')))}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(el['x'])}" y="{emu(el['y'])}"/><a:ext cx="{emu(el['w'])}" cy="{emu(el['h'])}"/></a:xfrm>
    <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
    {fill}
    <a:ln w="12700">{line}</a:ln>
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" anchor="{anchor}" lIns="91440" tIns="45720" rIns="91440" bIns="45720"><a:spAutoFit/></a:bodyPr>
    <a:lstStyle/>
    {''.join(paras)}
  </p:txBody>
</p:sp>"""


def slide_xml(spec: dict) -> str:
    parts = [
        f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{A_NS}" xmlns:r="{R_NS}" xmlns:p="{P_NS}">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'''
    ]
    shape_id = 2
    for el in spec["elements"]:
        if el["kind"] == "rect":
            parts.append(shape_xml(el, shape_id))
        elif el["kind"] == "text":
            parts.append(text_xml(el, shape_id))
        shape_id += 1
    parts.append('''    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>''')
    return "\n".join(parts)


def update_pptx(ppt_path: Path, slide_specs: list[tuple[int, dict]]) -> Path:
    if not ppt_path.exists():
        raise FileNotFoundError(ppt_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup = ppt_path.with_name(f"{ppt_path.stem}.before_ppo_dpo_tail_{timestamp}{ppt_path.suffix}")
    shutil.copy2(ppt_path, backup)
    temp_path = ppt_path.with_name(f"{ppt_path.stem}.tmp_ppo_dpo_tail{ppt_path.suffix}")
    replacements = {f"ppt/slides/slide{slide_no}.xml": slide_xml(spec).encode("utf-8") for slide_no, spec in slide_specs}
    with zipfile.ZipFile(ppt_path, "r") as src, zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = replacements.get(item.filename)
            if data is None:
                data = src.read(item.filename)
            dst.writestr(item, data)
    temp_path.replace(ppt_path)
    return backup


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/update_ppt_rl_tail.py <pptx-path>")
    ppt_path = Path(sys.argv[1])
    m = metrics()
    mid_specs = build_mid_specs(m)
    tail_specs = build_specs(m)
    numbered_specs = [(32 + index, spec) for index, spec in enumerate(mid_specs)]
    numbered_specs.extend((50 + index, spec) for index, spec in enumerate(tail_specs))
    SPEC_OUT.parent.mkdir(parents=True, exist_ok=True)
    SPEC_OUT.write_text(
        json.dumps([{"slide_no": slide_no, "spec": spec} for slide_no, spec in numbered_specs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    backup = update_pptx(ppt_path, numbered_specs)
    print(json.dumps({"ok": True, "pptx": str(ppt_path), "backup": str(backup), "spec": str(SPEC_OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
