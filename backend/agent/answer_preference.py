from __future__ import annotations

import math
import re
from typing import Any


ANSWER_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "length_norm",
    "prompt_keyword_coverage",
    "domain_term_density",
    "evidence_language",
    "structure_language",
    "concrete_detail",
    "generic_or_unsupported",
    "uncertainty_without_evidence",
)


DOMAIN_TERMS = (
    "ppo",
    "dpo",
    "orpo",
    "rag",
    "reward",
    "policy",
    "state",
    "action",
    "trajectory",
    "强化学习",
    "策略",
    "奖励",
    "状态",
    "动作",
    "轨迹",
    "证据",
)

EVIDENCE_TERMS = ("根据", "证据", "来源", "论文", "项目", "知识库", "检索", "文件", "引用", "实验")
STRUCTURE_TERMS = ("首先", "其次", "最后", "因此", "例如", "包括", "可以", "关键", "优势", "限制", "：", "；")
GENERIC_TERMS = ("大致", "后续再补充", "没有检索", "没有引用", "不知道", "可能吧", "泛泛", "缺少证据")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", (text or "").lower())


def _contains_any(text: str, terms: tuple[str, ...]) -> float:
    lowered = (text or "").lower()
    return min(sum(1 for term in terms if term.lower() in lowered) / 4.0, 1.0)


def answer_features(prompt: str, answer: str) -> list[float]:
    prompt_tokens = set(_tokens(prompt))
    answer_tokens = set(_tokens(answer))
    coverage = len(prompt_tokens & answer_tokens) / max(len(prompt_tokens), 1)
    answer_len = len(answer or "")
    domain_hits = sum(1 for term in DOMAIN_TERMS if term.lower() in (answer or "").lower())
    concrete_numbers = len(re.findall(r"\d+|PPO|DPO|RAG|top[-_ ]?k|reward|policy", answer or "", flags=re.I))
    evidence_language = _contains_any(answer, EVIDENCE_TERMS)
    if any(term in (answer or "") for term in ("没有检索", "没有引用", "缺少证据", "无证据")):
        evidence_language = 0.0
    return [
        1.0,
        min(answer_len / 700.0, 1.0),
        min(coverage, 1.0),
        min(domain_hits / 8.0, 1.0),
        evidence_language,
        _contains_any(answer, STRUCTURE_TERMS),
        min(concrete_numbers / 6.0, 1.0),
        _contains_any(answer, GENERIC_TERMS),
        1.0 if ("可能" in answer or "大概" in answer) and not _contains_any(answer, EVIDENCE_TERMS) else 0.0,
    ]


def score_answer(features: list[float], weights: list[float]) -> float:
    total = sum(float(feature) * float(weight) for feature, weight in zip(features, weights))
    return total if math.isfinite(total) else 0.0


def format_preference_row(row: dict[str, Any]) -> dict[str, str]:
    prompt = str(row.get("prompt") or row.get("instruction") or row.get("query") or "").strip()
    chosen = str(row.get("chosen") or row.get("preferred") or row.get("answer") or "").strip()
    rejected = str(row.get("rejected") or row.get("dispreferred") or "").strip()
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
