from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.retrieval.knowledge_store import SOURCE_ALIASES, search_knowledge


@dataclass(frozen=True)
class RetrievalAction:
    """一个离散检索动作：描述如何改写 query、取多少 top-k，以及这个动作的语义。"""
    name: str
    description: str
    top_k: int
    query_suffixes: tuple[str, ...] = ()


# 核心5：离散检索动作集合。强化学习不直接改 Qwen，而是学习回答前该使用哪种检索策略/top-k/query 后缀。
ACTIONS: tuple[RetrievalAction, ...] = (
    RetrievalAction("baseline", "Use the original query and a balanced top-k.", top_k=4),
    RetrievalAction(
        "rl_focus",
        "Bias retrieval toward reinforcement learning terminology.",
        top_k=4,
        query_suffixes=("reinforcement learning policy optimization actor-critic", "PPO trajectory optimization"),
    ),
    RetrievalAction(
        "trajectory_focus",
        "Bias retrieval toward trajectory-domain terminology.",
        top_k=4,
        query_suffixes=("trajectory analysis trajectory modeling", "trajectory planning trajectory optimization"),
    ),
    RetrievalAction(
        "paper_focus",
        "Bias retrieval toward papers and named methods.",
        top_k=5,
        query_suffixes=("paper method benchmark ablation", "survey theorem experiment"),
    ),
    RetrievalAction(
        "compression_focus",
        "Bias retrieval toward compression and simplification materials.",
        top_k=5,
        query_suffixes=("trajectory compression trajectory simplification", "Douglas-Peucker DOTS Fast Trajectory Simplification"),
    ),
    RetrievalAction(
        "planning_focus",
        "Bias retrieval toward planning and optimization materials.",
        top_k=5,
        query_suffixes=("trajectory planning motion planning path planning", "trajectory optimization differentiable trajectory optimization"),
    ),
    RetrievalAction(
        "similarity_focus",
        "Bias retrieval toward similarity and subtrajectory search.",
        top_k=5,
        query_suffixes=("trajectory similarity subtrajectory search", "Frechet Hausdorff DTW"),
    ),
    RetrievalAction(
        "reward_focus",
        "Bias retrieval toward reward design and inverse reinforcement learning.",
        top_k=5,
        query_suffixes=("reward design inverse reinforcement learning", "reward shaping preference optimization"),
    ),
    RetrievalAction(
        "broad_search",
        "Use a broader query and a larger top-k to gather more evidence.",
        top_k=7,
        query_suffixes=("trajectory reinforcement learning experiment benchmark",),
    ),
)


def compose_retrieval_query(query: str, action: RetrievalAction) -> str:
    """把原始问题和动作携带的检索后缀拼成最终送入知识库的 query。"""
    suffix = " ".join(part for part in action.query_suffixes if part)
    return f"{query} {suffix}".strip() if suffix else query


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 jsonl 训练/评测数据，每一行是一条检索问题样本。"""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _normalize(text: str) -> str:
    """统一小写、去掉多余空白，便于做来源/主题命中匹配。"""
    return " ".join((text or "").strip().lower().split())


# 核心5指标：Source Hit 衡量检索结果是否找到了该问题预期的论文、文件或证据来源。
def source_hit(results: list[dict[str, Any]], expected_sources: list[str]) -> float:
    if not expected_sources:
        return 1.0
    haystack = _normalize(" ".join((item.get("path", "") + " " + item.get("title", "")) for item in results))
    hits = 0
    for source in expected_sources:
        aliases = SOURCE_ALIASES.get(_normalize(source), [source])
        if any(_normalize(alias) in haystack for alias in aliases):
            hits += 1
    return hits / len(expected_sources)


# 核心5指标：Topic Hit 衡量检索片段是否覆盖预期主题，不强依赖具体文件名。
def topic_hit(results: list[dict[str, Any]], expected_topics: list[str]) -> float:
    if not expected_topics:
        return 1.0
    topics = {topic.lower() for item in results for topic in item.get("topics", [])}
    hits = sum(1 for topic in expected_topics if topic.lower() in topics)
    return hits / len(expected_topics)


def point_recall(text: str, expected_points: list[str]) -> float:
    """衡量检索片段中是否召回了预期答案要点。"""
    if not expected_points:
        return 1.0
    normalized = _normalize(text)
    hits = sum(1 for point in expected_points if _normalize(point) in normalized)
    return hits / len(expected_points)


# 核心5状态：把用户问题转换成紧凑特征向量，供 PPO/DPO/ORPO、LinUCB、DQN、Dueling DDQN 使用。
def features_for_query(query: str) -> list[float]:
    lowered = query.lower()
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", query))
    english_count = len(re.findall(r"[a-zA-Z]", query))
    return [
      1.0 if "ppo" in lowered else 0.0,
      1.0 if "dqn" in lowered else 0.0,
      1.0 if "sac" in lowered else 0.0,
      1.0 if any(token in lowered for token in ["compression", "simplification"]) or "压缩" in query or "简化" in query else 0.0,
      1.0 if "trajectory" in lowered or "轨迹" in query else 0.0,
      1.0 if any(token in lowered for token in ["planning", "optimization"]) or "规划" in query or "优化" in query else 0.0,
      1.0 if any(token in lowered for token in ["similarity", "subtrajectory", "frechet", "hausdorff", "dtw"]) or "相似" in query else 0.0,
      1.0 if any(token in lowered for token in ["reward", "preference", "inverse reinforcement"]) or "奖励" in query or "偏好" in query else 0.0,
      1.0 if any(token in lowered for token in ["paper", "survey", "benchmark", "experiment"]) or "论文" in query or "实验" in query else 0.0,
      min(len(query) / 120.0, 1.0),
      min(cjk_count / 40.0, 1.0),
      min(english_count / 80.0, 1.0),
    ]


# 核心5环境：一个问题就是一个 episode。策略选择检索动作，环境执行 RAG 搜索并返回 Source/Topic/Point reward。
class RetrievalRLEnv:
    def __init__(self, examples: list[dict[str, Any]], reward_weights: dict[str, float] | None = None) -> None:
        self.examples = examples
        self.reward_weights = reward_weights or {
            "source_hit": 0.5,
            "topic_hit": 0.3,
            "point_recall": 0.2,
        }
        self.current_index = 0
        self._search_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    @property
    def action_space(self) -> tuple[RetrievalAction, ...]:
        return ACTIONS

    @property
    def state_size(self) -> int:
        return len(features_for_query("dummy"))

    @property
    def action_size(self) -> int:
        return len(ACTIONS)

    def sample_index(self) -> int:
        return random.randrange(len(self.examples))

    def reset(self, index: int) -> list[float]:
        """切换到指定样本，并返回该问题对应的状态特征。"""
        self.current_index = index
        example = self.examples[index]
        query = str(example.get("query") or example.get("task") or "").strip()
        return features_for_query(query)

    def _compose_query(self, query: str, action: RetrievalAction) -> str:
        return compose_retrieval_query(query, action)

    def step(self, action_index: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        """执行一次检索动作，计算 reward，并返回下一状态、奖励、结束标记和调试信息。"""
        example = self.examples[self.current_index]
        query = str(example.get("query") or example.get("task") or "").strip()
        expected_sources = [str(item) for item in example.get("expected_sources", [])]
        expected_topics = [str(item) for item in example.get("expected_topics", [])]
        expected_points = [str(item) for item in example.get("expected_points", [])]

        action = ACTIONS[action_index]
        retrieval_query = self._compose_query(query, action)
        cache_key = (retrieval_query, action.top_k)
        if cache_key not in self._search_cache:
            self._search_cache[cache_key] = search_knowledge(retrieval_query, top_k=action.top_k)
        results = self._search_cache[cache_key]
        combined_snippets = " ".join(item.get("snippet", "") for item in results)

        source_score = source_hit(results, expected_sources)
        topic_score = topic_hit(results, expected_topics)
        point_score = point_recall(combined_snippets, expected_points)
        # 核心5奖励：证据质量加权得分减去宽 top-k 的轻微成本，确保提升来自更精准的检索。
        reward = (
            self.reward_weights["source_hit"] * source_score
            + self.reward_weights["topic_hit"] * topic_score
            + self.reward_weights["point_recall"] * point_score
            - 0.015 * max(action.top_k - 4, 0)
        )

        info = {
            "query": query,
            "retrieval_query": retrieval_query,
            "action_index": action_index,
            "action": action.name,
            "action_description": action.description,
            "top_k": action.top_k,
            "expected_sources": expected_sources,
            "expected_topics": expected_topics,
            "expected_points": expected_points,
            "source_hit": round(source_score, 4),
            "topic_hit": round(topic_score, 4),
            "point_recall": round(point_score, 4),
            "reward": round(reward, 4),
            "retrieved_titles": [item.get("title", "") for item in results],
            "retrieved_paths": [item.get("path", "") for item in results],
            "retrieved_topics": [item.get("topics", []) for item in results],
            "results": results,
        }
        return features_for_query(query), reward, True, info


def build_examples_from_rag_eval(path: Path) -> list[dict[str, Any]]:
    """从 RAG 评测文件中构造强化学习样本，只保留有 query/task 的行。"""
    examples = read_jsonl(path)
    return [row for row in examples if str(row.get("query") or row.get("task") or "").strip()]
