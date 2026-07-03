from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.retrieval.knowledge_store import search_knowledge
from backend.settings import ROOT


AGENT_ACTION_NAMES: tuple[str, ...] = (
    "read_file",
    "search_project_docs",
    "rerank_evidence",
    "compress_context",
    "second_search",
    "generate_answer",
)

_SEARCH_CACHE: dict[tuple[str, int], list[dict[str, Any]]] = {}


def cached_search_knowledge(query: str, top_k: int) -> list[dict[str, Any]]:
    key = (query, int(top_k))
    if key not in _SEARCH_CACHE:
        _SEARCH_CACHE[key] = search_knowledge(query, top_k=top_k)
    return list(_SEARCH_CACHE[key])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def extract_candidate_paths(text: str) -> list[str]:
    matches = re.findall(
        r"([A-Za-z]:\\[^\s\"']+|(?:[\w.\-_/]+/)+[\w.\-_/]+|[\w.\-_/]+\.(?:py|md|txt|json|jsonl|yaml|yml|toml|csv|log))",
        text or "",
    )
    seen: list[str] = []
    for item in matches:
        value = item.strip().strip(".,;:，。；：")
        if value and value not in seen:
            seen.append(value)
    return seen


def point_recall(text: str, expected_points: list[str]) -> float:
    if not expected_points:
        return 1.0
    lowered = normalize_text(text)
    hits = sum(1 for point in expected_points if normalize_text(point) in lowered)
    return hits / max(len(expected_points), 1)


def token_overlap_score(query: str, text: str, expected_points: list[str] | None = None) -> float:
    expected_points = expected_points or []
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", query.lower()))
    tokens.update(normalize_text(point) for point in expected_points if point)
    haystack = normalize_text(text)
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token and token in haystack) / len(tokens)


def _safe_read_text(path_text: str, max_chars: int = 4000) -> tuple[str, str]:
    candidate = Path(path_text.strip().strip("\"'"))
    if not candidate.is_absolute():
        candidate = (ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not (ROOT.resolve() == candidate or ROOT.resolve() in candidate.parents):
        return "", f"blocked_outside_workspace:{candidate}"
    if not candidate.exists() or not candidate.is_file():
        return "", f"missing:{candidate}"
    if candidate.suffix.lower() not in {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".log", ".yml", ".yaml"}:
        return "", f"unsupported_suffix:{candidate.suffix}"
    return candidate.read_text(encoding="utf-8", errors="ignore")[:max_chars], str(candidate)


@dataclass
class AgentStep:
    action: str
    reward: float
    done: bool
    observation: str


class AgentWorkflowEnv:
    """A small multi-step Agent environment for PPO training.

    The environment models the Agent workflow instead of only selecting one
    retrieval action: read local files, search RAG evidence, rerank, compress
    context, search again, and finally answer.
    """

    def __init__(self, examples: list[dict[str, Any]], *, max_steps: int = 6) -> None:
        if not examples:
            raise ValueError("AgentWorkflowEnv requires at least one example.")
        self.examples = examples
        self.max_steps = max_steps
        self.current_index = 0
        self.example: dict[str, Any] = {}
        self.query = ""
        self.expected_points: list[str] = []
        self.expected_tools: list[str] = []
        self.paths: list[str] = []
        self.step_count = 0
        self.evidence: list[dict[str, Any]] = []
        self.file_text = ""
        self.file_status = ""
        self.compressed_context = ""
        self.answer = ""
        self.actions_taken: list[str] = []
        self.reranked = False
        self.used_second_search = False
        self.done = False

    @property
    def action_size(self) -> int:
        return len(AGENT_ACTION_NAMES)

    @property
    def state_size(self) -> int:
        return len(self._features())

    def reset(self, index: int) -> list[float]:
        self.current_index = index
        self.example = self.examples[index]
        self.query = str(self.example.get("task") or self.example.get("query") or self.example.get("prompt") or "").strip()
        self.expected_points = [str(item) for item in self.example.get("expected_points", [])]
        self.expected_tools = [str(item) for item in self.example.get("expected_tools", [])]
        self.paths = extract_candidate_paths(self.query)
        self.step_count = 0
        self.evidence = []
        self.file_text = ""
        self.file_status = ""
        self.compressed_context = ""
        self.answer = ""
        self.actions_taken = []
        self.reranked = False
        self.used_second_search = False
        self.done = False
        return self._features()

    def context_text(self) -> str:
        snippets = " ".join(str(item.get("snippet", "")) for item in self.evidence)
        return " ".join(part for part in [self.file_text, self.compressed_context, snippets, self.answer] if part)

    def current_point_recall(self) -> float:
        return point_recall(self.context_text(), self.expected_points)

    def expected_tool_hit(self) -> float:
        if not self.expected_tools:
            return 1.0
        used = set(self.actions_taken)
        hits = 0
        for tool in self.expected_tools:
            if tool == "read_local_file" and "read_file" in used:
                hits += 1
            elif tool == "search_project_docs" and ("search_project_docs" in used or "second_search" in used):
                hits += 1
            elif tool in used:
                hits += 1
        return hits / len(self.expected_tools)

    def _features(self) -> list[float]:
        lowered = self.query.lower()
        context_len = len(self.context_text())
        repeated = len(self.actions_taken) - len(set(self.actions_taken))
        return [
            1.0 if "ppo" in lowered else 0.0,
            1.0 if "dpo" in lowered else 0.0,
            1.0 if "强化学习" in self.query or "reinforcement" in lowered else 0.0,
            1.0 if "轨迹" in self.query or "trajectory" in lowered else 0.0,
            1.0 if any(token in self.query for token in ["读取", "文件"]) or bool(self.paths) else 0.0,
            min(len(self.query) / 160.0, 1.0),
            self.step_count / max(self.max_steps, 1),
            1.0 if self.file_text else 0.0,
            min(len(self.evidence) / 8.0, 1.0),
            1.0 if self.reranked else 0.0,
            1.0 if self.compressed_context else 0.0,
            1.0 if self.used_second_search else 0.0,
            min(context_len / 5000.0, 1.0),
            self.current_point_recall(),
            self.expected_tool_hit(),
            min(repeated / 3.0, 1.0),
            max((self.max_steps - self.step_count) / max(self.max_steps, 1), 0.0),
        ]

    def oracle_action_index(self) -> int:
        if self.done:
            return AGENT_ACTION_NAMES.index("generate_answer")
        expects_read = "read_local_file" in self.expected_tools or bool(self.paths)
        expects_search = "search_project_docs" in self.expected_tools or not expects_read
        if expects_read and not self.file_text and "read_file" not in self.actions_taken:
            return AGENT_ACTION_NAMES.index("read_file")
        if expects_search and not self.evidence:
            return AGENT_ACTION_NAMES.index("search_project_docs")
        if self.evidence and not self.reranked:
            return AGENT_ACTION_NAMES.index("rerank_evidence")
        if (self.evidence or self.file_text) and not self.compressed_context:
            return AGENT_ACTION_NAMES.index("compress_context")
        if expects_search and not self.used_second_search and self.current_point_recall() < 0.95:
            return AGENT_ACTION_NAMES.index("second_search")
        return AGENT_ACTION_NAMES.index("generate_answer")

    def valid_action_indices(self) -> list[int]:
        valid: list[int] = []
        if self.paths and not self.file_text and "read_file" not in self.actions_taken:
            valid.append(AGENT_ACTION_NAMES.index("read_file"))
        if "search_project_docs" not in self.actions_taken:
            valid.append(AGENT_ACTION_NAMES.index("search_project_docs"))
        if self.evidence and not self.reranked:
            valid.append(AGENT_ACTION_NAMES.index("rerank_evidence"))
        if (self.evidence or self.file_text) and not self.compressed_context:
            valid.append(AGENT_ACTION_NAMES.index("compress_context"))
        if (self.evidence or "search_project_docs" in self.actions_taken) and not self.used_second_search and self.current_point_recall() < 0.95:
            valid.append(AGENT_ACTION_NAMES.index("second_search"))
        valid.append(AGENT_ACTION_NAMES.index("generate_answer"))
        return sorted(set(valid))

    def step(self, action_index: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        if self.done:
            return self._features(), 0.0, True, self.metrics()
        action_index = max(0, min(action_index, len(AGENT_ACTION_NAMES) - 1))
        action = AGENT_ACTION_NAMES[action_index]
        before_recall = self.current_point_recall()
        reward = -0.025
        observation = ""
        was_repeated = action in self.actions_taken
        invalid_action = action_index not in self.valid_action_indices()
        if was_repeated:
            reward -= 0.2
        if invalid_action and action != "generate_answer":
            reward -= 0.15
            action = "generate_answer"
            action_index = AGENT_ACTION_NAMES.index(action)
            observation = "invalid_action_fallback_to_generate"
        self.actions_taken.append(action)
        self.step_count += 1

        if action == "read_file":
            reward += self._read_file_reward()
            observation = self.file_status
        elif action == "search_project_docs":
            reward += self._search_reward(self.query, top_k=4)
            observation = f"evidence_count={len(self.evidence)}"
        elif action == "rerank_evidence":
            reward += self._rerank_reward()
            observation = "reranked" if self.reranked else "no_evidence"
        elif action == "compress_context":
            reward += self._compress_reward()
            observation = f"compressed_chars={len(self.compressed_context)}"
        elif action == "second_search":
            reward += self._second_search_reward()
            observation = f"evidence_count={len(self.evidence)}"
        elif action == "generate_answer":
            reward += self._generate_answer_reward()
            self.done = True
            observation = self.answer[:200]

        after_recall = self.current_point_recall()
        reward += 0.3 * max(after_recall - before_recall, 0.0)
        if self.step_count >= self.max_steps and not self.done:
            reward += self._generate_answer_reward() - 0.08
            self.done = True
        info = self.metrics() | {
            "action": action,
            "observation": observation,
            "step_reward": round(float(reward), 4),
        }
        return self._features(), float(reward), self.done, info

    def _read_file_reward(self) -> float:
        if not self.paths:
            self.file_status = "no_candidate_path"
            return -0.12
        text, status = _safe_read_text(self.paths[0])
        self.file_status = status
        if not text:
            return -0.12
        self.file_text = text
        reward = 0.18
        if "read_local_file" in self.expected_tools:
            reward += 0.22
        return reward + 0.2 * point_recall(text, self.expected_points)

    def _search_reward(self, query: str, *, top_k: int) -> float:
        results = cached_search_knowledge(query, top_k=top_k)
        self._merge_evidence(results)
        snippets = " ".join(str(item.get("snippet", "")) for item in results)
        reward = 0.08 if results else -0.08
        if "search_project_docs" in self.expected_tools:
            reward += 0.18
        return reward + 0.25 * point_recall(snippets, self.expected_points)

    def _merge_evidence(self, results: list[dict[str, Any]]) -> None:
        seen = {(str(item.get("title", "")), str(item.get("path", ""))) for item in self.evidence}
        for item in results:
            key = (str(item.get("title", "")), str(item.get("path", "")))
            if key not in seen:
                self.evidence.append(item)
                seen.add(key)
        self.evidence = self.evidence[:10]

    def _rerank_reward(self) -> float:
        if not self.evidence:
            return -0.08
        self.evidence.sort(
            key=lambda item: token_overlap_score(
                self.query,
                f"{item.get('title', '')} {item.get('snippet', '')} {' '.join(item.get('topics', []))}",
                self.expected_points,
            ),
            reverse=True,
        )
        self.reranked = True
        top_text = " ".join(str(item.get("snippet", "")) for item in self.evidence[:4])
        return 0.08 + 0.16 * point_recall(top_text, self.expected_points)

    def _compress_reward(self) -> float:
        source = self.context_text()
        if not source.strip():
            return -0.08
        sentences = re.split(r"(?<=[。！？.!?])\s+|\n+", source)
        ranked = sorted(sentences, key=lambda item: token_overlap_score(self.query, item, self.expected_points), reverse=True)
        compact = " ".join(item.strip() for item in ranked[:6] if item.strip())
        self.compressed_context = compact[:1200]
        density = point_recall(self.compressed_context, self.expected_points)
        length_bonus = 0.04 if len(source) > len(self.compressed_context) else 0.0
        return 0.08 + length_bonus + 0.18 * density

    def _second_search_reward(self) -> float:
        missing = [point for point in self.expected_points if normalize_text(point) not in normalize_text(self.context_text())]
        augmented = f"{self.query} {' '.join(missing)}".strip()
        before = len(self.evidence)
        self.used_second_search = True
        reward = self._search_reward(augmented, top_k=5)
        new_items = max(len(self.evidence) - before, 0)
        return reward + min(new_items * 0.03, 0.12)

    def _generate_answer_reward(self) -> float:
        context = self.context_text()
        if context.strip():
            evidence_titles = [str(item.get("title", "")) for item in self.evidence[:3] if item.get("title")]
            title_text = "；".join(evidence_titles)
            supported_points = [
                point
                for point in self.expected_points
                if normalize_text(point) in normalize_text(context)
            ]
            point_text = "、".join(supported_points) if supported_points else "当前问题"
            self.answer = (
                "基于已读取文件和检索证据，回答应围绕："
                + point_text
                + ("。可用证据包括：" + title_text if title_text else "。")
                + " "
                + self.compressed_context[:500]
            )
        else:
            self.answer = "当前缺少文件或检索证据，只能给出泛化回答。"
        answer_recall = point_recall(self.answer, self.expected_points)
        tool_hit = self.expected_tool_hit()
        evidence_score = min(len(self.evidence) / 4.0, 1.0)
        step_penalty = 0.015 * max(self.step_count - 3, 0)
        return 0.55 * answer_recall + 0.25 * tool_hit + 0.12 * evidence_score - step_penalty

    def metrics(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "answer_point_recall": round(point_recall(self.answer, self.expected_points), 4),
            "context_point_recall": round(self.current_point_recall(), 4),
            "expected_tool_hit": round(self.expected_tool_hit(), 4),
            "evidence_count": len(self.evidence),
            "used_tools": list(self.actions_taken),
            "expected_tools": list(self.expected_tools),
            "expected_points": list(self.expected_points),
            "compressed": bool(self.compressed_context),
            "used_second_search": self.used_second_search,
            "steps": self.step_count,
        }


def run_baseline_plan(example: dict[str, Any], *, max_steps: int = 6) -> dict[str, Any]:
    env = AgentWorkflowEnv([example], max_steps=max_steps)
    env.reset(0)
    total_reward = 0.0
    if "read_local_file" in env.expected_tools or env.paths:
        _, reward, _, _ = env.step(AGENT_ACTION_NAMES.index("read_file"))
    else:
        _, reward, _, _ = env.step(AGENT_ACTION_NAMES.index("search_project_docs"))
    total_reward += float(reward)
    if not env.done:
        _, reward, _, info = env.step(AGENT_ACTION_NAMES.index("generate_answer"))
        total_reward += float(reward)
    else:
        info = env.metrics()
    info["total_reward"] = round(total_reward, 4)
    info["policy"] = "baseline"
    return info


def average_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key) or 0.0) for row in rows) / len(rows), 4)


def safe_mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def discounted_returns(rewards: list[float], dones: list[bool], gamma: float) -> list[float]:
    returns: list[float] = []
    running = 0.0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        running = float(reward) + gamma * running * (0.0 if done else 1.0)
        returns.append(running)
    return list(reversed(returns))


def finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0
