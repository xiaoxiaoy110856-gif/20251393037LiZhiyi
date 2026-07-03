from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from backend.bootstrap import ensure_python_paths
from backend.retrieval.rl_env import ACTIONS, compose_retrieval_query, features_for_query
from backend.settings import ROOT, get_retrieval_policy_path, retrieval_policy_enabled

ensure_python_paths()


# 核心5上线顺序：优先加载效果更好的已训练策略（LinUCB/DDQN/DPO/ORPO/PPO），都不可用时回退 baseline。
def _candidate_policy_paths() -> list[Path]:
    configured = get_retrieval_policy_path()
    paths: list[Path] = []
    if configured:
        paths.append(Path(configured))
    paths.extend(
        [
            ROOT / "outputs" / "retrieval_policy_linucb" / "retrieval_policy_linucb.json",
            ROOT / "outputs" / "retrieval_policy_dueling_ddqn" / "retrieval_policy_dueling_ddqn.pt",
            ROOT / "outputs" / "retrieval_policy_dpo_torch" / "retrieval_policy_dpo.pt",
            ROOT / "outputs" / "retrieval_policy_dpo_torch" / "retrieval_policy_lookup.json",
            ROOT / "outputs" / "retrieval_policy_orpo_torch" / "retrieval_policy_orpo.pt",
            ROOT / "outputs" / "retrieval_policy_ppo_torch_60" / "retrieval_policy_lookup.json",
            ROOT / "outputs" / "retrieval_policy_ppo_torch_60" / "retrieval_policy_ppo.pt",
            ROOT / "outputs" / "retrieval_policy_ppo_torch" / "retrieval_policy_lookup.json",
            ROOT / "outputs" / "retrieval_policy_ppo_torch" / "retrieval_policy_ppo.pt",
            ROOT / "outputs" / "retrieval_policy_ppo" / "retrieval_policy_ppo.json",
            ROOT / "outputs" / "retrieval_policy_ppo" / "retrieval_policy_ppo.pt",
        ]
    )
    sweep_summary = ROOT / "outputs" / "retrieval_reward_sweep" / "reward_sweep_summary.json"
    if sweep_summary.exists():
        try:
            best = json.loads(sweep_summary.read_text(encoding="utf-8")).get("best", {})
            best_output = Path(str(best.get("output") or ""))
            if best_output.exists():
                paths.append(best_output / "retrieval_policy_lookup.json")
                paths.append(best_output / "retrieval_policy_dqn.pt")
        except Exception:
            pass
    paths.append(ROOT / "outputs" / "retrieval_policy" / "retrieval_policy_dqn.pt")
    sweep_root = ROOT / "outputs" / "retrieval_reward_sweep"
    if sweep_root.exists():
        paths.extend(sorted(sweep_root.glob("*/retrieval_policy_dqn.pt"), key=lambda path: path.stat().st_mtime, reverse=True))
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        resolved = str(path.expanduser().resolve())
        if resolved not in seen:
            deduped.append(Path(resolved))
            seen.add(resolved)
    return deduped


def _torch_load(path: Path) -> dict[str, Any]:
    """加载 PyTorch checkpoint，并兼容不同 torch 版本的 weights_only 参数。"""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


class _DQNPolicy:
    """普通 DQN checkpoint 的运行时包装，用于旧版检索策略回放。"""
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        import torch.nn as nn

        state_size = int(checkpoint.get("state_size") or len(features_for_query("dummy")))
        action_size = int(checkpoint.get("action_size") or len(ACTIONS))
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

    def choose(self, state: list[float]) -> tuple[int, list[float]]:
        import torch

        with torch.no_grad():
            values = self.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0]
        return int(torch.argmax(values).item()), [round(float(item), 4) for item in values.tolist()]


# 核心5：Dueling DDQN checkpoint 的运行时包装，是神经网络检索策略对照组。
class _DuelingDQNPolicy:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        import torch.nn as nn

        state_size = int(checkpoint.get("state_size") or len(features_for_query("dummy")))
        action_size = int(checkpoint.get("action_size") or len(ACTIONS))

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(state_size, 96),
                    nn.ReLU(),
                    nn.Linear(96, 96),
                    nn.ReLU(),
                )
                self.value = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))
                self.advantage = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, action_size))

            def forward(self, states: Any) -> Any:
                hidden = self.trunk(states)
                value = self.value(hidden)
                advantage = self.advantage(hidden)
                return value + advantage - advantage.mean(dim=1, keepdim=True)

        self.net = Net()
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()

    def choose(self, state: list[float]) -> tuple[int, list[float]]:
        import torch

        with torch.no_grad():
            values = self.net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0]
        return int(torch.argmax(values).item()), [round(float(item), 4) for item in values.tolist()]


# 核心5：PPO 类 actor checkpoint 的运行时包装。DPO/ORPO 训练后也复用同一 actor 网络形状。
class _PPOPolicy:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        import torch.nn as nn

        state_size = int(checkpoint.get("state_size") or len(features_for_query("dummy")))
        action_size = int(checkpoint.get("action_size") or len(ACTIONS))
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_size)
        state_dict = checkpoint.get("actor_critic_state_dict") or checkpoint.get("state_dict")
        self.load_state_dict(state_dict)
        self.eval()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        own_state = {"trunk." + key: value for key, value in self.trunk.state_dict().items()}
        own_state.update({"actor." + key: value for key, value in self.actor.state_dict().items()})
        mapped = {key: value for key, value in state_dict.items() if key in own_state}
        self.trunk.load_state_dict({key.removeprefix("trunk."): value for key, value in mapped.items() if key.startswith("trunk.")}, strict=False)
        self.actor.load_state_dict({key.removeprefix("actor."): value for key, value in mapped.items() if key.startswith("actor.")}, strict=False)

    def eval(self) -> None:
        self.trunk.eval()
        self.actor.eval()

    def choose(self, state: list[float]) -> tuple[int, list[float]]:
        import torch

        with torch.no_grad():
            hidden = self.trunk(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            logits = self.actor(hidden)[0]
            probs = torch.softmax(logits, dim=-1)
        return int(torch.argmax(probs).item()), [round(float(item), 4) for item in probs.tolist()]


class _LinearSoftmaxPolicy:
    """无 torch 的线性 softmax 策略包装，用于轻量 PPO 结果。"""
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.weights = [[float(value) for value in row] for row in checkpoint.get("weights", [])]
        self.bias = [float(value) for value in checkpoint.get("bias", [])]

    @staticmethod
    def _softmax(logits: list[float]) -> list[float]:
        import math

        max_logit = max(logits) if logits else 0.0
        exps = [math.exp(value - max_logit) for value in logits]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def choose(self, state: list[float]) -> tuple[int, list[float]]:
        logits = [
            sum(weight * feature for weight, feature in zip(row, state)) + self.bias[index]
            for index, row in enumerate(self.weights)
        ]
        probs = self._softmax(logits)
        if not probs:
            return 0, []
        action_index = max(range(len(probs)), key=lambda index: probs[index])
        return action_index, [round(float(item), 4) for item in probs]


# 核心5：LinUCB 的运行时包装。当前检索任务更像 contextual bandit，因此这是推荐上线策略。
class _LinUCBPolicy:
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.theta = [[float(value) for value in row] for row in checkpoint.get("theta", [])]
        self.a_inv = [
            [[float(value) for value in inner] for inner in matrix]
            for matrix in checkpoint.get("a_inv", [])
        ]
        self.alpha = float(checkpoint.get("alpha") or 0.0)

    def choose(self, state: list[float]) -> tuple[int, list[float]]:
        import math

        scores: list[float] = []
        for action_index, theta in enumerate(self.theta):
            mean = sum(weight * value for weight, value in zip(theta, state))
            matrix = self.a_inv[action_index] if action_index < len(self.a_inv) else []
            uncertainty_sq = 0.0
            for row_index, row in enumerate(matrix):
                if row_index >= len(state):
                    continue
                uncertainty_sq += state[row_index] * sum(row[col_index] * state[col_index] for col_index in range(min(len(row), len(state))))
            scores.append(mean + self.alpha * math.sqrt(max(uncertainty_sq, 0.0)))
        if not scores:
            return 0, []
        action_index = max(range(len(scores)), key=lambda index: scores[index])
        return action_index, [round(float(item), 4) for item in scores]


class _LookupPolicy:
    """查表策略包装：根据历史样本 query 的 token overlap 选择动作，主要用于导出的 lookup 策略。"""
    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self.rules = [
            {
                "query": str(rule.get("query") or ""),
                "action_index": int(rule.get("action_index") or 0),
                "tokens": set(str(rule.get("query") or "").lower().replace("，", " ").replace("？", " ").split()),
            }
            for rule in checkpoint.get("rules", [])
            if str(rule.get("query") or "").strip()
        ]

    def choose(self, state: list[float], query: str = "") -> tuple[int, list[float]]:
        query_tokens = set((query or "").lower().replace("，", " ").replace("？", " ").split())
        best_score = 0.0
        best_action = 0
        for rule in self.rules:
            tokens = rule["tokens"]
            if not tokens:
                continue
            overlap = len(query_tokens & tokens) / max(len(tokens), 1)
            if overlap > best_score:
                best_score = overlap
                best_action = int(rule["action_index"])
        scores = [0.0 for _ in ACTIONS]
        if scores:
            scores[max(0, min(best_action, len(scores) - 1))] = round(best_score, 4)
        return best_action if best_score >= 0.25 else 0, scores


@lru_cache(maxsize=4)
def _load_policy(path_text: str) -> tuple[str, Any] | None:
    """根据 checkpoint 内容判断算法类型，并构造对应的运行时策略对象。"""
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        checkpoint = _json_load(path) if path.suffix.lower() == ".json" else _torch_load(path)
    except Exception:
        return None
    algorithm = str(checkpoint.get("algorithm") or checkpoint.get("run_type") or "").lower()
    if "lookup" in algorithm or "rules" in checkpoint:
        return "lookup", _LookupPolicy(checkpoint)
    if "linucb" in algorithm and "theta" in checkpoint:
        return "linucb", _LinUCBPolicy(checkpoint)
    if "ppo_linear" in algorithm or "weights" in checkpoint:
        return "ppo-linear", _LinearSoftmaxPolicy(checkpoint)
    if "dueling" in algorithm and "state_dict" in checkpoint:
        return "dueling-ddqn", _DuelingDQNPolicy(checkpoint)
    if "orpo" in algorithm and "actor_critic_state_dict" in checkpoint:
        return "orpo", _PPOPolicy(checkpoint)
    if "dpo" in algorithm and "actor_critic_state_dict" in checkpoint:
        return "dpo", _PPOPolicy(checkpoint)
    if "ppo" in algorithm or "actor_critic_state_dict" in checkpoint:
        return "ppo", _PPOPolicy(checkpoint)
    if "state_dict" in checkpoint:
        return "dqn", _DQNPolicy(checkpoint)
    return None


def choose_retrieval_action(query: str, requested_top_k: int = 4) -> dict[str, Any]:
    # 核心5：线上策略决策点。search_project_docs() 会先调用这里，再查询本地知识库。
    if not retrieval_policy_enabled():
        action = ACTIONS[0]
        return {
            "enabled": False,
            "available": False,
            "algorithm": "baseline",
            "checkpoint": "",
            "action_index": 0,
            "action": action.name,
            "action_description": action.description,
            "original_query": query,
            "retrieval_query": compose_retrieval_query(query, action),
            "top_k": int(requested_top_k or action.top_k),
            "scores": {},
        }

    for path in _candidate_policy_paths():
        loaded = _load_policy(str(path))
        if not loaded:
            continue
        algorithm, policy = loaded
        state = features_for_query(query)
        try:
            action_index, raw_scores = policy.choose(state, query=query)
        except TypeError:
            action_index, raw_scores = policy.choose(state)
        action_index = max(0, min(action_index, len(ACTIONS) - 1))
        action = ACTIONS[action_index]
        return {
            "enabled": True,
            "available": True,
            "algorithm": algorithm,
            "checkpoint": str(path),
            "action_index": action_index,
            "action": action.name,
            "action_description": action.description,
            "original_query": query,
            "retrieval_query": compose_retrieval_query(query, action),
            "top_k": action.top_k,
            "scores": {ACTIONS[index].name: raw_scores[index] for index in range(min(len(ACTIONS), len(raw_scores)))},
        }

    action = ACTIONS[0]
    return {
        "enabled": True,
        "available": False,
        "algorithm": "baseline",
        "checkpoint": "",
        "action_index": 0,
        "action": action.name,
        "action_description": action.description,
        "original_query": query,
        "retrieval_query": compose_retrieval_query(query, action),
        "top_k": int(requested_top_k or action.top_k),
        "scores": {},
    }
