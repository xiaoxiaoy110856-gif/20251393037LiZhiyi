from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LOCAL_RAG_FORCE_FALLBACK", "1")
os.environ.setdefault("LOCAL_RAG_FAST_FALLBACK", "1")

from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.retrieval.rl_env import ACTIONS, RetrievalRLEnv, build_examples_from_rag_eval

try:
    import numpy as np
except Exception as error:  # pragma: no cover
    raise SystemExit(f"NumPy is required for LinUCB retrieval policy training: {error}")


DEFAULT_REWARD_WEIGHTS = {
    "source_hit": 0.5,
    "topic_hit": 0.3,
    "point_recall": 0.2,
}


class LinUCBPolicy:
    """LinUCB 上下文 bandit 策略：用线性奖励预测和不确定性 bonus 选择检索动作。"""
    def __init__(self, theta: np.ndarray, a_inv: np.ndarray, alpha: float) -> None:
        self.theta = theta
        self.a_inv = a_inv
        self.alpha = alpha

    def scores(self, state: list[float]) -> list[float]:
        # 核心5/LinUCB：score = 预测奖励 + 不确定性 bonus，适合单步检索策略选择。
        x = np.asarray(state, dtype=np.float64)
        values: list[float] = []
        for action_index in range(self.theta.shape[0]):
            mean = float(self.theta[action_index] @ x)
            uncertainty = float(np.sqrt(max(x @ self.a_inv[action_index] @ x, 0.0)))
            values.append(mean + self.alpha * uncertainty)
        return values

    def choose(self, state: list[float]) -> tuple[int, list[float]]:
        values = self.scores(state)
        return int(max(range(len(values)), key=lambda index: values[index])), [round(value, 4) for value in values]


def parse_args() -> argparse.Namespace:
    """解析 LinUCB 训练参数，支持 online 和 offline 两种模式。"""
    parser = argparse.ArgumentParser(description="Train a LinUCB contextual-bandit retrieval policy.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs" / "retrieval_policy_dpo_torch" / "dpo_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_linucb")
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--episodes", type=int, default=420)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.12)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["source_hit"])
    parser.add_argument("--topic-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["topic_hit"])
    parser.add_argument("--point-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["point_recall"])
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, Any]]:
    """读取 DPO 偏好对，用于构造离线 reward row。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if {"example_index", "state", "chosen_action_index", "rejected_action_index"}.issubset(row):
            rows.append(row)
    return rows


def build_reward_rows(pairs: list[dict[str, Any]], action_size: int, margin: float) -> list[dict[str, Any]]:
    """把 chosen/rejected 偏好对转成每个动作的 reward vector，供 LinUCB 离线拟合。"""
    by_example: dict[int, dict[str, Any]] = {}
    for pair in pairs:
        example_index = int(pair["example_index"])
        row = by_example.setdefault(
            example_index,
            {
                "example_index": example_index,
                "query": str(pair.get("query") or ""),
                "state": [float(value) for value in pair["state"]],
                "rewards": {},
            },
        )
        row["rewards"][int(pair["chosen_action_index"])] = float(pair["chosen_reward"])
        row["rewards"][int(pair["rejected_action_index"])] = float(pair["rejected_reward"])

    reward_rows: list[dict[str, Any]] = []
    for row in sorted(by_example.values(), key=lambda item: item["example_index"]):
        observed = dict(row["rewards"])
        best_reward = max(observed.values()) if observed else 0.0
        filled = [
            float(observed.get(action_index, max(best_reward - margin / 2.0, 0.0)))
            for action_index in range(action_size)
        ]
        row["reward_vector"] = filled
        row["observed_action_count"] = len(observed)
        reward_rows.append(row)
    return reward_rows


def train_linucb(
    rows: list[dict[str, Any]],
    *,
    action_size: int,
    state_size: int,
    epochs: int,
    alpha: float,
    l2: float,
    seed: int,
) -> tuple[LinUCBPolicy, list[dict[str, Any]]]:
    # 核心5/LinUCB：用 DPO reward rows 做离线拟合，也可作为 online contextual bandit 快速更新。
    random.seed(seed)
    a_mats = np.stack([np.eye(state_size, dtype=np.float64) * l2 for _ in range(action_size)])
    b_vecs = np.zeros((action_size, state_size), dtype=np.float64)
    observations = [
        (np.asarray(row["state"], dtype=np.float64), action_index, float(reward))
        for row in rows
        for action_index, reward in enumerate(row["reward_vector"])
    ]
    trace: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        random.shuffle(observations)
        for state, action_index, reward in observations:
            a_mats[action_index] += np.outer(state, state)
            b_vecs[action_index] += reward * state

        a_inv = np.linalg.inv(a_mats)
        theta = np.einsum("aij,aj->ai", a_inv, b_vecs)
        squared_errors = []
        for state, action_index, reward in observations:
            prediction = float(theta[action_index] @ state)
            squared_errors.append((prediction - reward) ** 2)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            trace.append(
                {
                    "epoch": epoch,
                    "loss": round(float(np.mean(squared_errors)), 6),
                    "alpha": alpha,
                    "l2": l2,
                    "algorithm": "linucb",
                }
            )

    a_inv = np.linalg.inv(a_mats)
    theta = np.einsum("aij,aj->ai", a_inv, b_vecs)
    return LinUCBPolicy(theta=theta, a_inv=a_inv, alpha=alpha), trace


def train_linucb_online(
    env: RetrievalRLEnv,
    *,
    episodes: int,
    alpha: float,
    l2: float,
    epsilon: float,
    seed: int,
) -> tuple[LinUCBPolicy, list[dict[str, Any]]]:
    random.seed(seed)
    action_size = env.action_size
    state_size = env.state_size
    a_mats = np.stack([np.eye(state_size, dtype=np.float64) * l2 for _ in range(action_size)])
    b_vecs = np.zeros((action_size, state_size), dtype=np.float64)
    trace: list[dict[str, Any]] = []

    for episode in range(1, episodes + 1):
        a_inv = np.linalg.inv(a_mats)
        theta = np.einsum("aij,aj->ai", a_inv, b_vecs)
        policy = LinUCBPolicy(theta=theta, a_inv=a_inv, alpha=alpha)

        example_index = random.randrange(len(env.examples))
        state = env.reset(example_index)
        warmup = episode <= action_size * 3
        if warmup or random.random() < epsilon:
            action_index = random.randrange(action_size)
            scores = policy.scores(state)
        else:
            action_index, scores = policy.choose(state)
        _, reward, _, info = env.step(action_index)

        x = np.asarray(state, dtype=np.float64)
        a_mats[action_index] += np.outer(x, x)
        b_vecs[action_index] += float(reward) * x

        trace.append(
            {
                "episode": episode,
                "query": info["query"],
                "action": info["action"],
                "reward": round(float(reward), 4),
                "score": round(float(scores[action_index]), 6),
                "source_hit": info["source_hit"],
                "topic_hit": info["topic_hit"],
                "point_recall": info["point_recall"],
                "algorithm": "linucb",
            }
        )

    a_inv = np.linalg.inv(a_mats)
    theta = np.einsum("aij,aj->ai", a_inv, b_vecs)
    return LinUCBPolicy(theta=theta, a_inv=a_inv, alpha=alpha), trace


def evaluate_policy(env: RetrievalRLEnv, policy: LinUCBPolicy) -> dict[str, Any]:
    """评测 LinUCB 策略在每个问题上的动作选择和平均指标。"""
    episodes: list[dict[str, Any]] = []
    totals = {"reward": 0.0, "source_hit": 0.0, "topic_hit": 0.0, "point_recall": 0.0}
    for index, example in enumerate(env.examples):
        state = env.reset(index)
        action_index, scores = policy.choose(state)
        _, reward, _, info = env.step(action_index)
        totals["reward"] += reward
        totals["source_hit"] += info["source_hit"]
        totals["topic_hit"] += info["topic_hit"]
        totals["point_recall"] += info["point_recall"]
        episodes.append(
            {
                "index": index,
                "query": example.get("query") or example.get("task"),
                "chosen_action": info["action"],
                "reward": round(reward, 4),
                "source_hit": info["source_hit"],
                "topic_hit": info["topic_hit"],
                "point_recall": info["point_recall"],
                "scores": {ACTIONS[i].name: scores[i] for i in range(min(len(ACTIONS), len(scores)))},
                "retrieved_titles": info.get("retrieved_titles", []),
            }
        )
    count = max(len(env.examples), 1)
    return {
        "average_reward": round(totals["reward"] / count, 4),
        "average_source_hit": round(totals["source_hit"] / count, 4),
        "average_topic_hit": round(totals["topic_hit"] / count, 4),
        "average_point_recall": round(totals["point_recall"] / count, 4),
        "episodes": episodes,
    }


def evaluate_fixed_action(env: RetrievalRLEnv, action_index: int) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    totals = {"reward": 0.0, "source_hit": 0.0, "topic_hit": 0.0, "point_recall": 0.0}
    for index, example in enumerate(env.examples):
        env.reset(index)
        _, reward, _, info = env.step(action_index)
        totals["reward"] += reward
        totals["source_hit"] += info["source_hit"]
        totals["topic_hit"] += info["topic_hit"]
        totals["point_recall"] += info["point_recall"]
        episodes.append(
            {
                "index": index,
                "query": example.get("query") or example.get("task"),
                "chosen_action": info["action"],
                "reward": round(reward, 4),
                "source_hit": info["source_hit"],
                "topic_hit": info["topic_hit"],
                "point_recall": info["point_recall"],
            }
        )
    count = max(len(env.examples), 1)
    return {
        "average_reward": round(totals["reward"] / count, 4),
        "average_source_hit": round(totals["source_hit"] / count, 4),
        "average_topic_hit": round(totals["topic_hit"] / count, 4),
        "average_point_recall": round(totals["point_recall"] / count, 4),
        "episodes": episodes,
    }


def main() -> None:
    """命令行入口：训练 LinUCB 并保存 json checkpoint、trace 和 evaluation。"""
    args = parse_args()
    reward_weights = {
        "source_hit": args.source_weight,
        "topic_hit": args.topic_weight,
        "point_recall": args.point_weight,
    }
    examples = build_examples_from_rag_eval(args.data)
    pairs = read_pairs(args.pairs)
    if not examples:
        raise SystemExit(f"No examples found in {args.data}")
    if not pairs:
        raise SystemExit(f"No preference pairs found in {args.pairs}")

    env = RetrievalRLEnv(examples, reward_weights=reward_weights)
    rows: list[dict[str, Any]] = []
    if args.mode == "online":
        policy, trace = train_linucb_online(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            l2=args.l2,
            epsilon=args.epsilon,
            seed=args.seed,
        )
    else:
        rows = build_reward_rows(pairs, env.action_size, args.margin)
        policy, trace = train_linucb(
            rows,
            action_size=env.action_size,
            state_size=env.state_size,
            epochs=args.epochs,
            alpha=args.alpha,
            l2=args.l2,
            seed=args.seed,
        )
    trained_policy = evaluate_policy(env, policy)
    baseline_policy = evaluate_fixed_action(env, 0)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "algorithm": "linucb",
        "state_size": env.state_size,
        "action_size": env.action_size,
        "actions": [action.__dict__ for action in ACTIONS],
        "theta": policy.theta.tolist(),
        "a_inv": policy.a_inv.tolist(),
        "alpha": args.alpha,
        "l2": args.l2,
        "mode": args.mode,
        "episodes": args.episodes if args.mode == "online" else None,
        "preference_pairs": len(pairs),
        "reward_rows": len(rows),
        "reward_weights": reward_weights,
    }
    (args.output / "retrieval_policy_linucb.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        (args.output / "reward_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "training_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation = {
        "trained_policy": trained_policy,
        "baseline_policy": baseline_policy,
        "preference_pair_count": len(pairs),
        "reward_row_count": len(rows),
        "mode": args.mode,
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "ok": True,
        "output": str(args.output),
        "preference_pairs": len(pairs),
        "reward_rows": len(rows),
        "baseline_average_reward": baseline_policy["average_reward"],
        "linucb_average_reward": trained_policy["average_reward"],
        "baseline_average_source_hit": baseline_policy["average_source_hit"],
        "linucb_average_source_hit": trained_policy["average_source_hit"],
        "baseline_average_topic_hit": baseline_policy["average_topic_hit"],
        "linucb_average_topic_hit": trained_policy["average_topic_hit"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
