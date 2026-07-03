from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.storage.db import create_retrieval_rl_run, fail_retrieval_rl_run, finalize_retrieval_rl_run, save_training_run
from backend.retrieval.rl_env import ACTIONS, RetrievalRLEnv, build_examples_from_rag_eval


DEFAULT_REWARD_WEIGHTS = {
    "source_hit": 0.5,
    "topic_hit": 0.3,
    "point_recall": 0.2,
}


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits) if logits else 0.0
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


class LinearSoftmaxPolicy:
    def __init__(self, state_size: int, action_size: int, seed: int) -> None:
        rng = random.Random(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.weights = [[rng.uniform(-0.01, 0.01) for _ in range(state_size)] for _ in range(action_size)]
        self.bias = [0.0 for _ in range(action_size)]
        self.value_weights = [0.0 for _ in range(state_size)]
        self.value_bias = 0.0

    def probs(self, state: list[float]) -> list[float]:
        logits = [
            sum(weight * feature for weight, feature in zip(row, state)) + self.bias[action_index]
            for action_index, row in enumerate(self.weights)
        ]
        return softmax(logits)

    def value(self, state: list[float]) -> float:
        return sum(weight * feature for weight, feature in zip(self.value_weights, state)) + self.value_bias

    def sample_action(self, state: list[float]) -> tuple[int, float]:
        probabilities = self.probs(state)
        threshold = random.random()
        running = 0.0
        for index, probability in enumerate(probabilities):
            running += probability
            if threshold <= running:
                return index, max(probability, 1e-8)
        return len(probabilities) - 1, max(probabilities[-1], 1e-8)

    def best_action(self, state: list[float]) -> int:
        probabilities = self.probs(state)
        return max(range(len(probabilities)), key=lambda index: probabilities[index])

    def update_policy(self, state: list[float], action_index: int, advantage: float, old_probability: float, lr: float, clip_range: float) -> None:
        probabilities = self.probs(state)
        new_probability = max(probabilities[action_index], 1e-8)
        ratio = new_probability / max(old_probability, 1e-8)
        clipped = (advantage >= 0 and ratio > 1.0 + clip_range) or (advantage < 0 and ratio < 1.0 - clip_range)
        if clipped:
            return
        for index in range(self.action_size):
            grad_logit = (1.0 if index == action_index else 0.0) - probabilities[index]
            step = lr * advantage * grad_logit
            self.bias[index] += step
            for feature_index, feature in enumerate(state):
                self.weights[index][feature_index] += step * feature

    def update_value(self, state: list[float], reward: float, lr: float) -> None:
        error = reward - self.value(state)
        self.value_bias += lr * error
        for index, feature in enumerate(state):
            self.value_weights[index] += lr * error * feature

    def checkpoint(self) -> dict[str, Any]:
        return {
            "algorithm": "ppo_linear",
            "weights": self.weights,
            "bias": self.bias,
            "value_weights": self.value_weights,
            "value_bias": self.value_bias,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "actions": [action.__dict__ for action in ACTIONS],
        }


def evaluate_policy(env: RetrievalRLEnv, model: LinearSoftmaxPolicy) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    totals = {"reward": 0.0, "source_hit": 0.0, "topic_hit": 0.0, "point_recall": 0.0}
    for index, example in enumerate(env.examples):
        state = env.reset(index)
        action_index = model.best_action(state)
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
                "retrieved_titles": info["retrieved_titles"],
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


def train_policy(
    *,
    data_path: Path,
    output_dir: Path,
    updates: int,
    rollout_size: int,
    ppo_epochs: int,
    lr: float,
    value_lr: float,
    clip_range: float,
    seed: int,
    reward_weights: dict[str, float] | None = None,
    record_db: bool = True,
) -> dict[str, Any]:
    random.seed(seed)
    examples = build_examples_from_rag_eval(data_path)
    if not examples:
        raise RuntimeError(f"No retrieval RL samples found in {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_weights = dict(DEFAULT_REWARD_WEIGHTS | (reward_weights or {}))
    run_name = output_dir.name
    retrieval_run_id = None
    if record_db:
        retrieval_run_id = create_retrieval_rl_run(run_name=run_name, data_path=str(data_path), output_path=str(output_dir), status="started")

    try:
        env = RetrievalRLEnv(examples, reward_weights=reward_weights)
        policy = LinearSoftmaxPolicy(env.state_size, env.action_size, seed=seed)
        training_trace: list[dict[str, Any]] = []
        for update_index in range(1, updates + 1):
            rollout: list[dict[str, Any]] = []
            for _ in range(rollout_size):
                index = random.randrange(len(examples))
                state = env.reset(index)
                action_index, old_probability = policy.sample_action(state)
                _, reward, _, info = env.step(action_index)
                advantage = float(reward) - policy.value(state)
                rollout.append(
                    {
                        "state": state,
                        "action_index": action_index,
                        "old_probability": old_probability,
                        "reward": float(reward),
                        "advantage": advantage,
                        "info": info,
                    }
                )
            for _ in range(ppo_epochs):
                random.shuffle(rollout)
                for item in rollout:
                    policy.update_policy(item["state"], item["action_index"], item["advantage"], item["old_probability"], lr, clip_range)
                    policy.update_value(item["state"], item["reward"], value_lr)
            avg_reward = sum(float(item["reward"]) for item in rollout) / max(len(rollout), 1)
            best = max(rollout, key=lambda item: float(item["reward"]))
            info = best["info"]
            training_trace.append(
                {
                    "episode": update_index,
                    "query": info["query"],
                    "action": info["action"],
                    "reward": round(avg_reward, 4),
                    "loss": None,
                    "source_hit": info["source_hit"],
                    "topic_hit": info["topic_hit"],
                    "point_recall": info["point_recall"],
                    "reward_weights": reward_weights,
                    "algorithm": "ppo-linear",
                }
            )

        evaluation = evaluate_policy(env, policy)
        baseline = evaluate_fixed_action(env, action_index=0)
        metrics = {
            "algorithm": "ppo-linear",
            "updates": updates,
            "rollout_size": rollout_size,
            "reward_weights": reward_weights,
            "trained_average_reward": evaluation["average_reward"],
            "trained_average_source_hit": evaluation["average_source_hit"],
            "trained_average_topic_hit": evaluation["average_topic_hit"],
            "trained_average_point_recall": evaluation["average_point_recall"],
            "baseline_average_reward": baseline["average_reward"],
            "baseline_average_source_hit": baseline["average_source_hit"],
            "baseline_average_topic_hit": baseline["average_topic_hit"],
            "baseline_average_point_recall": baseline["average_point_recall"],
            "reward_gain_vs_baseline": round(evaluation["average_reward"] - baseline["average_reward"], 4),
            "source_hit_gain_vs_baseline": round(evaluation["average_source_hit"] - baseline["average_source_hit"], 4),
            "topic_hit_gain_vs_baseline": round(evaluation["average_topic_hit"] - baseline["average_topic_hit"], 4),
            "point_recall_gain_vs_baseline": round(evaluation["average_point_recall"] - baseline["average_point_recall"], 4),
        }
        checkpoint = policy.checkpoint()
        checkpoint["data_path"] = str(data_path)
        checkpoint["reward_weights"] = reward_weights
        (output_dir / "retrieval_policy_ppo.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "training_trace.json").write_text(json.dumps(training_trace, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "evaluation.json").write_text(
            json.dumps({"trained_policy": evaluation, "baseline_policy": baseline}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "reward_weights.json").write_text(json.dumps(reward_weights, ensure_ascii=False, indent=2), encoding="utf-8")
        if record_db and retrieval_run_id is not None:
            finalize_retrieval_rl_run(
                run_id=retrieval_run_id,
                status="completed",
                metrics=metrics,
                evaluation={"trained_policy": evaluation, "baseline_policy": baseline},
                trace=training_trace,
            )
            save_training_run(
                run_type="retrieval_policy_ppo_linear",
                model_path=str(output_dir / "retrieval_policy_ppo.json"),
                data_path=str(data_path),
                output_path=str(output_dir),
                status="completed",
                notes="Torch-free PPO-style retrieval policy completed.",
                metrics=metrics,
            )
        return {
            "ok": True,
            "metrics": metrics,
            "output": str(output_dir),
            "evaluation": {"trained_policy": evaluation, "baseline_policy": baseline},
            "training_trace": training_trace,
        }
    except Exception as error:
        if record_db and retrieval_run_id is not None:
            fail_retrieval_rl_run(retrieval_run_id, str(error))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a torch-free PPO-style retrieval policy.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_ppo")
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--rollout-size", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--value-lr", type=float, default=0.02)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["source_hit"])
    parser.add_argument("--topic-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["topic_hit"])
    parser.add_argument("--point-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["point_recall"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_policy(
        data_path=args.data,
        output_dir=args.output,
        updates=args.updates,
        rollout_size=args.rollout_size,
        ppo_epochs=args.ppo_epochs,
        lr=args.lr,
        value_lr=args.value_lr,
        clip_range=args.clip_range,
        seed=args.seed,
        reward_weights={
            "source_hit": args.source_weight,
            "topic_hit": args.topic_weight,
            "point_recall": args.point_weight,
        },
        record_db=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
