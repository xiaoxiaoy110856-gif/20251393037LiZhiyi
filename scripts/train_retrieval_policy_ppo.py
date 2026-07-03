from __future__ import annotations

import argparse
import json
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

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except Exception as error:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for PPO retrieval policy training: {error}")


DEFAULT_REWARD_WEIGHTS = {
    "source_hit": 0.5,
    "topic_hit": 0.3,
    "point_recall": 0.2,
}


class ActorCritic(nn.Module):
    """PPO 使用的 Actor-Critic 网络：actor 选择检索动作，critic 估计当前问题状态价值。"""
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向计算动作 logits 和状态价值。"""
        hidden = self.trunk(states)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def act(self, state: torch.Tensor) -> tuple[int, float, float]:
        """按当前策略采样一个检索动作，并返回动作、log_prob 和 value。"""
        logits, value = self.forward(state.unsqueeze(0))
        dist = Categorical(logits=logits[0])
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value[0].item())


def parse_args() -> argparse.Namespace:
    """解析 PPO 训练参数，包括样本路径、输出目录、更新轮数和 reward 权重。"""
    parser = argparse.ArgumentParser(description="Train a PPO retrieval policy for trajectory/RL RAG.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_ppo")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=0, help="Alias for --updates, for experiment logs that use epoch wording.")
    parser.add_argument("--rollout-size", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["source_hit"])
    parser.add_argument("--topic-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["topic_hit"])
    parser.add_argument("--point-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["point_recall"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def evaluate_policy(env: RetrievalRLEnv, model: ActorCritic) -> dict[str, Any]:
    """用训练后的策略逐题评测，输出平均 reward、Source Hit、Topic Hit 和每题明细。"""
    per_example: list[dict[str, Any]] = []
    totals = {"reward": 0.0, "source_hit": 0.0, "topic_hit": 0.0, "point_recall": 0.0}

    model.eval()
    with torch.no_grad():
        for index, example in enumerate(env.examples):
            state = to_tensor(env.reset(index)).unsqueeze(0)
            logits, _ = model(state)
            action_index = int(torch.argmax(logits[0]).item())
            _, reward, _, info = env.step(action_index)
            totals["reward"] += reward
            totals["source_hit"] += info["source_hit"]
            totals["topic_hit"] += info["topic_hit"]
            totals["point_recall"] += info["point_recall"]
            per_example.append(
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
        "episodes": per_example,
    }


def evaluate_fixed_action(env: RetrievalRLEnv, action_index: int) -> dict[str, Any]:
    """固定选择某个动作进行评测，action_index=0 时就是 baseline。"""
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


def _collect_rollout(env: RetrievalRLEnv, model: ActorCritic, rollout_size: int) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    # 核心5/PPO：收集 on-policy 检索 episode。每个采样问题都会产生动作、奖励、log_prob 和 value。
    states: list[torch.Tensor] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    trace: list[dict[str, Any]] = []

    model.eval()
    for _ in range(rollout_size):
        index = random.randrange(len(env.examples))
        state = to_tensor(env.reset(index))
        action_index, log_prob, value = model.act(state)
        _, reward, _, info = env.step(action_index)
        states.append(state)
        actions.append(action_index)
        old_log_probs.append(log_prob)
        values.append(value)
        rewards.append(float(reward))
        trace.append(
            {
                "query": info["query"],
                "action": info["action"],
                "reward": round(float(reward), 4),
                "source_hit": info["source_hit"],
                "topic_hit": info["topic_hit"],
                "point_recall": info["point_recall"],
            }
        )

    batch = {
        "states": torch.stack(states),
        "actions": torch.tensor(actions, dtype=torch.int64),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "returns": torch.tensor(rewards, dtype=torch.float32),
        "values": torch.tensor(values, dtype=torch.float32),
    }
    advantages = batch["returns"] - batch["values"]
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch["advantages"] = advantages
    return batch, trace


def train_policy(
    *,
    data_path: Path,
    output_dir: Path,
    examples: list[dict[str, Any]] | None = None,
    updates: int = 80,
    rollout_size: int = 32,
    ppo_epochs: int = 4,
    minibatch_size: int = 16,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.02,
    lr: float = 3e-4,
    seed: int = 42,
    reward_weights: dict[str, float] | None = None,
    run_name: str | None = None,
    record_db: bool = True,
) -> dict[str, Any]:
    # 核心5/PPO：主训练入口。根据 Source Hit、Topic Hit、Point Recall 奖励优化检索动作 actor。
    set_seed(seed)
    loaded_examples = examples if examples is not None else build_examples_from_rag_eval(data_path)
    if not loaded_examples:
        raise RuntimeError(f"No retrieval RL samples found in {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    reward_weights = dict(DEFAULT_REWARD_WEIGHTS | (reward_weights or {}))
    run_name = run_name or output_dir.name
    retrieval_run_id = None
    if record_db:
        retrieval_run_id = create_retrieval_rl_run(run_name=run_name, data_path=str(data_path), output_path=str(output_dir), status="started")
        save_training_run(
            run_type="retrieval_policy_ppo",
            model_path=str(output_dir / "retrieval_policy_ppo.pt"),
            data_path=str(data_path),
            output_path=str(output_dir),
            status="started",
            notes=f"PPO retrieval policy training started. reward_weights={reward_weights}",
            metrics={"algorithm": "ppo", "reward_weights": reward_weights},
        )

    try:
        env = RetrievalRLEnv(loaded_examples, reward_weights=reward_weights)
        model = ActorCritic(env.state_size, env.action_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        training_trace: list[dict[str, Any]] = []

        for update_index in range(1, updates + 1):
            batch, trace = _collect_rollout(env, model, rollout_size)
            model.train()
            indices = list(range(rollout_size))
            last_loss = 0.0
            for _ in range(ppo_epochs):
                random.shuffle(indices)
                for start in range(0, rollout_size, minibatch_size):
                    selected = indices[start : start + minibatch_size]
                    states = batch["states"][selected]
                    actions = batch["actions"][selected]
                    old_log_probs = batch["old_log_probs"][selected]
                    returns = batch["returns"][selected]
                    advantages = batch["advantages"][selected]

                    logits, values = model(states)
                    dist = Categorical(logits=logits)
                    log_probs = dist.log_prob(actions)
                    ratio = torch.exp(log_probs - old_log_probs)
                    unclipped = ratio * advantages
                    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = F.mse_loss(values, returns)
                    entropy = dist.entropy().mean()
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    last_loss = float(loss.item())

            avg_reward = sum(item["reward"] for item in trace) / max(len(trace), 1)
            best_step = max(trace, key=lambda item: float(item.get("reward", 0.0)))
            training_trace.append(
                {
                    "episode": update_index,
                    "query": best_step["query"],
                    "action": best_step["action"],
                    "reward": round(avg_reward, 4),
                    "loss": round(last_loss, 6),
                    "source_hit": best_step["source_hit"],
                    "topic_hit": best_step["topic_hit"],
                    "point_recall": best_step["point_recall"],
                    "reward_weights": reward_weights,
                    "algorithm": "ppo",
                }
            )

        # 核心5/结果：evaluation.json 保存 trained-vs-baseline 指标，前端 Source Hit/Topic Hit 折线图会读取它。
        evaluation = evaluate_policy(env, model)
        baseline = evaluate_fixed_action(env, action_index=0)
        metrics_payload = {
            "algorithm": "ppo",
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

        torch.save(
            {
                "algorithm": "ppo",
                "actor_critic_state_dict": model.state_dict(),
                "actions": [action.__dict__ for action in ACTIONS],
                "state_size": env.state_size,
                "action_size": env.action_size,
                "data_path": str(data_path),
                "reward_weights": reward_weights,
            },
            output_dir / "retrieval_policy_ppo.pt",
        )
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
                metrics=metrics_payload,
                evaluation={"trained_policy": evaluation, "baseline_policy": baseline},
                trace=training_trace,
            )
            save_training_run(
                run_type="retrieval_policy_ppo",
                model_path=str(output_dir / "retrieval_policy_ppo.pt"),
                data_path=str(data_path),
                output_path=str(output_dir),
                status="completed",
                notes=f"PPO retrieval policy training completed. reward_weights={reward_weights}",
                metrics=metrics_payload,
            )

        return {
            "ok": True,
            "metrics": metrics_payload,
            "output": str(output_dir),
            "evaluation": {"trained_policy": evaluation, "baseline_policy": baseline},
            "training_trace": training_trace,
            "reward_weights": reward_weights,
        }
    except Exception as error:
        if record_db and retrieval_run_id is not None:
            fail_retrieval_rl_run(retrieval_run_id, str(error))
            save_training_run(
                run_type="retrieval_policy_ppo",
                model_path=str(output_dir / "retrieval_policy_ppo.pt"),
                data_path=str(data_path),
                output_path=str(output_dir),
                status="failed",
                notes=str(error),
                metrics={"algorithm": "ppo", "reward_weights": reward_weights},
            )
        raise


def main() -> None:
    """命令行入口：解析参数并启动 PPO 检索策略训练。"""
    args = parse_args()
    updates = args.epochs if args.epochs > 0 else args.updates
    result = train_policy(
        data_path=args.data,
        output_dir=args.output,
        updates=updates,
        rollout_size=args.rollout_size,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        seed=args.seed,
        reward_weights={
            "source_hit": args.source_weight,
            "topic_hit": args.topic_weight,
            "point_recall": args.point_weight,
        },
        run_name=args.output.name,
        record_db=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
