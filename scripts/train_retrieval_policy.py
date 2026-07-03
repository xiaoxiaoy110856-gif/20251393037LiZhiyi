from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
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
except Exception as error:  # pragma: no cover - import guard
    raise SystemExit(f"PyTorch is required for retrieval policy training: {error}")


DEFAULT_REWARD_WEIGHTS = {
    "source_hit": 0.5,
    "topic_hit": 0.3,
    "point_recall": 0.2,
}


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN retrieval policy for trajectory/RL RAG.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy")
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.985)
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


def evaluate_policy(env: RetrievalRLEnv, model: QNetwork) -> dict[str, Any]:
    per_example: list[dict[str, Any]] = []
    total_reward = 0.0
    total_source = 0.0
    total_topic = 0.0
    total_points = 0.0

    model.eval()
    with torch.no_grad():
        for index, example in enumerate(env.examples):
            state = to_tensor(env.reset(index)).unsqueeze(0)
            q_values = model(state)[0]
            action_index = int(torch.argmax(q_values).item())
            _, reward, _, info = env.step(action_index)
            total_reward += reward
            total_source += info["source_hit"]
            total_topic += info["topic_hit"]
            total_points += info["point_recall"]
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
        "average_reward": round(total_reward / count, 4),
        "average_source_hit": round(total_source / count, 4),
        "average_topic_hit": round(total_topic / count, 4),
        "average_point_recall": round(total_points / count, 4),
        "episodes": per_example,
    }


def evaluate_fixed_action(env: RetrievalRLEnv, action_index: int) -> dict[str, Any]:
    total_reward = 0.0
    total_source = 0.0
    total_topic = 0.0
    total_points = 0.0
    episodes: list[dict[str, Any]] = []

    for index, example in enumerate(env.examples):
        env.reset(index)
        _, reward, _, info = env.step(action_index)
        total_reward += reward
        total_source += info["source_hit"]
        total_topic += info["topic_hit"]
        total_points += info["point_recall"]
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
        "average_reward": round(total_reward / count, 4),
        "average_source_hit": round(total_source / count, 4),
        "average_topic_hit": round(total_topic / count, 4),
        "average_point_recall": round(total_points / count, 4),
        "episodes": episodes,
    }


def train_policy(
    *,
    data_path: Path,
    output_dir: Path,
    examples: list[dict[str, Any]] | None = None,
    episodes: int = 240,
    batch_size: int = 32,
    gamma: float = 0.95,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.985,
    seed: int = 42,
    reward_weights: dict[str, float] | None = None,
    run_name: str | None = None,
    record_db: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    loaded_examples = examples if examples is not None else build_examples_from_rag_eval(data_path)
    if not loaded_examples:
        raise RuntimeError(f"No retrieval RL samples found in {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    reward_weights = dict(DEFAULT_REWARD_WEIGHTS | (reward_weights or {}))
    run_name = run_name or output_dir.name

    retrieval_run_id = None
    if record_db:
        retrieval_run_id = create_retrieval_rl_run(
            run_name=run_name,
            data_path=str(data_path),
            output_path=str(output_dir),
            status="started",
        )
        save_training_run(
            run_type="retrieval_policy_dqn",
            model_path=str(output_dir / "retrieval_policy_dqn.pt"),
            data_path=str(data_path),
            output_path=str(output_dir),
            status="started",
            notes=f"Retrieval strategy learning started. reward_weights={reward_weights}",
            metrics={"reward_weights": reward_weights},
        )

    try:
        env = RetrievalRLEnv(loaded_examples, reward_weights=reward_weights)
        q_net = QNetwork(env.state_size, env.action_size)
        target_net = QNetwork(env.state_size, env.action_size)
        target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
        replay_buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=512)

        epsilon = epsilon_start
        training_trace: list[dict[str, Any]] = []
        target_update_interval = 20

        for episode in range(1, episodes + 1):
            index = random.randrange(len(loaded_examples))
            state = to_tensor(env.reset(index))

            if random.random() < epsilon:
                action_index = random.randrange(env.action_size)
            else:
                with torch.no_grad():
                    action_index = int(torch.argmax(q_net(state.unsqueeze(0))[0]).item())

            next_state_values, reward, done, info = env.step(action_index)
            next_state = to_tensor(next_state_values)
            replay_buffer.append((state, action_index, reward, next_state, done))

            loss_value = None
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states = torch.stack([item[0] for item in batch])
                actions = torch.tensor([item[1] for item in batch], dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
                next_states = torch.stack([item[3] for item in batch])
                dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)

                q_values = q_net(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(dim=1).values
                    targets = rewards + gamma * next_q_values * (1.0 - dones)

                loss = F.mse_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = float(loss.item())

            if episode % target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            training_trace.append(
                {
                    "episode": episode,
                    "query": info["query"],
                    "action": info["action"],
                    "reward": round(reward, 4),
                    "epsilon": round(epsilon, 4),
                    "loss": round(loss_value, 6) if loss_value is not None else None,
                    "source_hit": info["source_hit"],
                    "topic_hit": info["topic_hit"],
                    "point_recall": info["point_recall"],
                    "reward_weights": reward_weights,
                }
            )

        evaluation = evaluate_policy(env, q_net)
        baseline = evaluate_fixed_action(env, action_index=0)

        torch.save(
            {
                "state_dict": q_net.state_dict(),
                "actions": [action.__dict__ for action in ACTIONS],
                "state_size": env.state_size,
                "action_size": env.action_size,
                "data_path": str(data_path),
                "reward_weights": reward_weights,
            },
            output_dir / "retrieval_policy_dqn.pt",
        )
        metrics_payload = {
            "algorithm": "dqn",
            "episodes": episodes,
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
                run_type="retrieval_policy_dqn",
                model_path=str(output_dir / "retrieval_policy_dqn.pt"),
                data_path=str(data_path),
                output_path=str(output_dir),
                status="completed",
                notes=f"Retrieval strategy learning completed. reward_weights={reward_weights}",
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
                run_type="retrieval_policy_dqn",
                model_path=str(output_dir / "retrieval_policy_dqn.pt"),
                data_path=str(data_path),
                output_path=str(output_dir),
                status="failed",
                notes=str(error),
                metrics={"reward_weights": reward_weights},
            )
        raise


def main() -> None:
    args = parse_args()
    result = train_policy(
        data_path=args.data,
        output_dir=args.output,
        episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
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
