from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import deque
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
from scripts.train_retrieval_policy_linucb import build_reward_rows, read_pairs

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as error:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for Dueling Double DQN retrieval policy training: {error}")


DEFAULT_REWARD_WEIGHTS = {
    "source_hit": 0.5,
    "topic_hit": 0.3,
    "point_recall": 0.2,
}


class DuelingQNetwork(nn.Module):
    # 核心5/DDQN：拆分状态价值和动作优势，比普通 Q 网络更稳定地比较检索动作。
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))
        self.advantage = nn.Sequential(nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, action_size))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """前向计算每个检索动作的 Q 值。"""
        hidden = self.trunk(states)
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def parse_args() -> argparse.Namespace:
    """解析 Dueling Double DQN 训练参数，包括 online/offline 模式、epsilon 和 epoch。"""
    parser = argparse.ArgumentParser(description="Train a Dueling Double DQN retrieval policy from offline rewards.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs" / "retrieval_policy_dpo_torch" / "dpo_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_dueling_ddqn")
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--episodes", type=int, default=520)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epsilon-start", type=float, default=0.9)
    parser.add_argument("--epsilon-end", type=float, default=0.06)
    parser.add_argument("--epsilon-decay", type=float, default=0.988)
    parser.add_argument("--margin", type=float, default=0.02)
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


def build_training_items(rows: list[dict[str, Any]]) -> list[tuple[list[float], int, float]]:
    """把 reward rows 展开成 (state, action, reward) 训练样本。"""
    return [
        (row["state"], action_index, float(reward))
        for row in rows
        for action_index, reward in enumerate(row["reward_vector"])
    ]


def train_dueling_ddqn(
    model: DuelingQNetwork,
    target_model: DuelingQNetwork,
    items: list[tuple[list[float], int, float]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> list[dict[str, Any]]:
    # 核心5/DDQN：神经网络策略对照组。当前检索任务是一跳 episode，所以 target 就是即时 reward。
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trace: list[dict[str, Any]] = []
    target_model.load_state_dict(model.state_dict())
    target_update_interval = max(5, epochs // 12)

    for epoch in range(1, epochs + 1):
        random.shuffle(items)
        losses: list[float] = []
        model.train()
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            states = torch.stack([to_tensor(row[0]) for row in batch])
            actions = torch.tensor([row[1] for row in batch], dtype=torch.int64)
            rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)

            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            # Retrieval action selection is a one-step MDP. The Double-DQN
            # target therefore collapses to the observed immediate reward.
            with torch.no_grad():
                targets = rewards
            loss = F.smooth_l1_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

        if epoch % target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 12) == 0:
            trace.append(
                {
                    "epoch": epoch,
                    "loss": round(sum(losses) / max(len(losses), 1), 6),
                    "algorithm": "dueling_double_dqn",
                }
            )
    return trace


def train_dueling_ddqn_online(
    env: RetrievalRLEnv,
    model: DuelingQNetwork,
    target_model: DuelingQNetwork,
    *,
    episodes: int,
    batch_size: int,
    lr: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    seed: int,
) -> list[dict[str, Any]]:
    """在线模式训练：用 epsilon-greedy 与环境交互，边采样边更新 Dueling DDQN。"""
    set_seed(seed)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    replay_buffer: deque[tuple[torch.Tensor, int, float]] = deque(maxlen=1024)
    epsilon = epsilon_start
    trace: list[dict[str, Any]] = []
    target_update_interval = 24

    for episode in range(1, episodes + 1):
        example_index = random.randrange(len(env.examples))
        state = to_tensor(env.reset(example_index))
        if random.random() < epsilon:
            action_index = random.randrange(env.action_size)
        else:
            with torch.no_grad():
                action_index = int(torch.argmax(model(state.unsqueeze(0))[0]).item())
        _, reward, _, info = env.step(action_index)
        replay_buffer.append((state, action_index, float(reward)))

        loss_value = None
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states = torch.stack([item[0] for item in batch])
            actions = torch.tensor([item[1] for item in batch], dtype=torch.int64)
            rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)

            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                targets = rewards
            loss = F.smooth_l1_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.item())

        if episode % target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        trace.append(
            {
                "episode": episode,
                "query": info["query"],
                "action": info["action"],
                "reward": round(float(reward), 4),
                "epsilon": round(epsilon, 4),
                "loss": round(loss_value, 6) if loss_value is not None else None,
                "source_hit": info["source_hit"],
                "topic_hit": info["topic_hit"],
                "point_recall": info["point_recall"],
                "algorithm": "dueling_double_dqn",
            }
        )
    return trace


def evaluate_policy(env: RetrievalRLEnv, model: DuelingQNetwork) -> dict[str, Any]:
    """评测训练后的 Dueling DDQN 策略，并输出每题检索动作和平均指标。"""
    episodes: list[dict[str, Any]] = []
    totals = {"reward": 0.0, "source_hit": 0.0, "topic_hit": 0.0, "point_recall": 0.0}
    model.eval()
    with torch.no_grad():
        for index, example in enumerate(env.examples):
            state = to_tensor(env.reset(index)).unsqueeze(0)
            q_values = model(state)[0]
            action_index = int(torch.argmax(q_values).item())
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
                    "q_values": {ACTIONS[i].name: round(float(q_values[i].item()), 4) for i in range(len(ACTIONS))},
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
    """命令行入口：训练 Dueling DDQN 并保存 pt checkpoint、trace 和 evaluation。"""
    args = parse_args()
    set_seed(args.seed)
    reward_weights = {
        "source_hit": args.source_weight,
        "topic_hit": args.topic_weight,
        "point_recall": args.point_weight,
    }
    examples = build_examples_from_rag_eval(args.data)
    pairs = read_pairs(args.pairs)
    if not examples:
        raise SystemExit(f"No examples found in {args.data}")
    if args.mode == "offline" and not pairs:
        raise SystemExit(f"No preference pairs found in {args.pairs}")

    env = RetrievalRLEnv(examples, reward_weights=reward_weights)
    rows: list[dict[str, Any]] = []
    items: list[tuple[list[float], int, float]] = []
    model = DuelingQNetwork(env.state_size, env.action_size)
    target_model = DuelingQNetwork(env.state_size, env.action_size)

    if args.mode == "online":
        trace = train_dueling_ddqn_online(
            env,
            model,
            target_model,
            episodes=args.episodes,
            batch_size=args.batch_size,
            lr=args.lr,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
        )
    else:
        rows = build_reward_rows(pairs, env.action_size, args.margin)
        items = build_training_items(rows)
        trace = train_dueling_ddqn(
            model,
            target_model,
            items,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    trained_policy = evaluate_policy(env, model)
    baseline_policy = evaluate_fixed_action(env, 0)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "algorithm": "dueling_double_dqn",
        "state_size": env.state_size,
        "action_size": env.action_size,
        "actions": [action.__dict__ for action in ACTIONS],
        "state_dict": model.state_dict(),
        "preference_pairs": len(pairs),
        "reward_rows": len(rows),
        "reward_weights": reward_weights,
        "mode": args.mode,
        "episodes": args.episodes if args.mode == "online" else None,
    }
    torch.save(checkpoint, args.output / "retrieval_policy_dueling_ddqn.pt")
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
        "dueling_ddqn_average_reward": trained_policy["average_reward"],
        "baseline_average_source_hit": baseline_policy["average_source_hit"],
        "dueling_ddqn_average_source_hit": trained_policy["average_source_hit"],
        "baseline_average_topic_hit": baseline_policy["average_topic_hit"],
        "dueling_ddqn_average_topic_hit": trained_policy["average_topic_hit"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
