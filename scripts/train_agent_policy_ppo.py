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

from backend.agent.rl_training import (
    AGENT_ACTION_NAMES,
    AgentWorkflowEnv,
    average_metric,
    discounted_returns,
    finite,
    read_jsonl,
    run_baseline_plan,
    safe_mean,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except Exception as error:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for multi-step Agent PPO training: {error}")


class AgentActorCritic(nn.Module):
    """Actor-Critic policy for multi-step Agent workflow decisions."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_size, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
        )
        self.actor = nn.Linear(96, action_size)
        self.critic = nn.Linear(96, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(states)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def act(self, state: torch.Tensor, valid_indices: list[int] | None = None) -> tuple[int, float, float, list[float]]:
        logits, value = self.forward(state.unsqueeze(0))
        masked_logits = logits[0].clone()
        if valid_indices:
            mask = torch.full_like(masked_logits, -1.0e9)
            mask[valid_indices] = 0.0
            masked_logits = masked_logits + mask
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        probs = torch.softmax(masked_logits, dim=-1)
        return int(action.item()), float(dist.log_prob(action).item()), float(value[0].item()), probs.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for a multi-step Agent workflow policy.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "agent_eval.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "agent_policy_ppo_multi_step")
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--rollout-episodes", type=int, default=12)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.015)
    parser.add_argument("--imitation-coef", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def collect_rollout(
    env: AgentWorkflowEnv,
    model: AgentActorCritic,
    *,
    rollout_episodes: int,
    gamma: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    states: list[torch.Tensor] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    values: list[float] = []
    oracle_actions: list[int] = []
    valid_masks: list[torch.Tensor] = []
    all_returns: list[float] = []
    all_rewards: list[float] = []
    all_dones: list[bool] = []
    trace: list[dict[str, Any]] = []

    model.eval()
    for _ in range(rollout_episodes):
        index = random.randrange(len(env.examples))
        state_values = env.reset(index)
        episode_rewards: list[float] = []
        episode_dones: list[bool] = []
        episode_transition_count = 0
        for _step in range(env.max_steps):
            state_tensor = to_tensor(state_values)
            oracle_action = env.oracle_action_index()
            valid_indices = env.valid_action_indices()
            action_index, log_prob, value, probs = model.act(state_tensor, valid_indices)
            valid_mask = torch.full((len(AGENT_ACTION_NAMES),), -1.0e9, dtype=torch.float32)
            valid_mask[valid_indices] = 0.0
            next_state, reward, done, info = env.step(action_index)

            states.append(state_tensor)
            actions.append(action_index)
            old_log_probs.append(log_prob)
            values.append(value)
            oracle_actions.append(oracle_action)
            valid_masks.append(valid_mask)
            episode_rewards.append(float(reward))
            episode_dones.append(bool(done))
            all_rewards.append(float(reward))
            all_dones.append(bool(done))
            episode_transition_count += 1
            trace.append(
                {
                    "query": info["query"],
                    "step": info["steps"],
                    "action": info["action"],
                    "oracle_action": AGENT_ACTION_NAMES[oracle_action],
                    "reward": round(float(reward), 4),
                    "answer_point_recall": info["answer_point_recall"],
                    "expected_tool_hit": info["expected_tool_hit"],
                    "probs": {AGENT_ACTION_NAMES[i]: round(float(probs[i]), 4) for i in range(len(AGENT_ACTION_NAMES))},
                }
            )
            state_values = next_state
            if done:
                break
        returns = discounted_returns(episode_rewards, episode_dones, gamma)
        all_returns.extend(returns[:episode_transition_count])

    batch = {
        "states": torch.stack(states),
        "actions": torch.tensor(actions, dtype=torch.int64),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "returns": torch.tensor(all_returns, dtype=torch.float32),
        "values": torch.tensor(values, dtype=torch.float32),
        "oracle_actions": torch.tensor(oracle_actions, dtype=torch.int64),
        "valid_masks": torch.stack(valid_masks),
        "rewards": torch.tensor(all_rewards, dtype=torch.float32),
        "dones": torch.tensor(all_dones, dtype=torch.bool),
    }
    advantages = batch["returns"] - batch["values"]
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch["advantages"] = advantages
    return batch, trace


def evaluate_trained_policy(env: AgentWorkflowEnv, model: AgentActorCritic) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for index, example in enumerate(env.examples):
            state = env.reset(index)
            total_reward = 0.0
            for _ in range(env.max_steps):
                logits, _value = model(to_tensor(state).unsqueeze(0))
                masked_logits = logits[0].clone()
                valid = env.valid_action_indices()
                mask = torch.full_like(masked_logits, -1.0e9)
                mask[valid] = 0.0
                action_index = int(torch.argmax(masked_logits + mask).item())
                state, reward, done, info = env.step(action_index)
                total_reward += float(reward)
                if done:
                    break
            row = env.metrics()
            row["total_reward"] = round(total_reward, 4)
            row["policy"] = "ppo_multi_step"
            row["example_index"] = index
            row["task"] = example.get("task") or example.get("query") or example.get("prompt")
            episodes.append(row)
    return summarize_episodes(episodes)


def evaluate_baseline(examples: list[dict[str, Any]], *, max_steps: int) -> dict[str, Any]:
    episodes = [run_baseline_plan(example, max_steps=max_steps) for example in examples]
    return summarize_episodes(episodes)


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "average_reward": average_metric(episodes, "total_reward"),
        "average_answer_point_recall": average_metric(episodes, "answer_point_recall"),
        "average_expected_tool_hit": average_metric(episodes, "expected_tool_hit"),
        "average_evidence_count": average_metric(episodes, "evidence_count"),
        "episodes": episodes,
    }


def train_policy(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    examples = read_jsonl(args.data)
    if not examples:
        raise RuntimeError(f"No Agent PPO examples found in {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)

    env = AgentWorkflowEnv(examples, max_steps=args.max_steps)
    env.reset(0)
    model = AgentActorCritic(env.state_size, env.action_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    training_trace: list[dict[str, Any]] = []

    for update_index in range(1, args.updates + 1):
        batch, trace = collect_rollout(env, model, rollout_episodes=args.rollout_episodes, gamma=args.gamma)
        model.train()
        count = int(batch["actions"].numel())
        indices = list(range(count))
        last_loss = 0.0
        last_policy = 0.0
        last_value = 0.0
        last_imitation = 0.0
        for _ in range(args.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, count, args.minibatch_size):
                selected = indices[start : start + args.minibatch_size]
                states = batch["states"][selected]
                actions = batch["actions"][selected]
                old_log_probs = batch["old_log_probs"][selected]
                returns = batch["returns"][selected]
                advantages = batch["advantages"][selected]
                oracle_actions = batch["oracle_actions"][selected]
                valid_masks = batch["valid_masks"][selected]

                logits, values = model(states)
                masked_logits = logits + valid_masks
                dist = Categorical(logits=masked_logits)
                log_probs = dist.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns)
                entropy = dist.entropy().mean()
                imitation_loss = F.cross_entropy(masked_logits, oracle_actions)
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy
                    + args.imitation_coef * imitation_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                last_loss = float(loss.item())
                last_policy = float(policy_loss.item())
                last_value = float(value_loss.item())
                last_imitation = float(imitation_loss.item())

        if update_index == 1 or update_index == args.updates or update_index % max(1, args.updates // 12) == 0:
            training_trace.append(
                {
                    "episode": update_index,
                    "algorithm": "ppo_multi_step_agent",
                    "avg_step_reward": round(finite(float(batch["rewards"].mean().item())), 4),
                    "loss": round(last_loss, 6),
                    "policy_loss": round(last_policy, 6),
                    "value_loss": round(last_value, 6),
                    "imitation_loss": round(last_imitation, 6),
                    "sample_actions": [row["action"] for row in trace[-min(len(trace), 8) :]],
                }
            )

    trained = evaluate_trained_policy(env, model)
    baseline = evaluate_baseline(examples, max_steps=args.max_steps)
    metrics = {
        "algorithm": "ppo_multi_step_agent",
        "updates": args.updates,
        "rollout_episodes": args.rollout_episodes,
        "max_steps": args.max_steps,
        "trained_average_reward": trained["average_reward"],
        "baseline_average_reward": baseline["average_reward"],
        "reward_gain_vs_baseline": round(trained["average_reward"] - baseline["average_reward"], 4),
        "trained_average_answer_point_recall": trained["average_answer_point_recall"],
        "baseline_average_answer_point_recall": baseline["average_answer_point_recall"],
        "answer_point_recall_gain_vs_baseline": round(
            trained["average_answer_point_recall"] - baseline["average_answer_point_recall"], 4
        ),
        "trained_average_expected_tool_hit": trained["average_expected_tool_hit"],
        "baseline_average_expected_tool_hit": baseline["average_expected_tool_hit"],
        "expected_tool_hit_gain_vs_baseline": round(
            trained["average_expected_tool_hit"] - baseline["average_expected_tool_hit"], 4
        ),
        "actions": list(AGENT_ACTION_NAMES),
    }

    torch.save(
        {
            "algorithm": "ppo_multi_step_agent",
            "state_size": env.state_size,
            "action_size": env.action_size,
            "actions": list(AGENT_ACTION_NAMES),
            "state_dict": model.state_dict(),
            "data_path": str(args.data),
            "max_steps": args.max_steps,
        },
        args.output / "agent_policy_ppo.pt",
    )
    (args.output / "training_trace.json").write_text(json.dumps(training_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "evaluation.json").write_text(
        json.dumps({"trained_policy": trained, "baseline_policy": baseline}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "output": str(args.output),
        "metrics": metrics,
        "training_trace": training_trace,
    }


def main() -> None:
    args = parse_args()
    result = train_policy(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
