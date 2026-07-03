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

from backend.retrieval.rl_env import ACTIONS, RetrievalRLEnv, build_examples_from_rag_eval
from scripts.train_retrieval_policy_ppo import ActorCritic, evaluate_fixed_action, evaluate_policy, to_tensor

try:
    import torch
    import torch.nn.functional as F
except Exception as error:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for DPO retrieval policy training: {error}")


DEFAULT_REWARD_WEIGHTS = {
    "source_hit": 0.5,
    "topic_hit": 0.3,
    "point_recall": 0.2,
}


def parse_args() -> argparse.Namespace:
    """解析 DPO 训练参数，包括 PPO 初始 checkpoint、偏好对、epoch 和 beta。"""
    parser = argparse.ArgumentParser(description="DPO-train a retrieval policy from action preference pairs.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "outputs" / "retrieval_policy_ppo_torch_60" / "retrieval_policy_ppo.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_dpo_torch")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--sft-coef", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["source_hit"])
    parser.add_argument("--topic-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["topic_hit"])
    parser.add_argument("--point-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["point_recall"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_checkpoint(path: Path, state_size: int, action_size: int) -> ActorCritic:
    """加载 PPO/DPO 共享的 Actor-Critic checkpoint，作为 DPO 初始策略或参考策略。"""
    model = ActorCritic(state_size, action_size)
    if path.exists():
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("actor_critic_state_dict") or checkpoint.get("state_dict")
        if state_dict:
            model.load_state_dict(state_dict, strict=False)
    return model


def build_preference_pairs(env: RetrievalRLEnv, margin: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 核心5/DPO：把检索 reward 转换成 chosen/rejected 动作对。每题最优动作是 chosen，较弱动作是 rejected。
    pairs: list[dict[str, Any]] = []
    oracle_episodes: list[dict[str, Any]] = []
    for example_index, example in enumerate(env.examples):
        action_rows: list[dict[str, Any]] = []
        for action_index, _action in enumerate(ACTIONS):
            state = env.reset(example_index)
            _, reward, _, info = env.step(action_index)
            action_rows.append(
                {
                    "state": state,
                    "action_index": action_index,
                    "action": info["action"],
                    "reward": float(reward),
                    "source_hit": float(info["source_hit"]),
                    "topic_hit": float(info["topic_hit"]),
                    "point_recall": float(info["point_recall"]),
                    "retrieved_titles": info.get("retrieved_titles", []),
                }
            )
        best = max(action_rows, key=lambda item: (item["reward"], item["source_hit"], item["topic_hit"]))
        oracle_episodes.append(
            {
                "index": example_index,
                "query": example.get("query") or example.get("task"),
                "chosen_action": best["action"],
                "reward": round(best["reward"], 4),
                "source_hit": round(best["source_hit"], 4),
                "topic_hit": round(best["topic_hit"], 4),
                "point_recall": round(best["point_recall"], 4),
                "retrieved_titles": best.get("retrieved_titles", []),
            }
        )
        for row in action_rows:
            if row["action_index"] == best["action_index"]:
                continue
            gap = best["reward"] - row["reward"]
            if gap >= margin:
                pairs.append(
                    {
                        "example_index": example_index,
                        "query": example.get("query") or example.get("task"),
                        "state": best["state"],
                        "chosen_action_index": best["action_index"],
                        "chosen_action": best["action"],
                        "chosen_reward": round(best["reward"], 4),
                        "rejected_action_index": row["action_index"],
                        "rejected_action": row["action"],
                        "rejected_reward": round(row["reward"], 4),
                        "reward_gap": round(gap, 4),
                    }
                )

    totals = {"reward": 0.0, "source_hit": 0.0, "topic_hit": 0.0, "point_recall": 0.0}
    for row in oracle_episodes:
        totals["reward"] += float(row["reward"])
        totals["source_hit"] += float(row["source_hit"])
        totals["topic_hit"] += float(row["topic_hit"])
        totals["point_recall"] += float(row["point_recall"])
    count = max(len(oracle_episodes), 1)
    oracle = {
        "average_reward": round(totals["reward"] / count, 4),
        "average_source_hit": round(totals["source_hit"] / count, 4),
        "average_topic_hit": round(totals["topic_hit"] / count, 4),
        "average_point_recall": round(totals["point_recall"] / count, 4),
        "episodes": oracle_episodes,
    }
    return pairs, oracle


def train_dpo(
    model: ActorCritic,
    ref_model: ActorCritic,
    pairs: list[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    sft_coef: float,
) -> list[dict[str, Any]]:
    # 核心5/DPO：偏好目标让策略相对参考模型更偏向 chosen action，同时保留一定 SFT 稳定项。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ref_model.eval()
    trace: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        losses: list[float] = []
        dpo_losses: list[float] = []
        sft_losses: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            states = torch.stack([to_tensor(row["state"]) for row in batch])
            chosen = torch.tensor([int(row["chosen_action_index"]) for row in batch], dtype=torch.int64)
            rejected = torch.tensor([int(row["rejected_action_index"]) for row in batch], dtype=torch.int64)

            logits, _ = model(states)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_logp = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
            rejected_logp = log_probs.gather(1, rejected.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                ref_logits, _ = ref_model(states)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_chosen = ref_log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
                ref_rejected = ref_log_probs.gather(1, rejected.unsqueeze(1)).squeeze(1)

            preference_logit = (chosen_logp - rejected_logp) - (ref_chosen - ref_rejected)
            dpo_loss = -F.logsigmoid(beta * preference_logit).mean()
            sft_loss = F.cross_entropy(logits, chosen)
            loss = dpo_loss + sft_coef * sft_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            dpo_losses.append(float(dpo_loss.item()))
            sft_losses.append(float(sft_loss.item()))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 12) == 0:
            trace.append(
                {
                    "epoch": epoch,
                    "loss": round(sum(losses) / max(len(losses), 1), 6),
                    "dpo_loss": round(sum(dpo_losses) / max(len(dpo_losses), 1), 6),
                    "sft_loss": round(sum(sft_losses) / max(len(sft_losses), 1), 6),
                }
            )
    return trace


def main() -> None:
    """命令行入口：构造偏好对、执行 DPO 训练，并保存 checkpoint 与 evaluation。"""
    args = parse_args()
    set_seed(args.seed)
    reward_weights = {
        "source_hit": args.source_weight,
        "topic_hit": args.topic_weight,
        "point_recall": args.point_weight,
    }
    examples = build_examples_from_rag_eval(args.data)
    if not examples:
        raise SystemExit(f"No examples found in {args.data}")
    env = RetrievalRLEnv(examples, reward_weights=reward_weights)
    model = load_checkpoint(args.init_checkpoint, env.state_size, env.action_size)
    ref_model = load_checkpoint(args.init_checkpoint, env.state_size, env.action_size)

    pairs, oracle_policy = build_preference_pairs(env, margin=args.margin)
    if not pairs:
        raise SystemExit("No DPO preference pairs were built; lower --margin or inspect retrieval rewards.")

    trace = train_dpo(
        model,
        ref_model,
        pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        sft_coef=args.sft_coef,
    )
    trained_policy = evaluate_policy(env, model)
    ppo_policy = evaluate_policy(env, ref_model)
    baseline_policy = evaluate_fixed_action(env, 0)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "algorithm": "dpo",
        "state_size": env.state_size,
        "action_size": env.action_size,
        "actions": [action.__dict__ for action in ACTIONS],
        "actor_critic_state_dict": model.state_dict(),
        "init_checkpoint": str(args.init_checkpoint),
        "preference_pairs": len(pairs),
        "reward_weights": reward_weights,
    }
    torch.save(checkpoint, args.output / "retrieval_policy_dpo.pt")
    (args.output / "dpo_pairs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pairs) + "\n",
        encoding="utf-8",
    )
    (args.output / "training_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation = {
        "trained_policy": trained_policy,
        "ppo_reference_policy": ppo_policy,
        "baseline_policy": baseline_policy,
        "oracle_policy": oracle_policy,
        "preference_pair_count": len(pairs),
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "ok": True,
        "output": str(args.output),
        "preference_pairs": len(pairs),
        "baseline_average_source_hit": baseline_policy["average_source_hit"],
        "ppo_average_source_hit": ppo_policy["average_source_hit"],
        "dpo_average_source_hit": trained_policy["average_source_hit"],
        "baseline_average_topic_hit": baseline_policy["average_topic_hit"],
        "ppo_average_topic_hit": ppo_policy["average_topic_hit"],
        "dpo_average_topic_hit": trained_policy["average_topic_hit"],
        "baseline_average_reward": baseline_policy["average_reward"],
        "ppo_average_reward": ppo_policy["average_reward"],
        "dpo_average_reward": trained_policy["average_reward"],
        "oracle_average_reward": oracle_policy["average_reward"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
