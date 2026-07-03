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
from scripts.train_retrieval_policy_dpo import build_preference_pairs, load_checkpoint
from scripts.train_retrieval_policy_ppo import ActorCritic, evaluate_fixed_action, evaluate_policy, to_tensor

try:
    import torch
    import torch.nn.functional as F
except Exception as error:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for ORPO retrieval policy training: {error}")


DEFAULT_REWARD_WEIGHTS = {
    "source_hit": 0.5,
    "topic_hit": 0.3,
    "point_recall": 0.2,
}


def parse_args() -> argparse.Namespace:
    """解析 ORPO 训练参数，包括偏好对路径、epoch、batch size 和 odds 系数。"""
    parser = argparse.ArgumentParser(description="ORPO-train a retrieval policy from action preference pairs.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--pairs", type=Path, default=ROOT / "outputs" / "retrieval_policy_dpo_torch" / "dpo_pairs.jsonl")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "outputs" / "retrieval_policy_ppo_torch_60" / "retrieval_policy_ppo.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "retrieval_policy_orpo_torch")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--odds-coef", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["source_hit"])
    parser.add_argument("--topic-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["topic_hit"])
    parser.add_argument("--point-weight", type=float, default=DEFAULT_REWARD_WEIGHTS["point_recall"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _log_odds(log_probs: torch.Tensor, action_indexes: torch.Tensor) -> torch.Tensor:
    """计算指定动作相对其他动作的 log-odds，用于 ORPO 偏好约束。"""
    action_log_probs = log_probs.gather(1, action_indexes.unsqueeze(1)).squeeze(1)
    action_probs = action_log_probs.exp().clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.log(action_probs) - torch.log1p(-action_probs)


def read_preference_pairs(path: Path) -> list[dict[str, Any]]:
    """读取 DPO 生成的 chosen/rejected 检索动作对，作为 ORPO 训练数据。"""
    if not path.exists():
        return []
    pairs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        required = {"state", "chosen_action_index", "rejected_action_index"}
        if required.issubset(row):
            pairs.append(row)
    return pairs


def train_orpo(
    model: ActorCritic,
    pairs: list[dict[str, Any]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    odds_coef: float,
) -> list[dict[str, Any]]:
    # 核心5/ORPO：偏好对照方法，把 chosen action 的 SFT 损失和 rejected action 的 odds-ratio 惩罚合在一起。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trace: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        losses: list[float] = []
        odds_losses: list[float] = []
        sft_losses: list[float] = []
        margins: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            states = torch.stack([to_tensor(row["state"]) for row in batch])
            chosen = torch.tensor([int(row["chosen_action_index"]) for row in batch], dtype=torch.int64)
            rejected = torch.tensor([int(row["rejected_action_index"]) for row in batch], dtype=torch.int64)

            logits, _ = model(states)
            log_probs = F.log_softmax(logits, dim=-1)
            chosen_log_odds = _log_odds(log_probs, chosen)
            rejected_log_odds = _log_odds(log_probs, rejected)
            odds_margin = chosen_log_odds - rejected_log_odds

            sft_loss = F.cross_entropy(logits, chosen)
            odds_loss = -F.logsigmoid(odds_margin).mean()
            loss = sft_loss + odds_coef * odds_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            odds_losses.append(float(odds_loss.item()))
            sft_losses.append(float(sft_loss.item()))
            margins.append(float(odds_margin.mean().item()))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 12) == 0:
            trace.append(
                {
                    "epoch": epoch,
                    "loss": round(sum(losses) / max(len(losses), 1), 6),
                    "orpo_loss": round(sum(odds_losses) / max(len(odds_losses), 1), 6),
                    "sft_loss": round(sum(sft_losses) / max(len(sft_losses), 1), 6),
                    "odds_margin": round(sum(margins) / max(len(margins), 1), 6),
                }
            )
    return trace


def main() -> None:
    """命令行入口：读取偏好对，执行 ORPO 对照训练，并保存评测结果。"""
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
    reference_model = load_checkpoint(args.init_checkpoint, env.state_size, env.action_size)

    pairs = read_preference_pairs(args.pairs)
    oracle_policy: dict[str, Any] = {}
    if pairs:
        dpo_eval_path = args.pairs.parent / "evaluation.json"
        if dpo_eval_path.exists():
            try:
                oracle_policy = json.loads(dpo_eval_path.read_text(encoding="utf-8")).get("oracle_policy", {})
            except json.JSONDecodeError:
                oracle_policy = {}
    else:
        pairs, oracle_policy = build_preference_pairs(env, margin=args.margin)
    if not pairs:
        raise SystemExit("No ORPO preference pairs were built; lower --margin or inspect retrieval rewards.")

    trace = train_orpo(
        model,
        pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        odds_coef=args.odds_coef,
    )
    trained_policy = evaluate_policy(env, model)
    ppo_policy = evaluate_policy(env, reference_model)
    baseline_policy = evaluate_fixed_action(env, 0)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "algorithm": "orpo",
        "state_size": env.state_size,
        "action_size": env.action_size,
        "actions": [action.__dict__ for action in ACTIONS],
        "actor_critic_state_dict": model.state_dict(),
        "init_checkpoint": str(args.init_checkpoint),
        "preference_pairs": len(pairs),
        "reward_weights": reward_weights,
        "odds_coef": args.odds_coef,
    }
    torch.save(checkpoint, args.output / "retrieval_policy_orpo.pt")
    (args.output / "orpo_pairs.jsonl").write_text(
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
        "odds_coef": args.odds_coef,
    }
    (args.output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "ok": True,
        "output": str(args.output),
        "preference_pairs": len(pairs),
        "baseline_average_source_hit": baseline_policy["average_source_hit"],
        "ppo_average_source_hit": ppo_policy["average_source_hit"],
        "orpo_average_source_hit": trained_policy["average_source_hit"],
        "baseline_average_topic_hit": baseline_policy["average_topic_hit"],
        "ppo_average_topic_hit": ppo_policy["average_topic_hit"],
        "orpo_average_topic_hit": trained_policy["average_topic_hit"],
        "baseline_average_reward": baseline_policy["average_reward"],
        "ppo_average_reward": ppo_policy["average_reward"],
        "orpo_average_reward": trained_policy["average_reward"],
        "oracle_average_reward": oracle_policy["average_reward"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
