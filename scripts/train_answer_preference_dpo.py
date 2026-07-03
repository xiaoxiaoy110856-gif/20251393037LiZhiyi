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

from backend.agent.answer_preference import ANSWER_FEATURE_NAMES, answer_features, format_preference_row, score_answer
from backend.agent.rl_training import read_jsonl

try:
    import torch
    import torch.nn.functional as F
except Exception as error:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for answer-preference DPO training: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DPO-style answer preference model.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "assistant_dpo.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "answer_preference_dpo")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_pairs(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for row in read_jsonl(path):
        parsed = format_preference_row(row)
        if parsed["prompt"] and parsed["chosen"] and parsed["rejected"]:
            pairs.append(parsed)
    return pairs


def pair_tensors(pairs: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = [answer_features(row["prompt"], row["chosen"]) for row in pairs]
    rejected = [answer_features(row["prompt"], row["rejected"]) for row in pairs]
    return torch.tensor(chosen, dtype=torch.float32), torch.tensor(rejected, dtype=torch.float32)


def evaluate_pairs(pairs: list[dict[str, str]], weights: torch.Tensor) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    wins = 0.0
    margins: list[float] = []
    weight_list = [float(item) for item in weights.detach().cpu().tolist()]
    for row in pairs:
        chosen_features = answer_features(row["prompt"], row["chosen"])
        rejected_features = answer_features(row["prompt"], row["rejected"])
        chosen_score = score_answer(chosen_features, weight_list)
        rejected_score = score_answer(rejected_features, weight_list)
        margin = chosen_score - rejected_score
        wins += 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        margins.append(margin)
        rows.append(
            {
                "prompt": row["prompt"],
                "chosen_score": round(chosen_score, 4),
                "rejected_score": round(rejected_score, 4),
                "margin": round(margin, 4),
                "preferred_correct": margin > 0,
                "chosen_features": {ANSWER_FEATURE_NAMES[i]: round(chosen_features[i], 4) for i in range(len(ANSWER_FEATURE_NAMES))},
                "rejected_features": {
                    ANSWER_FEATURE_NAMES[i]: round(rejected_features[i], 4) for i in range(len(ANSWER_FEATURE_NAMES))
                },
            }
        )
    return {
        "pairwise_accuracy": round(wins / max(len(pairs), 1), 4),
        "average_margin": round(sum(margins) / max(len(margins), 1), 4),
        "pairs": rows,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    pairs = load_pairs(args.data)
    if not pairs:
        raise RuntimeError(f"No answer DPO pairs found in {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)

    chosen_features, rejected_features = pair_tensors(pairs)
    weights = torch.zeros(chosen_features.shape[1], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([weights], lr=args.lr, weight_decay=args.l2)
    trace: list[dict[str, Any]] = []
    indices = list(range(len(pairs)))

    initial_eval = evaluate_pairs(pairs, weights.detach())
    for epoch in range(1, args.epochs + 1):
        random.shuffle(indices)
        losses: list[float] = []
        margins: list[float] = []
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start : start + args.batch_size]
            chosen_batch = chosen_features[selected]
            rejected_batch = rejected_features[selected]
            chosen_scores = chosen_batch @ weights
            rejected_scores = rejected_batch @ weights
            margin = chosen_scores - rejected_scores
            dpo_loss = -F.logsigmoid(args.beta * margin).mean()
            loss = dpo_loss + args.l2 * torch.sum(weights * weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            margins.extend(float(item) for item in margin.detach().cpu().tolist())
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 12) == 0:
            trace.append(
                {
                    "epoch": epoch,
                    "loss": round(sum(losses) / max(len(losses), 1), 6),
                    "average_margin": round(sum(margins) / max(len(margins), 1), 4),
                    "algorithm": "answer_preference_dpo",
                }
            )

    final_eval = evaluate_pairs(pairs, weights.detach())
    weight_list = [float(item) for item in weights.detach().cpu().tolist()]
    metrics = {
        "algorithm": "answer_preference_dpo",
        "train_pairs": len(pairs),
        "epochs": args.epochs,
        "beta": args.beta,
        "initial_pairwise_accuracy": initial_eval["pairwise_accuracy"],
        "trained_pairwise_accuracy": final_eval["pairwise_accuracy"],
        "pairwise_accuracy_gain": round(final_eval["pairwise_accuracy"] - initial_eval["pairwise_accuracy"], 4),
        "initial_average_margin": initial_eval["average_margin"],
        "trained_average_margin": final_eval["average_margin"],
        "average_margin_gain": round(final_eval["average_margin"] - initial_eval["average_margin"], 4),
    }
    model_payload = {
        "algorithm": "answer_preference_dpo",
        "feature_names": list(ANSWER_FEATURE_NAMES),
        "weights": weight_list,
        "data_path": str(args.data),
        "beta": args.beta,
    }
    (args.output / "answer_preference_model.json").write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "training_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "evaluation.json").write_text(
        json.dumps({"initial_policy": initial_eval, "trained_policy": final_eval}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "output": str(args.output),
        "metrics": metrics,
        "weights": {ANSWER_FEATURE_NAMES[i]: round(weight_list[i], 4) for i in range(len(ANSWER_FEATURE_NAMES))},
        "training_trace": trace,
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
