from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.storage.db import save_training_run
from backend.retrieval.rl_env import build_examples_from_rag_eval
from train_retrieval_policy import train_policy


REWARD_CANDIDATES: list[dict[str, Any]] = [
    {"name": "balanced", "weights": {"source_hit": 0.50, "topic_hit": 0.30, "point_recall": 0.20}},
    {"name": "source_priority", "weights": {"source_hit": 0.60, "topic_hit": 0.25, "point_recall": 0.15}},
    {"name": "topic_priority", "weights": {"source_hit": 0.40, "topic_hit": 0.40, "point_recall": 0.20}},
    {"name": "coverage_priority", "weights": {"source_hit": 0.35, "topic_hit": 0.25, "point_recall": 0.40}},
    {"name": "strict_grounding", "weights": {"source_hit": 0.55, "topic_hit": 0.35, "point_recall": 0.10}},
    {"name": "answer_support", "weights": {"source_hit": 0.45, "topic_hit": 0.15, "point_recall": 0.40}},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously test retrieval rewards and store all results.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval_extended.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "retrieval_reward_sweep")
    parser.add_argument("--episodes-per-trial", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--time-budget-minutes", type=float, default=20.0)
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def score_trial(metrics: dict[str, Any]) -> float:
    reward_gain = float(metrics.get("reward_gain_vs_baseline", 0.0))
    source_gain = float(metrics.get("trained_average_source_hit", 0.0)) - float(metrics.get("baseline_average_source_hit", 0.0))
    topic_gain = float(metrics.get("trained_average_topic_hit", 0.0)) - float(metrics.get("baseline_average_topic_hit", 0.0))
    point_gain = float(metrics.get("trained_average_point_recall", 0.0)) - float(metrics.get("baseline_average_point_recall", 0.0))
    return round(reward_gain + 0.4 * source_gain + 0.2 * topic_gain + 0.2 * point_gain, 4)


def main() -> None:
    args = parse_args()
    examples = build_examples_from_rag_eval(args.data)
    if not examples:
        raise SystemExit(f"No retrieval RL samples found in {args.data}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    deadline = started_at + max(args.time_budget_minutes, 0.1) * 60.0

    summaries: list[dict[str, Any]] = []
    for trial_index, candidate in enumerate(REWARD_CANDIDATES[: args.max_trials], start=1):
        if time.time() >= deadline:
            break
        trial_dir = args.output_root / f"{trial_index:02d}_{candidate['name']}"
        result = train_policy(
            data_path=args.data,
            output_dir=trial_dir,
            examples=examples,
            episodes=args.episodes_per_trial,
            batch_size=args.batch_size,
            seed=args.seed + trial_index,
            reward_weights=candidate["weights"],
            run_name=f"{args.output_root.name}_{candidate['name']}",
            record_db=True,
        )
        metrics = dict(result["metrics"])
        summary = {
            "trial": trial_index,
            "name": candidate["name"],
            "weights": candidate["weights"],
            "score": score_trial(metrics),
            "metrics": metrics,
            "output": result["output"],
        }
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No reward trials were executed.")

    best = max(summaries, key=lambda item: float(item["score"]))
    payload = {
        "data": str(args.data),
        "time_budget_minutes": args.time_budget_minutes,
        "episodes_per_trial": args.episodes_per_trial,
        "batch_size": args.batch_size,
        "trial_count": len(summaries),
        "best": best,
        "trials": summaries,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }

    summary_path = args.output_root / "reward_sweep_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_training_run(
        run_type="retrieval_reward_sweep",
        model_path=str(Path(best["output"]) / "retrieval_policy_dqn.pt"),
        data_path=str(args.data),
        output_path=str(summary_path),
        status="completed",
        notes=f"Reward sweep completed. best={best['name']} score={best['score']}",
        metrics=payload,
    )
    print(json.dumps({"ok": True, "summary": str(summary_path), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
