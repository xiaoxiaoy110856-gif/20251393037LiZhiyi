from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.retrieval.rl_env import ACTIONS, RetrievalRLEnv, build_examples_from_rag_eval

try:
    import torch
except Exception as error:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for replay: {error}")

from scripts.train_retrieval_policy import QNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one retrieval-policy episode.")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "retrieval_policy" / "retrieval_policy_dqn.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "retrieval_rl_eval.jsonl")
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def tensor_from(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    examples = build_examples_from_rag_eval(args.data)
    if not examples:
        raise SystemExit(f"No retrieval RL samples found in {args.data}")
    if not args.model.exists():
        raise SystemExit(f"Checkpoint not found: {args.model}")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    env = RetrievalRLEnv(examples)
    model = QNetwork(checkpoint["state_size"], checkpoint["action_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    index = max(0, min(args.index, len(examples) - 1))
    state = tensor_from(env.reset(index)).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)[0]
        action_index = int(torch.argmax(q_values).item())
    _, reward, _, info = env.step(action_index)

    payload = {
        "index": index,
        "query": info["query"],
        "chosen_action": info["action"],
        "action_description": info["action_description"],
        "reward": reward,
        "metrics": {
            "source_hit": info["source_hit"],
            "topic_hit": info["topic_hit"],
            "point_recall": info["point_recall"],
        },
        "retrieval_query": info["retrieval_query"],
        "top_k": info["top_k"],
        "retrieved_titles": info["retrieved_titles"],
        "retrieved_topics": info["retrieved_topics"],
        "q_values": {ACTIONS[i].name: round(float(q_values[i].item()), 4) for i in range(len(ACTIONS))},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
