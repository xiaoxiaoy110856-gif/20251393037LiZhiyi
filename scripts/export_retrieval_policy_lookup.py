from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.retrieval.rl_env import ACTIONS, features_for_query


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def action_index(name: str) -> int:
    lowered = name.strip().lower()
    for index, action in enumerate(ACTIONS):
        if action.name == lowered:
            return index
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a torch-free lookup retrieval policy from evaluation.json.")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    evaluation = read_json(args.evaluation)
    episodes = ((evaluation.get("trained_policy") or {}).get("episodes") or []) if isinstance(evaluation.get("trained_policy"), dict) else []
    if not episodes:
        raise SystemExit(f"No trained episodes found in {args.evaluation}")
    rules = []
    for row in episodes:
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        rules.append(
            {
                "query": query,
                "features": features_for_query(query),
                "action": str(row.get("chosen_action") or "baseline"),
                "action_index": action_index(str(row.get("chosen_action") or "baseline")),
                "reward": float(row.get("reward") or 0),
            }
        )
    output = args.output or (args.evaluation.parent / "retrieval_policy_lookup.json")
    output.write_text(
        json.dumps(
            {
                "algorithm": "lookup",
                "source_evaluation": str(args.evaluation),
                "actions": [action.__dict__ for action in ACTIONS],
                "rules": rules,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "count": len(rules), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
