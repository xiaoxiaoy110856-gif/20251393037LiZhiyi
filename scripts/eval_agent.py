from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.loop import agent_chat
from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.storage.db import save_eval_run


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def keyword_recall(answer: str, expected_points: list[str]) -> float:
    if not expected_points:
        return 1.0
    normalized_answer = _normalize(answer)
    hits = sum(1 for item in expected_points if _normalize(item) in normalized_answer)
    return hits / len(expected_points)


def tool_recall(tools_used: list[str], expected_tools: list[str]) -> float:
    if not expected_tools:
        return 1.0
    used = {item.lower() for item in tools_used}
    hits = sum(1 for item in expected_tools if item.lower() in used)
    return hits / len(expected_tools)


def has_grounding(sources: list[dict[str, Any]]) -> float:
    return 1.0 if sources else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local trajectory/RL agent.")
    parser.add_argument("--data", type=Path, required=True, help="Agent evaluation dataset in JSONL format.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "agent_eval_results.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    examples = read_jsonl(args.data)
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"No agent evaluation samples found in {args.data}")

    details: list[dict[str, Any]] = []
    keyword_total = 0.0
    tool_total = 0.0
    grounding_total = 0.0

    for index, example in enumerate(examples, start=1):
        task = str(example.get("task") or example.get("query") or example.get("instruction") or "").strip()
        expected_points = [str(item) for item in example.get("expected_points", [])]
        expected_tools = [str(item) for item in example.get("expected_tools", [])]
        result = agent_chat(task, history=[], top_k=args.top_k)
        tools_used = [trace["name"] for trace in result.get("tool_traces", [])]
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        keyword_score = keyword_recall(answer, expected_points)
        tool_score = tool_recall(tools_used, expected_tools)
        grounding_score = has_grounding(sources)

        keyword_total += keyword_score
        tool_total += tool_score
        grounding_total += grounding_score

        details.append(
            {
                "index": index,
                "task": task,
                "expected_points": expected_points,
                "expected_tools": expected_tools,
                "answer": answer,
                "tools_used": tools_used,
                "source_count": len(sources),
                "keyword_recall": round(keyword_score, 4),
                "tool_recall": round(tool_score, 4),
                "grounding_score": round(grounding_score, 4),
            }
        )

    count = len(details)
    summary = {
        "count": count,
        "metrics": {
            "keyword_recall": round(keyword_total / count, 4),
            "tool_recall": round(tool_total / count, 4),
            "grounding_score": round(grounding_total / count, 4),
        },
    }
    payload = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_eval_run(
        eval_type="agent",
        dataset_path=str(args.data),
        output_path=str(args.output),
        metrics=summary["metrics"],
        details=details,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
