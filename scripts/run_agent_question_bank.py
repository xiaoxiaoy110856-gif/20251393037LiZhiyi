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

from backend.app import chat_payload, create_session_payload
from backend.storage.db import save_eval_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a large ML/DL/RL question bank against the local agent.")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "agent_question_bank.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "agent_question_bank_results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions. 0 means all.")
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--category", type=str, default="", help="Optional category filter, e.g. reinforcement_learning")
    parser.add_argument("--save-every", type=int, default=1, help="Write partial results every N completed questions.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def format_seconds(seconds: float) -> str:
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def write_partial_output(
    output_path: Path,
    *,
    session_id: str,
    category: str,
    top_k: int,
    total_questions: int,
    completed_questions: int,
    results: list[dict[str, Any]],
    started_at: float,
) -> None:
    elapsed_seconds = time.time() - started_at
    average_seconds = elapsed_seconds / completed_questions if completed_questions > 0 else 0.0
    remaining_questions = max(total_questions - completed_questions, 0)
    estimated_remaining_seconds = average_seconds * remaining_questions
    payload = {
        "session_id": session_id,
        "question_count": total_questions,
        "completed_questions": completed_questions,
        "category": category,
        "top_k": top_k,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "estimated_remaining_seconds": round(estimated_remaining_seconds, 2),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data)
    if args.category:
        rows = [row for row in rows if str(row.get("category", "")).lower() == args.category.lower()]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No questions found in {args.data}")

    session = create_session_payload(title="Question Bank Evaluation")["session"]
    results: list[dict[str, Any]] = []
    total_questions = len(rows)
    started_at = time.time()

    print(
        f"Starting question bank run: {total_questions} questions, category={args.category or 'all'}, top_k={args.top_k}",
        flush=True,
    )

    for index, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        question_started_at = time.time()
        response = chat_payload(query=query, session_id=session["id"], top_k=args.top_k)
        answer = response["answer"]
        sources = response.get("sources", [])
        results.append(
            {
                "index": index,
                "id": row.get("id"),
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
                "query": query,
                "answer": answer,
                "source_count": len(sources),
                "source_titles": [item.get("title", "") for item in sources[:5]],
                "tool_trace_count": len(response.get("toolTraces", [])),
            }
        )
        completed_questions = len(results)
        elapsed_seconds = time.time() - started_at
        average_seconds = elapsed_seconds / completed_questions if completed_questions > 0 else 0.0
        remaining_questions = max(total_questions - completed_questions, 0)
        estimated_remaining_seconds = average_seconds * remaining_questions
        question_elapsed_seconds = time.time() - question_started_at

        print(
            f"[{completed_questions}/{total_questions}] {row.get('id', index)} "
            f"category={row.get('category', 'unknown')} "
            f"spent={format_seconds(question_elapsed_seconds)} "
            f"elapsed={format_seconds(elapsed_seconds)} "
            f"eta={format_seconds(estimated_remaining_seconds)} "
            f"sources={len(sources)} traces={len(response.get('toolTraces', []))}",
            flush=True,
        )

        if completed_questions % max(args.save_every, 1) == 0:
            write_partial_output(
                args.output,
                session_id=session["id"],
                category=args.category or "all",
                top_k=args.top_k,
                total_questions=total_questions,
                completed_questions=completed_questions,
                results=results,
                started_at=started_at,
            )
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    write_partial_output(
        args.output,
        session_id=session["id"],
        category=args.category or "all",
        top_k=args.top_k,
        total_questions=total_questions,
        completed_questions=len(results),
        results=results,
        started_at=started_at,
    )
    save_eval_run(
        eval_type="agent_question_bank",
        dataset_path=str(args.data),
        output_path=str(args.output),
        metrics={"question_count": len(results), "category": args.category or "all", "top_k": args.top_k},
        details=results,
    )
    print(json.dumps({"ok": True, "session_id": session["id"], "output": str(args.output), "count": len(results)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
