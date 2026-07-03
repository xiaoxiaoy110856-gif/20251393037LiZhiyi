from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.storage.db import save_eval_run
from backend.retrieval.knowledge_store import SOURCE_ALIASES, build_context_block, search_knowledge
from backend.llm.service import chat_reply


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


def source_hit(results: list[dict[str, Any]], expected_sources: list[str]) -> float:
    if not expected_sources:
        return 1.0
    haystack = _normalize(" ".join((item.get("path", "") + " " + item.get("title", "")) for item in results))
    hits = 0
    for source in expected_sources:
        aliases = SOURCE_ALIASES.get(_normalize(source), [source])
        if any(_normalize(alias) in haystack for alias in aliases):
            hits += 1
    return hits / len(expected_sources)


def topic_hit(results: list[dict[str, Any]], expected_topics: list[str]) -> float:
    if not expected_topics:
        return 1.0
    topics = {topic.lower() for item in results for topic in item.get("topics", [])}
    hits = sum(1 for topic in expected_topics if topic.lower() in topics)
    return hits / len(expected_topics)


def point_recall(answer: str, expected_points: list[str]) -> float:
    if not expected_points:
        return 1.0
    normalized = _normalize(answer)
    hits = sum(1 for point in expected_points if _normalize(point) in normalized)
    return hits / len(expected_points)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and grounded answering.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rag_eval_results.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--answer", action="store_true", help="Generate answers in addition to retrieval evaluation.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    examples = read_jsonl(args.data)
    if args.limit > 0:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"No RAG evaluation samples found in {args.data}")

    details: list[dict[str, Any]] = []
    source_total = 0.0
    topic_total = 0.0
    point_total = 0.0

    for index, example in enumerate(examples, start=1):
        query = str(example.get("query") or example.get("task") or "").strip()
        expected_sources = [str(item) for item in example.get("expected_sources", [])]
        expected_topics = [str(item) for item in example.get("expected_topics", [])]
        expected_points = [str(item) for item in example.get("expected_points", [])]
        results = search_knowledge(query, top_k=args.top_k)
        source_score = source_hit(results, expected_sources)
        topic_score = topic_hit(results, expected_topics)
        answer = ""
        point_score = 0.0
        if args.answer:
            context_block = build_context_block(results)
            answer = chat_reply(query, history=[], context_block=context_block)
            point_score = point_recall(answer, expected_points)
        source_total += source_score
        topic_total += topic_score
        point_total += point_score
        details.append(
            {
                "index": index,
                "query": query,
                "expected_sources": expected_sources,
                "expected_topics": expected_topics,
                "expected_points": expected_points,
                "retrieved_titles": [item.get("title", "") for item in results],
                "retrieved_paths": [item.get("path", "") for item in results],
                "retrieved_topics": [item.get("topics", []) for item in results],
                "source_hit": round(source_score, 4),
                "topic_hit": round(topic_score, 4),
                "answer_point_recall": round(point_score, 4),
                "answer": answer,
            }
        )

    count = len(details)
    summary = {
        "count": count,
        "answerMode": args.answer,
        "metrics": {
            "source_hit": round(source_total / count, 4),
            "topic_hit": round(topic_total / count, 4),
            "answer_point_recall": round(point_total / count, 4) if args.answer else None,
        },
    }
    payload = {"summary": summary, "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_eval_run(
        eval_type="rag",
        dataset_path=str(args.data),
        output_path=str(args.output),
        metrics=summary["metrics"],
        details=details,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
