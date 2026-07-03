from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize_prompt(row: dict[str, Any]) -> str:
    return str(row.get("prompt") or row.get("task") or row.get("query") or row.get("instruction") or "").strip()


def generic_rejected(prompt: str, expected_points: list[str]) -> str:
    topic_hint = "、".join(expected_points[:3]) if expected_points else "强化学习和轨迹任务"
    return (
        f"这个问题大致和{topic_hint}有关，但我没有检索或引用项目知识库。"
        "可以后续再补充细节。"
    )


def from_preference_jsonl(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for row in read_jsonl(path):
        prompt = normalize_prompt(row)
        chosen = str(row.get("chosen") or row.get("preferred") or "").strip()
        rejected = str(row.get("rejected") or row.get("dispreferred") or "").strip()
        if prompt and chosen and rejected:
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def from_agent_eval(path: Path, min_score: float = 0.5) -> list[dict[str, str]]:
    payload = read_json(path)
    details = payload.get("details", []) if isinstance(payload.get("details"), list) else []
    pairs: list[dict[str, str]] = []
    for row in details:
        prompt = normalize_prompt(row)
        chosen = str(row.get("answer") or "").strip()
        expected_points = [str(item) for item in row.get("expected_points", [])]
        keyword_score = float(row.get("keyword_recall") or 0.0)
        grounding_score = float(row.get("grounding_score") or 0.0)
        if not prompt or not chosen or keyword_score < min_score:
            continue
        rejected = generic_rejected(prompt, expected_points)
        if grounding_score <= 0:
            rejected = f"{rejected}\n\n注意：这个回答没有可靠证据支撑。"
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def from_sft_jsonl(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for row in read_jsonl(path):
        prompt = normalize_prompt(row)
        chosen = str(row.get("output") or row.get("answer") or "").strip()
        if not prompt or not chosen:
            continue
        points = [token for token in re.findall(r"PPO|DPO|DQN|SAC|reward|trajectory|轨迹|奖励|策略", prompt, flags=re.IGNORECASE)]
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": generic_rejected(prompt, points)})
    return pairs


def dedupe_pairs(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for row in pairs:
        key = (row["prompt"], row["chosen"], row["rejected"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO chosen/rejected pairs for the local RL assistant.")
    parser.add_argument("--preference-data", type=Path, default=ROOT / "training_data" / "assistant_preference.jsonl")
    parser.add_argument("--agent-eval", type=Path, default=ROOT / "outputs" / "agent_eval_results.json")
    parser.add_argument("--sft-data", type=Path, default=ROOT / "training_data" / "assistant_sft.example.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "training_data" / "assistant_dpo.jsonl")
    parser.add_argument("--min-agent-score", type=float, default=0.5)
    args = parser.parse_args()

    pairs: list[dict[str, str]] = []
    pairs.extend(from_preference_jsonl(args.preference_data))
    pairs.extend(from_agent_eval(args.agent_eval, min_score=args.min_agent_score))
    pairs.extend(from_sft_jsonl(args.sft_data))
    pairs = dedupe_pairs([row for row in pairs if row.get("prompt") and row.get("chosen") and row.get("rejected")])
    if not pairs:
        raise SystemExit("No DPO pairs were built. Add training_data/assistant_preference.jsonl or run agent eval first.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in pairs) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "count": len(pairs), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
