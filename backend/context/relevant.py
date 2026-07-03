from __future__ import annotations

import re
from typing import Any

from backend.context.utils import token_counter
from backend.settings import context_retrieval_max_items, context_retrieval_max_tokens


def _keywords(text: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z_][A-Za-z0-9_\-]{2,}", text or "")
    seen = []
    for term in terms:
        lowered = term.lower()
        if lowered not in seen:
            seen.append(lowered)
        if len(seen) >= 16:
            break
    return seen


class RelevantContextRetriever:
    def score_text(self, query: str, text: str) -> int:
        terms = _keywords(query)
        haystack = (text or "").lower()
        return sum(1 for term in terms if term in haystack)

    def retrieve(
        self,
        query: str,
        summaries: list[dict[str, Any]],
        old_messages: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        max_items = context_retrieval_max_items()
        max_tokens = context_retrieval_max_tokens()
        summary_rows = []
        for item in summaries:
            text = f"{item.get('content', '')} {item.get('structured_json', '')}"
            score = self.score_text(query, text)
            if score:
                row = dict(item)
                row["score"] = score
                summary_rows.append(row)
        message_rows = []
        for item in old_messages:
            score = self.score_text(query, str(item.get("content", "")))
            if score:
                row = dict(item)
                row["score"] = score
                message_rows.append(row)
        summary_rows.sort(key=lambda item: item["score"], reverse=True)
        message_rows.sort(key=lambda item: item["score"], reverse=True)

        selected_summaries: list[dict[str, Any]] = []
        selected_messages: list[dict[str, Any]] = []
        used_tokens = 0
        for item in summary_rows[:max_items]:
            cost = int(item.get("token_count") or token_counter.count_text_tokens(str(item.get("content", ""))))
            if used_tokens + cost > max_tokens:
                break
            selected_summaries.append(item)
            used_tokens += cost
        for item in message_rows[:max_items]:
            cost = int(item.get("token_count") or token_counter.count_text_tokens(str(item.get("content", ""))))
            if used_tokens + cost > max_tokens:
                break
            selected_messages.append(item)
            used_tokens += cost
        return {"summaries": selected_summaries, "messages": selected_messages}
