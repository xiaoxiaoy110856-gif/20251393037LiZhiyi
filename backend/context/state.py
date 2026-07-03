from __future__ import annotations

from typing import Any

from backend.context.utils import token_counter
from backend.settings import context_summary_max_tokens


CONTEXT_VERSION = "context_compression_v1"


DEFAULT_STATE = {
    "current_goal": "",
    "user_requirements": [],
    "decisions": [],
    "constraints": [],
    "open_tasks": [],
    "important_entities": [],
    "user_preferences": [],
    "tool_context": [],
    "files_or_artifacts": [],
    "last_updated_from_message_id": 0,
}


class ConversationStateManager:
    def default_state(self) -> dict[str, Any]:
        return {key: (list(value) if isinstance(value, list) else value) for key, value in DEFAULT_STATE.items()}

    def merge_summary(self, state: dict[str, Any] | None, summary: dict[str, Any], end_message_id: int) -> dict[str, Any]:
        merged = self.default_state()
        if state:
            for key, value in state.items():
                merged[key] = value

        if summary.get("summary") and not merged.get("current_goal"):
            merged["current_goal"] = str(summary.get("summary", ""))[:500]

        for key in [
            "user_requirements",
            "decisions",
            "constraints",
            "open_tasks",
            "important_entities",
            "user_preferences",
            "files_or_artifacts",
            "risks",
        ]:
            existing = merged.setdefault(key, [])
            seen = {self._item_key(item) for item in existing}
            for item in summary.get(key, []) or []:
                marker = self._item_key(item)
                if marker and marker not in seen:
                    existing.append(item)
                    seen.add(marker)
            merged[key] = existing[:40]

        merged["last_updated_from_message_id"] = max(int(merged.get("last_updated_from_message_id") or 0), int(end_message_id))
        return self._trim_state(merged)

    def _item_key(self, item: Any) -> str:
        if isinstance(item, dict):
            return str(item.get("content") or item.get("name") or item)[:240]
        return str(item)[:240]

    def _trim_state(self, state: dict[str, Any]) -> dict[str, Any]:
        budget = context_summary_max_tokens()
        while token_counter.count_text_tokens(str(state)) > budget:
            trimmed = False
            for key in ["open_tasks", "user_requirements", "decisions", "constraints", "important_entities"]:
                if isinstance(state.get(key), list) and len(state[key]) > 8:
                    state[key] = state[key][-max(8, len(state[key]) - 4) :]
                    trimmed = True
            if not trimmed:
                break
        return state
