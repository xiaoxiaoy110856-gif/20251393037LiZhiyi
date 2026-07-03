from __future__ import annotations

import json
from uuid import uuid4
from typing import Any

from backend.context.repositories import ContextRepository
from backend.context.utils import token_counter
from backend.context.relevant import RelevantContextRetriever
from backend.settings import (
    context_build_log_enabled,
    context_compression_enabled,
    context_max_input_tokens,
    context_recent_message_count,
)

# 上下文组装器：MySQL 仍保存全部原始消息，这里只决定本轮发送给模型的较短上下文。


class ContextAssembler:
    """按 token 预算组装 rolling state、相关摘要、相关旧消息和最近消息。"""
    def __init__(self, repository: ContextRepository | None = None, retriever: RelevantContextRetriever | None = None) -> None:
        self.repository = repository or ContextRepository()
        self.retriever = retriever or RelevantContextRetriever()

    def assemble(
        self,
        conversation_id: str,
        current_user_message: str,
        history_fallback: list[dict[str, Any]] | None = None,
        max_input_tokens: int | None = None,
    ) -> dict[str, Any]:
        # 核心2：模型生成前调用，避免把完整历史都发给 Qwen；返回压缩后的消息列表和选择依据。
        # 兼容模式：MySQL 或压缩关闭时，只裁剪最近历史，不依赖摘要表。
        budget = max_input_tokens or context_max_input_tokens()
        if not context_compression_enabled() or not self.repository.mysql_available():
            messages = self._trim_messages(history_fallback or [], budget - token_counter.count_text_tokens(current_user_message))
            return {"messages": messages, "metadata": {"mode": "fallback", "total_input_tokens": token_counter.count_messages_tokens(messages)}}

        messages = self.repository.fetch_messages(conversation_id)
        for message in messages:
            if not message.get("token_count"):
                message["token_count"] = token_counter.count_text_tokens(str(message.get("content", "")))
        recent = messages[-context_recent_message_count():]
        recent_ids = [int(message["id"]) for message in recent]
        old_messages = [message for message in messages if int(message["id"]) not in set(recent_ids)]
        summaries = self.repository.fetch_summaries(conversation_id)
        state = self.repository.fetch_state(conversation_id)
        relevant = self.retriever.retrieve(current_user_message, summaries, old_messages)

        # Priority order: rolling state, relevant summaries, relevant old raw
        # messages, then the recent window. The current user message is budgeted
        # separately by the caller and is never dropped here.
        assembled: list[dict[str, str]] = []
        selected_summary_ids: list[int] = []
        selected_relevant_ids: list[int] = []

        if state and state.get("state_json"):
            assembled.append(
                {
                    "role": "system",
                    "content": "[Conversation rolling state]\n" + json.dumps(state["state_json"], ensure_ascii=False, indent=2),
                }
            )

        for summary in relevant["summaries"]:
            message = {
                "role": "system",
                "content": (
                    f"[Relevant conversation summary #{summary['id']} covering messages "
                    f"{summary['start_message_id']}-{summary['end_message_id']}]\n{summary['content']}"
                ),
            }
            if self._can_add(assembled, message, budget, current_user_message):
                assembled.append(message)
                selected_summary_ids.append(int(summary["id"]))

        for old in relevant["messages"]:
            message = {"role": old["role"], "content": f"[Relevant earlier message id={old['id']}]\n{old['content']}"}
            if self._can_add(assembled, message, budget, current_user_message):
                assembled.append(message)
                selected_relevant_ids.append(int(old["id"]))

        recent_messages = [{"role": item["role"], "content": item["content"]} for item in recent]
        assembled.extend(self._trim_messages(recent_messages, max(0, budget - token_counter.count_messages_tokens(assembled) - token_counter.count_text_tokens(current_user_message))))
        total = token_counter.count_messages_tokens(assembled) + token_counter.count_text_tokens(current_user_message)
        metadata = {
            "mode": "compressed",
            "selected_recent_message_ids": recent_ids,
            "selected_summary_ids": selected_summary_ids,
            "selected_relevant_message_ids": selected_relevant_ids,
            "total_input_tokens": total,
        }
        if context_build_log_enabled():
            self.repository.log_context_build(conversation_id, uuid4().hex[:12], recent_ids, selected_summary_ids, selected_relevant_ids, total)
        return {"messages": assembled, "metadata": metadata}

    def _can_add(self, messages: list[dict[str, str]], candidate: dict[str, str], budget: int, current_user_message: str) -> bool:
        """判断候选上下文加入后是否仍在 token 预算内。"""
        projected = messages + [candidate]
        return token_counter.count_messages_tokens(projected) + token_counter.count_text_tokens(current_user_message) <= budget

    def _trim_messages(self, messages: list[dict[str, Any]], budget: int) -> list[dict[str, str]]:
        """在 token 预算有限时从后往前保留最近消息。"""
        selected: list[dict[str, str]] = []
        used = 0
        for message in reversed(messages):
            normalized = {"role": str(message.get("role", "user")), "content": str(message.get("content", ""))}
            cost = token_counter.count_messages_tokens([normalized])
            if selected and used + cost > budget:
                break
            if used + cost <= budget:
                selected.append(normalized)
                used += cost
        selected.reverse()
        return selected
