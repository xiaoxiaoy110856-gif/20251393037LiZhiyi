from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from backend.context.repositories import ContextRepository
from backend.context.state import CONTEXT_VERSION, ConversationStateManager
from backend.context.utils import COMPRESSION_PROMPT, redact_sensitive_text, token_counter
from backend.llm.service import run_messages
from backend.settings import (
    context_compression_enabled,
    context_compression_trigger_tokens,
    context_recent_message_count,
    context_segment_message_count,
    get_ollama_model,
)


logger = logging.getLogger(__name__)

Summarizer = Callable[[list[dict[str, Any]]], dict[str, Any]]

# ConversationCompressor summarizes old messages into segment summaries and a
# rolling state. It never deletes raw chat_messages; compression is an extra
# index used later by ContextAssembler.


def _extract_json(text: str) -> dict[str, Any]:
    """从 Qwen 生成的压缩结果中解析 JSON，兼容 ```json 代码块包裹的情况。"""
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Compression result is not a JSON object.")
    return parsed


class ConversationCompressor:
    """上下文压缩器：把旧对话压成摘要和滚动状态，但保留原始消息不删除。"""
    def __init__(
        self,
        repository: ContextRepository | None = None,
        state_manager: ConversationStateManager | None = None,
        summarizer: Summarizer | None = None,
        model_id: str | None = None,
    ) -> None:
        self.repository = repository or ContextRepository()
        self.state_manager = state_manager or ConversationStateManager()
        self.summarizer = summarizer
        self.model_id = model_id

    def maybe_compress(self, conversation_id: str) -> dict[str, Any]:
        # 核心2：压缩入口。只有开启上下文压缩且 MySQL 可用时才执行，原始聊天记录始终保留。
        if not context_compression_enabled() or not self.repository.mysql_available():
            return {"compressed": False, "reason": "disabled_or_no_mysql"}
        try:
            return self._compress(conversation_id)
        except Exception as error:
            logger.warning("context compression failed conversation_id=%s error=%s", conversation_id, error)
            return {"compressed": False, "reason": "compression_failed", "error": str(error)}

    def _compress(self, conversation_id: str) -> dict[str, Any]:
        """选择超过最近窗口的旧消息，按段总结并更新 rolling state。"""
        messages = self.repository.fetch_messages(conversation_id)
        if not messages:
            return {"compressed": False, "reason": "no_messages"}
        for message in messages:
            if not message.get("token_count"):
                message["token_count"] = token_counter.count_text_tokens(str(message.get("content", "")))
        self.repository.update_message_token_counts(messages)

        state_row = self.repository.fetch_state(conversation_id)
        state = state_row["state_json"] if state_row else self.state_manager.default_state()
        last_compressed_id = int(state_row.get("last_compressed_message_id") or 0) if state_row else 0
        recent_count = context_recent_message_count()
        compressible = [message for message in messages[:-recent_count] if int(message["id"]) > last_compressed_id]
        old_tokens = sum(int(message.get("token_count") or 0) for message in compressible)
        if old_tokens < context_compression_trigger_tokens():
            return {"compressed": False, "reason": "below_trigger", "tokens": old_tokens}

        segment_size = max(2, context_segment_message_count())
        compressed = 0
        for start in range(0, len(compressible), segment_size):
            segment = compressible[start : start + segment_size]
            if not segment:
                continue
            summary = self._summarize_segment(segment)
            source_ids = [int(message["id"]) for message in segment]
            summary["source_message_ids"] = summary.get("source_message_ids") or source_ids
            summary = self._redact_summary(summary)
            content = str(summary.get("summary") or "")
            self.repository.insert_summary(
                conversation_id=conversation_id,
                summary_type="segment",
                start_message_id=source_ids[0],
                end_message_id=source_ids[-1],
                content=content,
                structured_json=summary,
                source_message_ids=source_ids,
                model=get_ollama_model(),
                version=CONTEXT_VERSION,
            )
            state = self.state_manager.merge_summary(state, summary, source_ids[-1])
            last_compressed_id = source_ids[-1]
            compressed += len(segment)

        self.repository.upsert_state(conversation_id, state, last_compressed_id, CONTEXT_VERSION)
        return {"compressed": compressed > 0, "messages": compressed, "last_compressed_message_id": last_compressed_id}

    def _summarize_segment(self, segment: list[dict[str, Any]]) -> dict[str, Any]:
        if self.summarizer:
            return self.summarizer(segment)
        # 核心2：旧消息段由本地模型总结成结构化 JSON，后续由 ContextAssembler 重新取回使用。
        transcript = "\n".join(
            f"[message_id={message['id']} role={message['role']}]\n{redact_sensitive_text(str(message.get('content', '')))}"
            for message in segment
        )
        raw = run_messages(
            [
                {"role": "system", "content": COMPRESSION_PROMPT},
                {"role": "user", "content": transcript},
            ],
            query="Compress this conversation segment.",
            context_block="",
            model_id=self.model_id,
        )
        return _extract_json(raw)

    def _redact_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        """对摘要中的敏感片段再次脱敏，避免把密钥、路径等内容写入长期摘要。"""
        return json.loads(redact_sensitive_text(json.dumps(summary, ensure_ascii=False)))
