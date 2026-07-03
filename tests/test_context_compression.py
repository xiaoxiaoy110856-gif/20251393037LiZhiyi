from __future__ import annotations

import os
import unittest
from typing import Any

from backend.context_assembler import ContextAssembler
from backend.context_state import ConversationStateManager
from backend.context_utils import TokenCounter, redact_sensitive_text, token_counter
from backend.conversation_compressor import ConversationCompressor


class FakeRepository:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.summaries: list[dict[str, Any]] = []
        self.state: dict[str, Any] | None = None
        self.logs: list[dict[str, Any]] = []
        self.available = True

    def mysql_available(self) -> bool:
        return self.available

    def fetch_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return list(self.messages)

    def update_message_token_counts(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages

    def fetch_state(self, conversation_id: str) -> dict[str, Any] | None:
        return self.state

    def upsert_state(self, conversation_id: str, state_json: dict[str, Any], last_compressed_message_id: int, version: str) -> None:
        self.state = {
            "state_json": state_json,
            "last_compressed_message_id": last_compressed_message_id,
            "version": version,
            "token_count": token_counter.count_text_tokens(str(state_json)),
        }

    def insert_summary(self, **kwargs: Any) -> int:
        row = dict(kwargs)
        row["id"] = len(self.summaries) + 1
        row["token_count"] = token_counter.count_text_tokens(row["content"])
        self.summaries.append(row)
        return row["id"]

    def fetch_summaries(self, conversation_id: str, summary_type: str = "segment") -> list[dict[str, Any]]:
        return list(self.summaries)

    def log_context_build(self, conversation_id: str, request_id: str, recent_ids: list[int], summary_ids: list[int], relevant_ids: list[int], total_tokens: int) -> None:
        self.logs.append({"recent": recent_ids, "summaries": summary_ids, "relevant": relevant_ids, "tokens": total_tokens})


def make_messages(count: int, content: str = "用户要求实现 MySQL 上下文压缩 token password=123456") -> list[dict[str, Any]]:
    rows = []
    for index in range(1, count + 1):
        rows.append(
            {
                "id": index,
                "position": index,
                "role": "user" if index % 2 else "assistant",
                "content": f"{content} #{index}",
                "token_count": token_counter.count_text_tokens(content) + 10,
            }
        )
    return rows


class ContextCompressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_env = os.environ.copy()
        os.environ["CONTEXT_ENABLE_COMPRESSION"] = "true"
        os.environ["CONTEXT_COMPRESSION_TRIGGER_TOKENS"] = "10"
        os.environ["CONTEXT_RECENT_MESSAGE_COUNT"] = "4"
        os.environ["CONTEXT_SEGMENT_MESSAGE_COUNT"] = "5"
        os.environ["CONTEXT_MAX_INPUT_TOKENS"] = "400"
        os.environ["CONTEXT_RETRIEVAL_MAX_ITEMS"] = "4"
        os.environ["CONTEXT_RETRIEVAL_MAX_TOKENS"] = "200"

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.old_env)

    def test_token_counter_counts_text(self) -> None:
        self.assertGreater(TokenCounter().count_text_tokens("你好 hello world"), 0)

    def test_message_token_count_written_by_repository_update(self) -> None:
        repo = FakeRepository()
        repo.messages = make_messages(2)
        repo.messages[0]["token_count"] = 0
        result = ConversationCompressor(repository=repo, summarizer=lambda segment: {"summary": "s", "source_message_ids": [m["id"] for m in segment]}).maybe_compress("c1")
        self.assertTrue(repo.messages[0]["token_count"] > 0)

    def test_trigger_compression_and_recent_not_compressed(self) -> None:
        repo = FakeRepository()
        repo.messages = make_messages(14)
        compressor = ConversationCompressor(repository=repo, summarizer=lambda segment: {"summary": "MySQL context compression", "source_message_ids": [m["id"] for m in segment]})
        result = compressor.maybe_compress("c1")
        self.assertTrue(result["compressed"])
        compressed_ids = [mid for summary in repo.summaries for mid in summary["source_message_ids"]]
        self.assertNotIn(14, compressed_ids)
        self.assertNotIn(13, compressed_ids)

    def test_summary_contains_range_and_source_ids(self) -> None:
        repo = FakeRepository()
        repo.messages = make_messages(10)
        ConversationCompressor(repository=repo, summarizer=lambda segment: {"summary": "segment", "source_message_ids": [m["id"] for m in segment]}).maybe_compress("c1")
        summary = repo.summaries[0]
        self.assertIn("start_message_id", summary)
        self.assertIn("end_message_id", summary)
        self.assertTrue(summary["source_message_ids"])

    def test_state_manager_updates_rolling_state(self) -> None:
        state = ConversationStateManager().merge_summary({}, {"summary": "goal", "user_requirements": [{"content": "keep raw messages"}]}, 5)
        self.assertEqual(state["last_updated_from_message_id"], 5)
        self.assertEqual(state["user_requirements"][0]["content"], "keep raw messages")

    def test_assembler_does_not_include_all_history_and_includes_state_recent_relevant_summary(self) -> None:
        repo = FakeRepository()
        repo.messages = make_messages(20, content="old login MySQL context")
        repo.state = {"state_json": {"current_goal": "compress context", "user_requirements": []}, "last_compressed_message_id": 10}
        repo.summaries = [
            {"id": 1, "start_message_id": 1, "end_message_id": 10, "content": "login logic summary", "structured_json": {}, "token_count": 10}
        ]
        assembled = ContextAssembler(repository=repo).assemble("c1", "where is login")
        text = "\n".join(message["content"] for message in assembled["messages"])
        self.assertIn("Conversation rolling state", text)
        self.assertIn("login logic summary", text)
        self.assertIn("#20", text)
        self.assertNotIn("#10", text)
        self.assertLess(len(assembled["metadata"]["selected_relevant_message_ids"]), 8)
        self.assertLessEqual(assembled["metadata"]["total_input_tokens"], 400)

    def test_tool_call_and_tool_result_recent_window_kept_together_when_recent(self) -> None:
        repo = FakeRepository()
        repo.messages = make_messages(6)
        repo.messages[-2]["role"] = "assistant"
        repo.messages[-2]["content"] = '{"type":"tool_use","name":"search_text"}'
        repo.messages[-1]["role"] = "tool"
        repo.messages[-1]["content"] = '{"type":"tool_result","output":[]}'
        assembled = ContextAssembler(repository=repo).assemble("c1", "continue")
        roles = [message["role"] for message in assembled["messages"]]
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)

    def test_sensitive_info_redacted(self) -> None:
        self.assertIn("[REDACTED]", redact_sensitive_text("OPENAI_API_KEY=sk-abcdef1234567890"))

    def test_compression_failure_does_not_raise(self) -> None:
        repo = FakeRepository()
        repo.messages = make_messages(10)
        result = ConversationCompressor(repository=repo, summarizer=lambda segment: (_ for _ in ()).throw(RuntimeError("boom"))).maybe_compress("c1")
        self.assertFalse(result["compressed"])

    def test_disabled_compression_fallback(self) -> None:
        os.environ["CONTEXT_ENABLE_COMPRESSION"] = "false"
        repo = FakeRepository()
        repo.messages = make_messages(10)
        result = ConversationCompressor(repository=repo, summarizer=lambda segment: {"summary": "x"}).maybe_compress("c1")
        self.assertEqual(result["reason"], "disabled_or_no_mysql")
        assembled = ContextAssembler(repository=repo).assemble("c1", "hello", history_fallback=repo.messages)
        self.assertEqual(assembled["metadata"]["mode"], "fallback")

    def test_repeated_compressor_does_not_duplicate_same_batch(self) -> None:
        repo = FakeRepository()
        repo.messages = make_messages(12)
        compressor = ConversationCompressor(repository=repo, summarizer=lambda segment: {"summary": "context", "source_message_ids": [m["id"] for m in segment]})
        compressor.maybe_compress("c1")
        first_count = len(repo.summaries)
        compressor.maybe_compress("c1")
        self.assertEqual(len(repo.summaries), first_count)


if __name__ == "__main__":
    unittest.main()
