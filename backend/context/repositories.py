from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from backend.context.utils import token_counter
from backend.storage.db import connection_scope, database_ready, init_db


def _json_load(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


class ContextRepository:
    def mysql_available(self) -> bool:
        return database_ready()

    def fetch_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        if not database_ready():
            return []
        init_db()
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, position, role, content, token_count, created_at
                    FROM chat_messages
                    WHERE session_id=%s
                    ORDER BY position ASC
                    """,
                    (conversation_id,),
                )
                return [
                    {
                        "id": int(row["id"]),
                        "position": int(row["position"]),
                        "role": row["role"],
                        "content": row["content"],
                        "token_count": int(row.get("token_count") or token_counter.count_text_tokens(row["content"])),
                        "created_at": row["created_at"].isoformat(timespec="seconds") if row.get("created_at") else "",
                    }
                    for row in cursor.fetchall()
                ]

    def update_message_token_counts(self, messages: list[dict[str, Any]]) -> None:
        if not database_ready():
            return
        rows = [(int(message["token_count"]), int(message["id"])) for message in messages if message.get("id")]
        if not rows:
            return
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.executemany("UPDATE chat_messages SET token_count=%s WHERE id=%s AND token_count=0", rows)

    def fetch_state(self, conversation_id: str) -> dict[str, Any] | None:
        if not database_ready():
            return None
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, state_json, token_count, last_compressed_message_id, version
                    FROM conversation_states
                    WHERE conversation_id=%s
                    """,
                    (conversation_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": int(row["id"]),
                    "state_json": _json_load(row["state_json"], {}),
                    "token_count": int(row["token_count"] or 0),
                    "last_compressed_message_id": int(row["last_compressed_message_id"] or 0),
                    "version": row["version"],
                }

    def upsert_state(self, conversation_id: str, state_json: dict[str, Any], last_compressed_message_id: int, version: str) -> None:
        if not database_ready():
            return
        now = datetime.now()
        token_count = token_counter.count_text_tokens(json.dumps(state_json, ensure_ascii=False))
        payload = json.dumps(state_json, ensure_ascii=False)
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversation_states
                    (conversation_id, state_json, token_count, last_compressed_message_id, version, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    state_json=VALUES(state_json),
                    token_count=VALUES(token_count),
                    last_compressed_message_id=VALUES(last_compressed_message_id),
                    version=VALUES(version),
                    updated_at=VALUES(updated_at)
                    """,
                    (conversation_id, payload, token_count, int(last_compressed_message_id), version, now, now),
                )

    def insert_summary(
        self,
        conversation_id: str,
        summary_type: str,
        start_message_id: int,
        end_message_id: int,
        content: str,
        structured_json: dict[str, Any],
        source_message_ids: list[int],
        model: str,
        version: str,
    ) -> int | None:
        if not database_ready():
            return None
        now = datetime.now()
        token_count = token_counter.count_text_tokens(content)
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO conversation_summaries
                    (conversation_id, summary_type, start_message_id, end_message_id, content, structured_json,
                     token_count, source_message_ids, model, version, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        summary_type,
                        int(start_message_id),
                        int(end_message_id),
                        content,
                        json.dumps(structured_json, ensure_ascii=False),
                        token_count,
                        json.dumps(source_message_ids, ensure_ascii=False),
                        model,
                        version,
                        now,
                        now,
                    ),
                )
                return int(cursor.lastrowid)

    def fetch_summaries(self, conversation_id: str, summary_type: str = "segment") -> list[dict[str, Any]]:
        if not database_ready():
            return []
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, summary_type, start_message_id, end_message_id, content, structured_json,
                           token_count, source_message_ids, model, version
                    FROM conversation_summaries
                    WHERE conversation_id=%s AND summary_type=%s
                    ORDER BY start_message_id ASC
                    """,
                    (conversation_id, summary_type),
                )
                return [
                    {
                        "id": int(row["id"]),
                        "summary_type": row["summary_type"],
                        "start_message_id": int(row["start_message_id"]),
                        "end_message_id": int(row["end_message_id"]),
                        "content": row["content"],
                        "structured_json": _json_load(row["structured_json"], {}),
                        "token_count": int(row["token_count"] or 0),
                        "source_message_ids": _json_load(row["source_message_ids"], []),
                        "model": row["model"] or "",
                        "version": row["version"],
                    }
                    for row in cursor.fetchall()
                ]

    def log_context_build(
        self,
        conversation_id: str,
        request_id: str,
        recent_ids: list[int],
        summary_ids: list[int],
        relevant_ids: list[int],
        total_tokens: int,
    ) -> None:
        if not database_ready():
            return
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO context_build_logs
                    (conversation_id, request_id, selected_recent_message_ids, selected_summary_ids,
                     selected_relevant_message_ids, total_input_tokens, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        conversation_id,
                        request_id,
                        json.dumps(recent_ids, ensure_ascii=False),
                        json.dumps(summary_ids, ensure_ascii=False),
                        json.dumps(relevant_ids, ensure_ascii=False),
                        int(total_tokens),
                        datetime.now(),
                    ),
                )
