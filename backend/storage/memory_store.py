from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.context.utils import token_counter
from backend.storage.db import connection_scope, database_ready
from backend.settings import ensure_runtime_dirs, get_conversations_dir


def _sessions_file() -> Path:
    ensure_runtime_dirs()
    return get_conversations_dir() / "sessions.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_store() -> dict[str, Any]:
    return {"active_session_id": None, "sessions": []}


def load_store() -> dict[str, Any]:
    path = _sessions_file()
    if not path.exists():
        return _default_store()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _default_store()
    payload.setdefault("active_session_id", None)
    payload.setdefault("sessions", [])
    return payload


def save_store(payload: dict[str, Any]) -> None:
    _sessions_file().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_summary(messages: list[dict[str, str]]) -> str:
    recent = messages[-6:]
    parts = []
    for item in recent:
        role = "user" if item.get("role") == "user" else "assistant"
        content = (item.get("content") or "").strip().replace("\n", " ")
        if content:
            parts.append(f"{role}: {content[:100]}")
    return " | ".join(parts)[:400]


def _db_fetch_messages(cursor, session_id: str) -> list[dict[str, str]]:
    cursor.execute(
        """
        SELECT id, position, role, content, token_count
        FROM chat_messages
        WHERE session_id=%s
        ORDER BY position ASC
        """,
        (session_id,),
    )
    return [
        {
            "id": int(row["id"]),
            "position": int(row["position"]),
            "role": row["role"],
            "content": row["content"],
            "token_count": int(row.get("token_count") or token_counter.count_text_tokens(row["content"])),
        }
        for row in cursor.fetchall()
    ]


def list_sessions() -> list[dict[str, Any]]:
    if database_ready():
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, title, summary, created_at, updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                    """
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "messages": [],
                        "summary": row["summary"] or "",
                        "created_at": row["created_at"].isoformat(timespec="seconds") if row["created_at"] else "",
                        "updated_at": row["updated_at"].isoformat(timespec="seconds") if row["updated_at"] else "",
                    }
                    for row in rows
                ]

    store = load_store()
    sessions = store.get("sessions", [])
    for session in sessions:
        session.setdefault("title", "New Chat")
        session.setdefault("messages", [])
        session.setdefault("summary", "")
        session.setdefault("created_at", "")
        session.setdefault("updated_at", session.get("created_at", ""))
    return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)


def create_session(title: str | None = None) -> dict[str, Any]:
    if database_ready():
        session_id = uuid4().hex[:12]
        now = datetime.now()
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_sessions (id, title, summary, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_id, title or "New Chat", "", now, now),
                )
            return {
                "id": session_id,
                "title": title or "New Chat",
                "messages": [],
                "summary": "",
                "created_at": now.isoformat(timespec="seconds"),
                "updated_at": now.isoformat(timespec="seconds"),
            }

    store = load_store()
    session_id = uuid4().hex[:12]
    session = {
        "id": session_id,
        "title": title or "New Chat",
        "messages": [],
        "summary": "",
        "created_at": _now(),
        "updated_at": _now(),
    }
    store["sessions"].append(session)
    store["active_session_id"] = session_id
    save_store(store)
    return session


def get_session(session_id: str | None) -> dict[str, Any]:
    if database_ready():
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                if session_id:
                    cursor.execute(
                        """
                        SELECT id, title, summary, created_at, updated_at
                        FROM chat_sessions
                        WHERE id=%s
                        """,
                        (session_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        return {
                            "id": row["id"],
                            "title": row["title"],
                            "messages": _db_fetch_messages(cursor, row["id"]),
                            "summary": row["summary"] or "",
                            "created_at": row["created_at"].isoformat(timespec="seconds") if row["created_at"] else "",
                            "updated_at": row["updated_at"].isoformat(timespec="seconds") if row["updated_at"] else "",
                        }
                cursor.execute(
                    """
                    SELECT id, title, summary, created_at, updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "title": row["title"],
                        "messages": _db_fetch_messages(cursor, row["id"]),
                        "summary": row["summary"] or "",
                        "created_at": row["created_at"].isoformat(timespec="seconds") if row["created_at"] else "",
                        "updated_at": row["updated_at"].isoformat(timespec="seconds") if row["updated_at"] else "",
                    }
        return create_session()

    store = load_store()
    target_id = session_id or store.get("active_session_id")
    for session in store.get("sessions", []):
        if session["id"] == target_id:
            return session
    return create_session()


def append_turn(session_id: str, user_message: str, assistant_message: str) -> dict[str, Any]:
    if database_ready():
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, title
                    FROM chat_sessions
                    WHERE id=%s
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    created = create_session(user_message[:24] if user_message.strip() else None)
                    return append_turn(created["id"], user_message, assistant_message)

                cursor.execute("SELECT COALESCE(MAX(position), 0) AS max_position FROM chat_messages WHERE session_id=%s", (session_id,))
                max_position = int((cursor.fetchone() or {}).get("max_position") or 0)
                now = datetime.now()
                user_tokens = token_counter.count_text_tokens(user_message)
                assistant_tokens = token_counter.count_text_tokens(assistant_message)
                cursor.execute(
                    """
                    INSERT INTO chat_messages (session_id, position, role, content, token_count, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s), (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id,
                        max_position + 1,
                        "user",
                        user_message,
                        user_tokens,
                        now,
                        session_id,
                        max_position + 2,
                        "assistant",
                        assistant_message,
                        assistant_tokens,
                        now,
                    ),
                )
                messages = _db_fetch_messages(cursor, session_id)
                title = row["title"]
                if title == "New Chat" and user_message.strip():
                    title = user_message.strip()[:24]
                summary = _build_summary(messages)
                cursor.execute(
                    """
                    UPDATE chat_sessions
                    SET title=%s, summary=%s, updated_at=%s
                    WHERE id=%s
                    """,
                    (title, summary, now, session_id),
                )
                return {
                    "id": session_id,
                    "title": title,
                    "messages": messages,
                    "summary": summary,
                    "created_at": "",
                    "updated_at": now.isoformat(timespec="seconds"),
                }

    store = load_store()
    for session in store.get("sessions", []):
        if session["id"] != session_id:
            continue
        session["messages"].append({"role": "user", "content": user_message})
        session["messages"].append({"role": "assistant", "content": assistant_message})
        if session["title"] == "New Chat" and user_message.strip():
            session["title"] = user_message.strip()[:24]
        session["summary"] = _build_summary(session["messages"])
        session["updated_at"] = _now()
        store["active_session_id"] = session_id
        save_store(store)
        return session
    session = create_session(user_message[:24] if user_message.strip() else None)
    return append_turn(session["id"], user_message, assistant_message)


def clear_session(session_id: str) -> dict[str, Any]:
    if database_ready():
        with connection_scope() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, title, created_at FROM chat_sessions WHERE id=%s", (session_id,))
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(f"Session not found: {session_id}")
                cursor.execute("DELETE FROM chat_messages WHERE session_id=%s", (session_id,))
                now = datetime.now()
                cursor.execute(
                    """
                    UPDATE chat_sessions
                    SET summary=%s, updated_at=%s
                    WHERE id=%s
                    """,
                    ("", now, session_id),
                )
                return {
                    "id": session_id,
                    "title": row["title"],
                    "messages": [],
                    "summary": "",
                    "created_at": row["created_at"].isoformat(timespec="seconds") if row["created_at"] else "",
                    "updated_at": now.isoformat(timespec="seconds"),
                }

    store = load_store()
    for session in store.get("sessions", []):
        if session["id"] == session_id:
            session["messages"] = []
            session["summary"] = ""
            session["updated_at"] = _now()
            save_store(store)
            return session
    raise KeyError(f"Session not found: {session_id}")
