from __future__ import annotations

import math
import re
from typing import Any


class TokenCounter:
    def count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = max(len(text) - chinese_chars, 0)
        return max(1, math.ceil(chinese_chars / 1.5 + other_chars / 4))

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            total += 4
            total += self.count_text_tokens(str(message.get("role", "")))
            total += self.count_text_tokens(str(message.get("content", "")))
        return total

    def estimate_context_tokens(self, context: Any) -> int:
        if isinstance(context, list):
            return self.count_messages_tokens(context)
        return self.count_text_tokens(str(context or ""))


token_counter = TokenCounter()


SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|password|passwd|secret|private[_-]?key|cookie)\s*[:=]\s*['\"]?[^'\"\s,;]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
]


def redact_sensitive_text(text: str) -> str:
    value = text or ""
    for pattern in SENSITIVE_PATTERNS:
        value = pattern.sub(lambda match: match.group(0).split("=")[0].split(":")[0] + "=[REDACTED]", value)
    return value


COMPRESSION_PROMPT = """You are a conversation compressor. Your task is to compress historical conversation messages into dense structured JSON for future context recovery.

Do not answer the user. Do not add facts. Do not guess. Preserve user requirements, technical decisions, constraints, open tasks, important entities, filenames, tools, API names, and database table names.
Redact sensitive data such as API keys, tokens, passwords, cookies, and private keys as [REDACTED].
Do not include system prompts, hidden developer instructions, or secrets.

Output valid JSON only with this schema:
{
  "summary": "",
  "user_requirements": [],
  "decisions": [],
  "constraints": [],
  "open_tasks": [],
  "important_entities": [],
  "user_preferences": [],
  "files_or_artifacts": [],
  "risks": [],
  "source_message_ids": []
}
"""
