from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelIdentity:
    requested_model: str
    canonical_model: str
    provider_kind: str


_ALIASES = {
    "opus": "claude-opus-4-1",
    "sonnet": "claude-3-7-sonnet",
    "haiku": "claude-3-5-haiku",
    "grok": "grok-3",
    "grok-3": "grok-3",
    "grok-mini": "grok-3-mini",
    "grok-3-mini": "grok-3-mini",
    "kimi": "kimi-k2.5",
    "qwen": "qwen-plus",
    "qwen-plus": "qwen-plus",
    "gpt4o": "gpt-4o",
}


def resolve_model_alias(model: str) -> str:
    trimmed = (model or "").strip()
    lower = trimmed.lower()
    return _ALIASES.get(lower, trimmed)


def detect_provider_kind(model: str, backend: str = "") -> str:
    canonical = resolve_model_alias(model)
    lower = canonical.lower()
    backend_lower = (backend or "").strip().lower()

    if backend_lower == "hf":
        return "huggingface-local"
    if backend_lower == "ollama":
        return "ollama-local"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("grok"):
        return "xai"
    if lower.startswith("gpt-") or lower.startswith("openai/"):
        return "openai"
    if lower.startswith("qwen") or lower.startswith("kimi"):
        return "openai-compatible"
    return "local"


def resolve_model_identity(model: str, backend: str = "") -> ModelIdentity:
    canonical = resolve_model_alias(model)
    return ModelIdentity(
        requested_model=(model or "").strip(),
        canonical_model=canonical,
        provider_kind=detect_provider_kind(canonical, backend=backend),
    )
