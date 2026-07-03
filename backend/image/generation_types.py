from __future__ import annotations

from dataclasses import dataclass


ALLOWED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}
ALLOWED_IMAGE_QUALITIES = {"low", "medium", "high", "auto"}
ALLOWED_IMAGE_FORMATS = {"png", "jpeg", "webp"}
ALLOWED_IMAGE_BACKGROUNDS = {"transparent", "opaque", "auto"}
MAX_IMAGE_COUNT = 4


MIME_BY_FORMAT = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    size: str
    quality: str
    format: str
    background: str
    n: int
    style_notes: str = ""
    user_visible_prompt: str = ""


@dataclass(frozen=True)
class ImageArtifact:
    id: str
    url: str
    path: str
    mime_type: str
    size: str
    quality: str
    format: str
    prompt: str
    created_at: str


class ImageGenerationError(RuntimeError):
    def __init__(self, code: str, message: str, details: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_payload(self) -> dict:
        payload = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload
