from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from backend.image.generation_types import (
    ALLOWED_IMAGE_BACKGROUNDS,
    ALLOWED_IMAGE_FORMATS,
    ALLOWED_IMAGE_QUALITIES,
    ALLOWED_IMAGE_SIZES,
    MAX_IMAGE_COUNT,
    ImageGenerationError,
    ImageGenerationRequest,
)
from backend.image.storage import save_image_bytes
from backend.image.openai_provider import OpenAIImageProvider
from backend.settings import (
    get_image_default_background,
    get_image_default_format,
    get_image_default_quality,
    get_image_default_size,
    get_image_model,
    get_image_provider,
)


logger = logging.getLogger(__name__)


def normalize_prompt(prompt: str, style_notes: str = "") -> str:
    cleaned = " ".join((prompt or "").split())
    notes = " ".join((style_notes or "").split())
    if notes:
        return f"{cleaned}\n\nStyle and composition notes: {notes}"
    return cleaned


def validate_image_generation_request(
    prompt: str,
    size: str = "",
    quality: str = "",
    format: str = "",
    background: str = "",
    n: int = 1,
    style_notes: str = "",
    user_visible_prompt: str = "",
) -> ImageGenerationRequest:
    final_prompt = normalize_prompt(prompt, style_notes)
    if not final_prompt:
        raise ImageGenerationError("IMAGE_PROMPT_EMPTY", "Image prompt cannot be empty.")
    final_size = (size or get_image_default_size()).strip()
    final_quality = (quality or get_image_default_quality()).strip().lower()
    final_format = (format or get_image_default_format()).strip().lower()
    final_background = (background or get_image_default_background()).strip().lower()
    if final_size not in ALLOWED_IMAGE_SIZES:
        raise ImageGenerationError("IMAGE_INVALID_SIZE", f"Unsupported image size: {final_size}.")
    if final_quality not in ALLOWED_IMAGE_QUALITIES:
        raise ImageGenerationError("IMAGE_INVALID_QUALITY", f"Unsupported image quality: {final_quality}.")
    if final_format not in ALLOWED_IMAGE_FORMATS:
        raise ImageGenerationError("IMAGE_INVALID_FORMAT", f"Unsupported image format: {final_format}.")
    if final_background not in ALLOWED_IMAGE_BACKGROUNDS:
        raise ImageGenerationError("IMAGE_INVALID_BACKGROUND", f"Unsupported image background: {final_background}.")
    try:
        count = int(n or 1)
    except Exception as error:
        raise ImageGenerationError("IMAGE_INVALID_COUNT", "Image count must be an integer.") from error
    if count < 1:
        raise ImageGenerationError("IMAGE_INVALID_COUNT", "Image count must be at least 1.")
    if count > MAX_IMAGE_COUNT:
        raise ImageGenerationError("IMAGE_INVALID_COUNT", f"Image count cannot exceed {MAX_IMAGE_COUNT}.")
    return ImageGenerationRequest(
        prompt=final_prompt,
        size=final_size,
        quality=final_quality,
        format=final_format,
        background=final_background,
        n=count,
        style_notes=style_notes,
        user_visible_prompt=user_visible_prompt,
    )


class ImageGenerationService:
    def __init__(self, provider: Any | None = None, provider_name: str | None = None, model: str | None = None) -> None:
        self.provider_name = provider_name or get_image_provider()
        self.model = model or get_image_model()
        self.provider = provider or self._build_provider()

    def _build_provider(self) -> Any:
        if self.provider_name == "openai":
            return OpenAIImageProvider(model=self.model)
        raise ImageGenerationError("IMAGE_PROVIDER_UNSUPPORTED", f"Unsupported image provider: {self.provider_name}.")

    def generate_image(
        self,
        prompt: str,
        size: str = "",
        quality: str = "",
        format: str = "",
        background: str = "",
        n: int = 1,
        style_notes: str = "",
        user_visible_prompt: str = "",
    ) -> dict[str, Any]:
        request = validate_image_generation_request(
            prompt=prompt,
            size=size,
            quality=quality,
            format=format,
            background=background,
            n=n,
            style_notes=style_notes,
            user_visible_prompt=user_visible_prompt,
        )
        logger.info(
            "image_generation requested provider=%s model=%s size=%s quality=%s format=%s background=%s n=%s",
            self.provider_name,
            self.model,
            request.size,
            request.quality,
            request.format,
            request.background,
            request.n,
        )
        raw_images = self.provider.generate(request)
        artifacts = [
            save_image_bytes(data, request.format, request.size, request.quality, request.prompt)
            for data in raw_images
        ]
        return {
            "ok": True,
            "type": "image_result",
            "images": [asdict(item) for item in artifacts],
            "provider": self.provider_name,
            "model": self.model,
        }

    def edit_image(self, *_: Any, **__: Any) -> dict[str, Any]:
        raise NotImplementedError("Image editing is reserved for a future provider implementation. Current version supports text-to-image only.")


def generate_image_with_openai(**kwargs: Any) -> dict[str, Any]:
    return ImageGenerationService(provider_name="openai").generate_image(**kwargs)
