from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request

from backend.image.generation_types import ImageGenerationError, ImageGenerationRequest
from backend.settings import get_image_model, get_openai_api_key, get_openai_base_url


class OpenAIImageProvider:
    provider = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else get_openai_api_key()
        self.model = model or get_image_model()
        self.base_url = (base_url or get_openai_base_url()).rstrip("/")

    def _post_json(self, path: str, payload: dict, timeout: int = 180) -> dict:
        if not self.api_key:
            raise ImageGenerationError("OPENAI_API_KEY_MISSING", "OPENAI_API_KEY is not configured.")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            code = "OPENAI_IMAGE_RATE_LIMIT" if error.code == 429 else "OPENAI_IMAGE_API_FAILED"
            raise ImageGenerationError(code, f"OpenAI image API failed with HTTP {error.code}.", detail) from error
        except TimeoutError as error:
            raise ImageGenerationError("OPENAI_IMAGE_TIMEOUT", "OpenAI image generation timed out.") from error
        except urllib.error.URLError as error:
            raise ImageGenerationError("OPENAI_IMAGE_NETWORK_ERROR", "OpenAI image API is not reachable.", str(error)) from error

    def generate(self, request: ImageGenerationRequest) -> list[bytes]:
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "size": request.size,
            "quality": request.quality,
            "n": request.n,
            "background": request.background,
            "output_format": request.format,
        }
        data = self._post_json("/images/generations", payload)
        rows = data.get("data") or []
        if not rows:
            raise ImageGenerationError("IMAGE_EMPTY_RESPONSE", "Image provider returned no images.")
        images: list[bytes] = []
        for item in rows:
            b64 = item.get("b64_json")
            if not b64:
                raise ImageGenerationError("IMAGE_EMPTY_RESPONSE", "Image provider response did not include b64_json.")
            try:
                images.append(base64.b64decode(b64))
            except Exception as error:
                raise ImageGenerationError("IMAGE_BASE64_DECODE_FAILED", "Could not decode provider image payload.", str(error)) from error
        return images
