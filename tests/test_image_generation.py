from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path

from backend.agent_loop import should_generate_image
from backend.image_generation_service import ImageGenerationService, validate_image_generation_request
from backend.image_generation_types import ImageGenerationError
from backend.openai_image_provider import OpenAIImageProvider
from backend.tool_registry import TOOL_REGISTRY


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeProvider:
    provider = "openai"

    def generate(self, request):
        return [PNG_1X1 for _ in range(request.n)]


class FakeOpenAIProvider(OpenAIImageProvider):
    def _post_json(self, path: str, payload: dict, timeout: int = 180) -> dict:
        return {"data": [{"b64_json": base64.b64encode(PNG_1X1).decode("ascii")}]}


class ImageGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_output = os.environ.get("IMAGE_OUTPUT_DIR")
        self.old_key = os.environ.get("OPENAI_API_KEY")
        self.temp = tempfile.TemporaryDirectory()
        os.environ["IMAGE_OUTPUT_DIR"] = self.temp.name

    def tearDown(self) -> None:
        if self.old_output is None:
            os.environ.pop("IMAGE_OUTPUT_DIR", None)
        else:
            os.environ["IMAGE_OUTPUT_DIR"] = self.old_output
        if self.old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self.old_key
        self.temp.cleanup()

    def test_generate_image_validation_accepts_defaults(self) -> None:
        request = validate_image_generation_request(prompt="a cyberpunk cat poster")
        self.assertEqual(request.size, "1024x1024")
        self.assertEqual(request.quality, "auto")
        self.assertEqual(request.format, "png")

    def test_empty_prompt_rejected(self) -> None:
        with self.assertRaises(ImageGenerationError) as ctx:
            validate_image_generation_request(prompt=" ")
        self.assertEqual(ctx.exception.code, "IMAGE_PROMPT_EMPTY")

    def test_enum_validation(self) -> None:
        for kwargs in [
            {"size": "999x999"},
            {"quality": "best"},
            {"format": "gif"},
            {"background": "blue"},
        ]:
            with self.assertRaises(ImageGenerationError):
                validate_image_generation_request(prompt="cat", **kwargs)

    def test_n_over_max_rejected(self) -> None:
        with self.assertRaises(ImageGenerationError) as ctx:
            validate_image_generation_request(prompt="cat", n=5)
        self.assertEqual(ctx.exception.code, "IMAGE_INVALID_COUNT")

    def test_openai_provider_parses_mock_b64(self) -> None:
        provider = FakeOpenAIProvider(api_key="test-key", model="gpt-image-2")
        request = validate_image_generation_request(prompt="cat")
        images = provider.generate(request)
        self.assertEqual(images, [PNG_1X1])

    def test_service_saves_base64_image_file(self) -> None:
        service = ImageGenerationService(provider=FakeProvider(), provider_name="openai", model="gpt-image-2")
        result = service.generate_image(prompt="a cat", n=1)
        image = result["images"][0]
        self.assertTrue(Path(image["path"]).exists())
        self.assertEqual(Path(image["path"]).read_bytes(), PNG_1X1)
        self.assertIn("/generated-images/", image["url"])

    def test_filename_does_not_include_prompt(self) -> None:
        service = ImageGenerationService(provider=FakeProvider(), provider_name="openai", model="gpt-image-2")
        result = service.generate_image(prompt="../secret prompt name", n=1)
        self.assertNotIn("secret", Path(result["images"][0]["path"]).name)

    def test_artifact_contains_required_fields(self) -> None:
        service = ImageGenerationService(provider=FakeProvider(), provider_name="openai", model="gpt-image-2")
        image = service.generate_image(prompt="a cat")["images"][0]
        for key in ["id", "url", "mime_type", "created_at", "prompt"]:
            self.assertTrue(image[key])

    def test_missing_openai_api_key_is_clear(self) -> None:
        os.environ.pop("OPENAI_API_KEY", None)
        provider = OpenAIImageProvider(api_key="")
        request = validate_image_generation_request(prompt="cat")
        with self.assertRaises(ImageGenerationError) as ctx:
            provider.generate(request)
        self.assertEqual(ctx.exception.code, "OPENAI_API_KEY_MISSING")

    def test_tool_registry_has_generate_image(self) -> None:
        self.assertIn("generate_image", TOOL_REGISTRY)

    def test_image_intent_selects_generate_image(self) -> None:
        self.assertTrue(should_generate_image("帮我生成一张赛博朋克风格的猫咪海报"))

    def test_prompt_discussion_does_not_select_generate_image(self) -> None:
        self.assertFalse(should_generate_image("怎么写图片生成 prompt 更好"))


if __name__ == "__main__":
    unittest.main()
