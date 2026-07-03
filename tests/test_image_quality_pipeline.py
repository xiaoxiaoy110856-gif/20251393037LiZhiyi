import os
import tempfile
import unittest
from pathlib import Path

from backend.comfyui_workflow import ComfyUIWorkflowPatcher
from backend.image_quality import (
    ImageCritic,
    ImageGenerationQualityController,
    ImageNegativePromptBuilder,
    ImagePromptRewriter,
    ImageQualityCheck,
    ImageQualityReport,
    get_preset,
    list_presets,
)
from backend.tool_registry import TOOL_REGISTRY
from backend.agent_loop import should_generate_image


class FakeRunner:
    def __init__(self, temp: tempfile.TemporaryDirectory, small_first: bool = False) -> None:
        self.temp = temp
        self.calls = 0
        self.small_first = small_first

    def generate(self, plan):
        self.calls += 1
        artifacts = []
        for index in range(plan.batch_size):
            path = Path(self.temp.name) / f"candidate_{self.calls}_{index}.png"
            data = b"x" * (200 if self.small_first and self.calls == 1 else 2048)
            path.write_bytes(data)
            artifacts.append(
                {
                    "id": f"img-{self.calls}-{index}",
                    "url": f"/generated-images/candidate_{self.calls}_{index}.png",
                    "path": str(path),
                    "width": 768,
                    "height": 512,
                }
            )
        return artifacts


class RetryCritic:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_image(self, image_artifact, plan):
        self.calls += 1
        passed = self.calls > 1
        score = 0.4 if not passed else 0.88
        return ImageQualityReport(
            image_id=image_artifact["id"],
            score=score,
            passed=passed,
            checks=[ImageQualityCheck("mock", score, passed, "mock")],
            issues=[] if passed else ["duplicate_object"],
            suggested_prompt_fixes=["single subject only"],
            suggested_parameter_fixes={"cfg": 6.0, "steps": 30},
        )


class ImageQualityPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.old_output = os.environ.get("IMAGE_OUTPUT_DIR")
        os.environ["IMAGE_OUTPUT_DIR"] = self.temp.name

    def tearDown(self) -> None:
        if self.old_output is None:
            os.environ.pop("IMAGE_OUTPUT_DIR", None)
        else:
            os.environ["IMAGE_OUTPUT_DIR"] = self.old_output
        self.temp.cleanup()

    def test_rewriter_builds_generation_plan(self):
        plan = ImagePromptRewriter().rewrite("a red sports car on a wet street")
        self.assertEqual(plan.preset, "automotive_photorealistic")
        self.assertIn("one car only", plan.positive_prompt.lower())
        self.assertIn("no other vehicles", plan.positive_prompt.lower())
        self.assertIn("closed doors", plan.positive_prompt.lower())

    def test_negative_prompt_builder_adds_automotive_terms(self):
        negative = ImageNegativePromptBuilder().build("automotive_photorealistic")
        self.assertIn("multiple cars", negative)
        self.assertIn("malformed wheels", negative)

    def test_presets_load_and_fallback(self):
        self.assertIn("automotive_photorealistic", list_presets())
        key, preset = get_preset("missing")
        self.assertEqual(key, "general_photorealistic")
        self.assertIn("default_size", preset)

    def test_workflow_patcher_updates_nodes(self):
        plan = ImagePromptRewriter().rewrite("a red sports car")
        workflow = ComfyUIWorkflowPatcher().build_basic_workflow("model.safetensors", plan, seed=1)
        patched = ComfyUIWorkflowPatcher().patch(workflow, plan, "other.safetensors", seed=42)
        self.assertEqual(patched["6"]["inputs"]["text"], plan.positive_prompt)
        self.assertEqual(patched["7"]["inputs"]["text"], plan.negative_prompt)
        self.assertEqual(patched["3"]["inputs"]["seed"], 42)
        self.assertEqual(patched["3"]["inputs"]["steps"], plan.steps)
        self.assertEqual(patched["3"]["inputs"]["cfg"], plan.cfg)
        self.assertEqual(patched["5"]["inputs"]["batch_size"], plan.batch_size)

    def test_critic_returns_score_and_issues(self):
        path = Path(self.temp.name) / "image.png"
        path.write_bytes(b"x" * 2048)
        plan = ImagePromptRewriter().rewrite("a red sports car")
        report = ImageCritic().evaluate_image({"id": "x", "path": str(path), "width": 768, "height": 512}, plan)
        self.assertGreater(report.score, 0)
        self.assertIsInstance(report.issues, list)

    def test_quality_controller_selects_best_and_returns_public_url(self):
        controller = ImageGenerationQualityController(runner=FakeRunner(self.temp), critic=ImageCritic())
        result = controller.generate("a red sports car", quality_mode="fast", batch_size=2, allow_retry=False)
        self.assertEqual(result["type"], "image_result")
        self.assertIn("/generated-images/", result["final_image"]["url"])
        self.assertIn("score", result["final_image"])

    def test_quality_controller_retries_low_score(self):
        runner = FakeRunner(self.temp)
        controller = ImageGenerationQualityController(runner=runner, critic=RetryCritic())
        result = controller.generate("a red sports car", batch_size=1, allow_retry=True)
        self.assertGreaterEqual(runner.calls, 2)
        self.assertTrue(result["quality_report"]["passed"])
        self.assertIn("single subject only", result["retries"][0]["plan"]["positive_prompt"])

    def test_generate_image_advanced_registered(self):
        self.assertIn("generate_image_advanced", TOOL_REGISTRY)

    def test_image_intent_still_detected(self):
        self.assertTrue(should_generate_image("帮我生成一张跑车图片"))
        self.assertFalse(should_generate_image("怎么写图片生成 prompt 更好"))


if __name__ == "__main__":
    unittest.main()
