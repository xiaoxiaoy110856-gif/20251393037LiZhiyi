from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from backend.image.comfyui_workflow import ComfyUIWorkflowRunner
from backend.settings import (
    get_image_generation_default_batch_size,
    get_image_generation_max_retries,
    get_image_quality_min_score,
)

# This module is the high-quality local image pipeline. It intentionally keeps
# prompt planning, presets, rule scoring, and retry orchestration together
# because they evolve as one feature and share the same ImageGenerationPlan.


@dataclass
class HighresPlan:
    upscale_factor: float = 1.5
    denoise: float = 0.32
    steps: int = 12


@dataclass
class QualityGate:
    min_score: float = 0.75
    max_retries: int = 2
    checks: list[str] = field(default_factory=list)


@dataclass
class ImageGenerationPlan:
    """Structured contract between the Agent and ComfyUI.

    The raw user prompt is preserved, but ComfyUI receives positive_prompt,
    negative_prompt, and generation parameters from this plan.
    """

    original_user_prompt: str
    positive_prompt: str
    negative_prompt: str
    preset: str = "general_photorealistic"
    style: str = "photorealistic"
    subject: str = ""
    scene: str = ""
    camera: str = ""
    lighting: str = ""
    composition: str = ""
    physical_constraints: list[str] = field(default_factory=list)
    quality_constraints: list[str] = field(default_factory=list)
    forbidden_elements: list[str] = field(default_factory=list)
    size: str = "768x512"
    batch_size: int = 1
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "euler"
    scheduler: str = "normal"
    seed: int | str = "random"
    model: str = ""
    loras: list[dict[str, Any]] = field(default_factory=list)
    use_highres_fix: bool = False
    highres: HighresPlan = field(default_factory=HighresPlan)
    quality_gate: QualityGate = field(default_factory=QualityGate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_user_prompt": self.original_user_prompt,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "preset": self.preset,
            "style": self.style,
            "subject": self.subject,
            "scene": self.scene,
            "camera": self.camera,
            "lighting": self.lighting,
            "composition": self.composition,
            "physical_constraints": self.physical_constraints,
            "quality_constraints": self.quality_constraints,
            "forbidden_elements": self.forbidden_elements,
            "size": self.size,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "cfg": self.cfg,
            "sampler": self.sampler,
            "scheduler": self.scheduler,
            "seed": self.seed,
            "model": self.model,
            "loras": self.loras,
            "use_highres_fix": self.use_highres_fix,
            "highres": {
                "upscale_factor": self.highres.upscale_factor,
                "denoise": self.highres.denoise,
                "steps": self.highres.steps,
            },
            "quality_gate": {
                "min_score": self.quality_gate.min_score,
                "max_retries": self.quality_gate.max_retries,
                "checks": self.quality_gate.checks,
            },
        }


@dataclass
class ImageQualityCheck:
    name: str
    score: float
    passed: bool
    reason: str


@dataclass
class ImageQualityReport:
    image_id: str
    score: float
    passed: bool
    checks: list[ImageQualityCheck] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggested_prompt_fixes: list[str] = field(default_factory=list)
    suggested_parameter_fixes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "score": self.score,
            "passed": self.passed,
            "checks": [check.__dict__ for check in self.checks],
            "issues": self.issues,
            "suggested_prompt_fixes": self.suggested_prompt_fixes,
            "suggested_parameter_fixes": self.suggested_parameter_fixes,
        }


@dataclass
class TrainingPlan:
    training_type: str = "lora"
    dataset_id: str = ""
    base_model: str = ""
    trigger_words: list[str] = field(default_factory=list)
    caption_strategy: str = "mixed"
    resolution: int = 1024
    epochs: int = 10
    rank: int = 16
    learning_rate: float = 0.0001


COMMON_NEGATIVE = (
    "low quality, blurry, distorted, deformed, bad anatomy, extra objects, duplicate objects, "
    "text, watermark, logo, signature, oversaturated, unrealistic lighting, bad perspective, "
    "cropped subject, floating parts"
)


IMAGE_PRESETS: dict[str, dict] = {
    "automotive_photorealistic": {
        "default_model": "",
        "default_size": "768x512",
        "default_batch_size": 2,
        "default_steps": 24,
        "default_cfg": 6.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "default_negative_prompt": (
            "multiple cars, extra vehicles, duplicate car, background car, open doors, broken body, "
            "deformed body, malformed wheels, extra wheels, missing wheels, distorted perspective, "
            "bad reflections, mirrored vehicle, floating parts, disassembled car, unrealistic car structure, "
            "extra headlights, extra mirrors, people, text, watermark, logo"
        ),
        "quality_checks": ["single_vehicle", "no_extra_vehicles", "vehicle_structure_integrity", "no_text", "no_watermark"],
        "highres_fix_default": False,
        "retry_strategy": "single_subject_structure",
    },
    "product_photography": {
        "default_model": "",
        "default_size": "768x768",
        "default_batch_size": 2,
        "default_steps": 22,
        "default_cfg": 6.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "default_negative_prompt": "warped product, duplicate product, wrong label, fake text, messy background, broken geometry, inconsistent material, bad reflection",
        "quality_checks": ["single_subject", "no_text", "structure_integrity", "aesthetic_quality"],
        "highres_fix_default": False,
        "retry_strategy": "clean_product",
    },
    "portrait_photorealistic": {
        "default_model": "",
        "default_size": "512x768",
        "default_batch_size": 2,
        "default_steps": 24,
        "default_cfg": 6.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "default_negative_prompt": "extra fingers, missing fingers, malformed hands, deformed face, asymmetrical eyes, bad teeth, extra limbs, broken anatomy, distorted body",
        "quality_checks": ["anatomy", "no_text", "aesthetic_quality"],
        "highres_fix_default": False,
        "retry_strategy": "portrait_anatomy",
    },
    "cinematic_scene": {
        "default_model": "",
        "default_size": "768x512",
        "default_batch_size": 2,
        "default_steps": 28,
        "default_cfg": 6.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "default_negative_prompt": "messy composition, blurry, low quality, lowres, noisy, jpeg artifacts, text, watermark, bad perspective",
        "quality_checks": ["prompt_alignment", "aesthetic_quality", "no_text"],
        "highres_fix_default": False,
        "retry_strategy": "general",
    },
    "poster_design": {
        "default_model": "",
        "default_size": "768x1024",
        "default_batch_size": 2,
        "default_steps": 28,
        "default_cfg": 6.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "default_negative_prompt": "garbled text, watermark, signature, low quality, cluttered layout",
        "quality_checks": ["composition", "aesthetic_quality"],
        "highres_fix_default": False,
        "retry_strategy": "poster",
    },
    "logo_draft": {
        "default_model": "",
        "default_size": "768x768",
        "default_batch_size": 2,
        "default_steps": 24,
        "default_cfg": 6.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "default_negative_prompt": "photorealistic, cluttered, complex background, watermark, signature, fake text",
        "quality_checks": ["simple_shape", "clean_background"],
        "highres_fix_default": False,
        "retry_strategy": "logo",
    },
    "interior_design": {
        "default_model": "",
        "default_size": "768x512",
        "default_batch_size": 2,
        "default_steps": 24,
        "default_cfg": 6.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "default_negative_prompt": "warped furniture, impossible geometry, clutter, low quality, text, watermark, distorted room perspective",
        "quality_checks": ["structure_integrity", "aesthetic_quality", "no_text"],
        "highres_fix_default": False,
        "retry_strategy": "interior",
    },
    "general_photorealistic": {
        "default_model": "",
        "default_size": "768x512",
        "default_batch_size": get_image_generation_default_batch_size(),
        "default_steps": 28,
        "default_cfg": 6.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "default_negative_prompt": "lowres, noisy, jpeg artifacts, bad composition, muddy details, flat lighting",
        "quality_checks": ["prompt_alignment", "aesthetic_quality", "no_text", "no_watermark"],
        "highres_fix_default": False,
        "retry_strategy": "general",
    },
}


def get_preset(name: str | None) -> tuple[str, dict]:
    key = (name or "").strip() or "general_photorealistic"
    if key not in IMAGE_PRESETS:
        key = "general_photorealistic"
    preset = deepcopy(IMAGE_PRESETS[key])
    preset["quality_min_score"] = get_image_quality_min_score()
    return key, preset


def list_presets() -> list[str]:
    return sorted(IMAGE_PRESETS)


class ImageNegativePromptBuilder:
    """Builds compact negative prompts from common terms plus preset terms."""

    def build(self, preset_name: str, forbidden_elements: list[str] | None = None, allow_text: bool = False) -> str:
        _, preset = get_preset(preset_name)
        parts = [COMMON_NEGATIVE, str(preset.get("default_negative_prompt", ""))]
        if forbidden_elements:
            parts.extend(forbidden_elements)
        if allow_text:
            parts = [part.replace("text", "").replace("garbled text", "") for part in parts]
        seen: set[str] = set()
        terms: list[str] = []
        for part in parts:
            for raw in str(part).split(","):
                term = " ".join(raw.strip().split())
                if not term:
                    continue
                key = term.lower()
                if key not in seen:
                    seen.add(key)
                    terms.append(term)
        return ", ".join(terms[:80])


AUTOMOTIVE_RE = re.compile(r"\b(car|supercar|sports car|racing|vehicle|lamborghini|ferrari|porsche|跑车|汽车|赛车)\b", re.I)
PRODUCT_RE = re.compile(r"\b(product|mockup|bottle|shoe|watch|phone|产品|商品)\b", re.I)
PORTRAIT_RE = re.compile(r"\b(portrait|person|woman|man|face|人物|人像|女孩|男孩)\b", re.I)
POSTER_RE = re.compile(r"\b(poster|海报|宣传图|cover)\b", re.I)
LOGO_RE = re.compile(r"\b(logo|icon|标志|图标)\b", re.I)
INTERIOR_RE = re.compile(r"\b(bedroom|room|interior|furniture|床|卧室|室内|客厅)\b", re.I)
SHOE_RE = re.compile(r"\b(shoe|sneaker|footwear|trainer|running shoe)\b", re.I)


class ImagePromptRewriter:
    """图片 prompt 改写器：根据用户输入、风格、预设和尺寸生成更稳定的正负提示词。"""
    """Turns a loose user request into a preset-backed ImageGenerationPlan."""

    def __init__(self, negative_builder: ImageNegativePromptBuilder | None = None) -> None:
        self.negative_builder = negative_builder or ImageNegativePromptBuilder()

    def infer_preset(self, prompt: str, explicit: str = "") -> str:
        """根据显式 preset 或 prompt 关键词判断图片类型，例如海报、图标、产品图。"""
        if explicit:
            return explicit
        text = prompt or ""
        if AUTOMOTIVE_RE.search(text):
            return "automotive_photorealistic"
        if LOGO_RE.search(text):
            return "logo_draft"
        if POSTER_RE.search(text):
            return "poster_design"
        if PRODUCT_RE.search(text):
            return "product_photography"
        if PORTRAIT_RE.search(text):
            return "portrait_photorealistic"
        if INTERIOR_RE.search(text):
            return "interior_design"
        return "general_photorealistic"

    def product_detail_hints(self, prompt: str) -> tuple[str, list[str]]:
        if SHOE_RE.search(prompt):
            return (
                "sneaker footwear silhouette, visible laces, outsole, heel counter, toe box, side profile",
                ["smartphone", "phone screen", "rectangular device"],
            )
        return ("", [])

    def rewrite(
        self,
        prompt: str,
        preset: str = "",
        style: str = "",
        size: str = "",
        batch_size: int | None = None,
        quality_mode: str = "balanced",
        use_highres_fix: bool | None = None,
    ) -> ImageGenerationPlan:
        """生成完整图片计划，包括正向 prompt、负向 prompt、尺寸、batch、采样参数和质量门槛。"""
        # First version is deterministic/rule-based. This keeps generation fast
        # and testable; a future LLM/VLM rewriter can replace this method behind
        # the same ImageGenerationPlan contract.
        raw_prompt = " ".join((prompt or "").strip().split())
        if not raw_prompt:
            raise ValueError("Image prompt cannot be empty.")
        preset_name, preset_config = get_preset(self.infer_preset(raw_prompt, preset))
        final_style = style or ("product" if preset_name == "product_photography" else "photorealistic")
        forbidden = ["no text", "no watermark"]
        allow_text = preset_name in {"poster_design", "logo_draft"} and bool(re.search(r"\b(text|文字|标题|slogan)\b", raw_prompt, re.I))
        if preset_name == "automotive_photorealistic":
            positive = (
                f"A single {raw_prompt}, one car only, parked alone or shown as the only main vehicle. "
                "Low-angle front three-quarter view, closed doors, intact body panels, realistic proportions, "
                "detailed headlights, realistic wheels, clean aerodynamic lines. Cinematic lighting, glossy reflections, "
                "high-end automotive photography, sharp focus, photorealistic, highly detailed, no people, no text, "
                "no watermark, no other vehicles."
            )
            physical = ["one car only", "closed doors", "intact body panels", "realistic wheels", "no other vehicles"]
            subject = "single vehicle"
            composition = "low-angle front three-quarter view"
        elif preset_name == "product_photography":
            product_hints, extra_forbidden = self.product_detail_hints(raw_prompt)
            positive = (
                f"Professional studio product photography of {raw_prompt}. Entire product fully visible in frame, "
                f"{product_hints + ', ' if product_hints else ''}"
                "centered composition with generous margins, clean seamless background, realistic proportions, "
                "accurate material texture, softbox lighting, subtle shadow on the table, sharp focus, high detail, "
                "commercial catalog quality, no text, no watermark."
            )
            physical = ["single product only", "entire product visible", "centered with margins", "realistic proportions"]
            forbidden.extend(extra_forbidden)
            subject = raw_prompt[:120]
            composition = "centered product shot with generous margins"
        else:
            positive = (
                f"{raw_prompt}. Clear primary subject, entire main subject visible unless a close-up is requested, coherent composition, realistic proportions, "
                f"{final_style} style, sharp focus, balanced lighting, detailed surfaces, high quality, "
                "clean background where appropriate, no watermark."
            )
            physical = ["clear primary subject", "coherent structure", "realistic proportions"]
            subject = raw_prompt[:120]
            composition = "clear centered composition"
        negative = self.negative_builder.build(preset_name, forbidden_elements=forbidden, allow_text=allow_text)
        mode = (quality_mode or "balanced").lower()
        default_batch = int(preset_config.get("default_batch_size", 1))
        if mode == "fast":
            default_batch = 1
            steps = min(18, int(preset_config.get("default_steps", 20)))
            retries = 0
        elif mode == "high":
            default_batch = 1
            steps = max(32, int(preset_config.get("default_steps", 24)))
            retries = get_image_generation_max_retries()
        else:
            default_batch = min(2, max(1, default_batch))
            steps = max(26, int(preset_config.get("default_steps", 22)))
            retries = min(1, get_image_generation_max_retries())
        final_batch = max(1, min(4, int(batch_size or default_batch)))
        final_use_highres = bool(
            (mode == "high" or preset_config.get("highres_fix_default", False))
            if use_highres_fix is None
            else use_highres_fix
        )
        return ImageGenerationPlan(
            original_user_prompt=raw_prompt,
            positive_prompt=positive,
            negative_prompt=negative,
            preset=preset_name,
            style=final_style,
            subject=subject,
            scene="inferred from user prompt",
            camera="35mm lens" if preset_name == "automotive_photorealistic" else "",
            lighting="cinematic lighting" if preset_name == "automotive_photorealistic" else "balanced lighting",
            composition=composition,
            physical_constraints=physical,
            quality_constraints=["sharp focus", "high quality", "realistic proportions"],
            forbidden_elements=forbidden,
            size=size or str(preset_config.get("default_size", "768x512")),
            batch_size=final_batch,
            steps=steps,
            cfg=float(preset_config.get("default_cfg", 7.0)),
            sampler=str(preset_config.get("sampler", "euler")),
            scheduler=str(preset_config.get("scheduler", "normal")),
            model=str(preset_config.get("default_model", "")),
            use_highres_fix=final_use_highres,
            highres=HighresPlan(),
            quality_gate=QualityGate(
                min_score=float(preset_config.get("quality_min_score", 0.75)),
                max_retries=retries,
                checks=list(preset_config.get("quality_checks", [])),
            ),
        )


class ImageCritic:
    """图片质量检查器：对候选图做轻量评分，判断是否需要重试或修复 prompt。"""
    """Lightweight image scorer.

    It does not do visual object detection yet; it scores artifacts and plan
    constraints so the controller can rank candidates and retry consistently.
    """

    def evaluate_image(self, image_artifact: dict, plan: ImageGenerationPlan) -> ImageQualityReport:
        """评估单张候选图，输出分数、是否通过、问题列表和修复建议。"""
        checks: list[ImageQualityCheck] = []
        issues: list[str] = []
        score = 0.6
        path = Path(str(image_artifact.get("path", "")))
        if path.exists() and path.stat().st_size > 1024:
            checks.append(ImageQualityCheck("file_saved", 1.0, True, "Image artifact exists and is non-empty."))
            score += 0.15
        else:
            checks.append(ImageQualityCheck("file_saved", 0.0, False, "Image file is missing or too small."))
            issues.append("file_missing_or_too_small")
        width = int(image_artifact.get("width") or 0)
        height = int(image_artifact.get("height") or 0)
        if width >= 512 and height >= 512:
            checks.append(ImageQualityCheck("image_size", 0.9, True, "Image has a usable generation size."))
            score += 0.1
        else:
            checks.append(ImageQualityCheck("image_size", 0.4, False, "Image is smaller than recommended."))
            issues.append("low_resolution")
        if plan.preset == "automotive_photorealistic" and "one car only" in plan.positive_prompt.lower():
            checks.append(ImageQualityCheck("single_vehicle_prompt_constraint", 0.85, True, "Plan contains single-vehicle constraints."))
            score += 0.1
        if "text" in plan.negative_prompt.lower() and "watermark" in plan.negative_prompt.lower():
            checks.append(ImageQualityCheck("no_text_watermark_prompt_constraint", 0.85, True, "Negative prompt bans text and watermark."))
            score += 0.05
        score = min(1.0, score)
        passed = score >= plan.quality_gate.min_score
        suggested_prompt_fixes = []
        suggested_parameter_fixes = {}
        if not passed:
            suggested_prompt_fixes.append("single subject only, centered composition, no background duplicates")
            suggested_parameter_fixes["cfg"] = min(float(plan.cfg), 6.0)
            suggested_parameter_fixes["steps"] = max(int(plan.steps), 28)
        return ImageQualityReport(
            image_id=str(image_artifact.get("id", "")),
            score=round(score, 4),
            passed=passed,
            checks=checks,
            issues=issues,
            suggested_prompt_fixes=suggested_prompt_fixes,
            suggested_parameter_fixes=suggested_parameter_fixes,
        )


class ImageGenerationQualityController:
    """高级图片生成控制器：串联 prompt 改写、批量生成、质量评分和失败重试。"""

    def __init__(
        self,
        rewriter: ImagePromptRewriter | None = None,
        runner: ComfyUIWorkflowRunner | None = None,
        critic: ImageCritic | None = None,
    ) -> None:
        self.rewriter = rewriter or ImagePromptRewriter()
        self.runner = runner or ComfyUIWorkflowRunner()
        self.critic = critic or ImageCritic()

    def generate(
        self,
        prompt: str,
        style: str = "",
        preset: str = "",
        size: str = "",
        batch_size: int | None = None,
        quality_mode: str = "balanced",
        allow_retry: bool = True,
        use_highres_fix: bool | None = None,
        notes: str = "",
        **_: Any,
    ) -> dict[str, Any]:
        # 核心3：generate_image_advanced 会进入这里。它是本地 ComfyUI 图片生成的质量门，
        # 负责改写 prompt、生成候选图、评分、必要时修复并返回最佳图片。
        plan = self.rewriter.rewrite(
            f"{prompt} {notes}".strip(),
            preset=preset,
            style=style,
            size=size,
            batch_size=batch_size,
            quality_mode=quality_mode,
            use_highres_fix=use_highres_fix,
        )
        attempts: list[dict[str, Any]] = []
        best_artifact: dict[str, Any] | None = None
        best_report = None
        max_retries = plan.quality_gate.max_retries if allow_retry else 0
        current_plan = plan
        for attempt in range(max_retries + 1):
            # Each retry runs a full candidate batch with a repaired plan. A
            # single bad image should not fail the whole request.
            artifacts = self.runner.generate(current_plan)
            reports = [self.critic.evaluate_image(artifact, current_plan) for artifact in artifacts]
            ranked = sorted(zip(artifacts, reports), key=lambda item: item[1].score, reverse=True)
            if ranked:
                best_artifact, best_report = ranked[0]
            attempts.append(
                {
                    "attempt": attempt,
                    "plan": current_plan.to_dict(),
                    "candidates": [{"id": artifact.get("id", ""), "url": artifact.get("url", "")} for artifact in artifacts],
                    "reports": [report.to_dict() for report in reports],
                    "candidate_count": len(artifacts),
                }
            )
            if best_report and best_report.passed:
                break
            if attempt < max_retries and best_report:
                current_plan = self._repair_plan(current_plan, best_report.to_dict())
        if not best_artifact or not best_report:
            raise RuntimeError("Image generation produced no candidates.")
        final_image = {
            **{key: value for key, value in best_artifact.items() if key != "path"},
            "score": best_report.score,
        }
        return {
            "ok": True,
            "type": "image_result",
            "provider": "comfyui",
            "model": current_plan.model or "comfyui-checkpoint",
            "final_image": final_image,
            "images": [final_image],
            "candidates": [
                {
                    "id": attempt_report["image_id"],
                    "url": next((item.get("url", "") for item in attempt["candidates"] if item.get("id") == attempt_report["image_id"]), ""),
                    "score": attempt_report["score"],
                    "passed": attempt_report["passed"],
                }
                for attempt in attempts
                for attempt_report in attempt["reports"]
            ],
            "generation_plan": {
                "positive_prompt": plan.positive_prompt,
                "negative_prompt": plan.negative_prompt,
                "preset": plan.preset,
                "parameters": {
                    "size": plan.size,
                    "batch_size": plan.batch_size,
                    "steps": plan.steps,
                    "cfg": plan.cfg,
                    "sampler": plan.sampler,
                    "scheduler": plan.scheduler,
                    "use_highres_fix": plan.use_highres_fix,
                },
            },
            "quality_report": best_report.to_dict(),
            "retries": attempts[1:],
            "detail": f"Generated {len(attempts)} attempt(s); selected score {best_report.score:.2f}.",
        }

    def _repair_plan(self, plan: ImageGenerationPlan, report: dict[str, Any]) -> ImageGenerationPlan:
        """根据质量报告追加负面约束并调整参数，为下一轮重试生成修复后的计划。"""
        prompt_fixes = report.get("suggested_prompt_fixes") or ["single subject only, no duplicates"]
        param_fixes = report.get("suggested_parameter_fixes") or {}
        positive = f"{plan.positive_prompt}, {', '.join(prompt_fixes)}"
        negative = f"{plan.negative_prompt}, duplicate object, extra object, malformed structure, broken geometry"
        return replace(
            plan,
            positive_prompt=positive,
            negative_prompt=negative,
            cfg=float(param_fixes.get("cfg", min(plan.cfg, 6.0))),
            steps=int(param_fixes.get("steps", max(plan.steps, 28))),
        )


# 核心3：工具注册层调用的高级图片生成入口。
def generate_image_advanced(**kwargs: Any) -> dict[str, Any]:
    return ImageGenerationQualityController().generate(**kwargs)
