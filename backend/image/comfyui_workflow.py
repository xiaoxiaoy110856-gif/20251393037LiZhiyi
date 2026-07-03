from __future__ import annotations

import copy
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from backend.image.service import _api_checkpoints, _post_json, _url_json
from backend.settings import get_comfyui_checkpoint, get_comfyui_timeout_seconds, get_comfyui_url, get_generated_images_dir

if TYPE_CHECKING:
    from backend.image.quality import ImageGenerationPlan

# ComfyUI expects an API-format workflow JSON. The project currently builds a
# small SD workflow in code, while patch() keeps the door open for external
# exported workflows with configurable node ids.


SAMPLER_ALIASES = {
    "DPM++ 2M": "dpmpp_2m",
    "dpmpp_2m": "dpmpp_2m",
    "euler": "euler",
    "Euler": "euler",
}

SCHEDULER_ALIASES = {
    "Karras": "karras",
    "karras": "karras",
    "normal": "normal",
}


class ComfyUIWorkflowPatcher:
    """Writes an ImageGenerationPlan into a ComfyUI workflow."""

    def __init__(self, node_mapping: dict[str, str] | None = None) -> None:
        self.node_mapping = node_mapping or {
            "sampler": "3",
            "checkpoint": "4",
            "latent": "5",
            "positive_prompt": "6",
            "negative_prompt": "7",
            "save_image": "9",
        }

    def build_basic_workflow(self, checkpoint: str, plan: ImageGenerationPlan, seed: int) -> dict:
        width, height = parse_size(plan.size)
        if plan.use_highres_fix:
            return self._build_highres_workflow(checkpoint, plan, seed, width, height)
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": plan.steps,
                    "cfg": plan.cfg,
                    "sampler_name": SAMPLER_ALIASES.get(plan.sampler, plan.sampler or "euler"),
                    "scheduler": SCHEDULER_ALIASES.get(plan.scheduler, plan.scheduler or "normal"),
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": plan.batch_size}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": plan.positive_prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": plan.negative_prompt, "clip": ["4", 1]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "trl_quality", "images": ["8", 0]}},
        }

    def _build_highres_workflow(self, checkpoint: str, plan: ImageGenerationPlan, seed: int, width: int, height: int) -> dict:
        # Highres fix: build composition at a smaller latent size, upscale the
        # latent, then run a low-denoise second pass to recover detail.
        factor = max(1.0, float(plan.highres.upscale_factor or 1.0))
        base_width = round_to_multiple(max(256, int(width / factor)), 8)
        base_height = round_to_multiple(max(256, int(height / factor)), 8)
        sampler_name = SAMPLER_ALIASES.get(plan.sampler, plan.sampler or "euler")
        scheduler = SCHEDULER_ALIASES.get(plan.scheduler, plan.scheduler or "normal")
        refine_seed = (seed + 1) % 18_446_744_073_709_551_615
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": plan.steps,
                    "cfg": plan.cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": base_width, "height": base_height, "batch_size": plan.batch_size}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": plan.positive_prompt, "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": plan.negative_prompt, "clip": ["4", 1]}},
            "10": {
                "class_type": "LatentUpscale",
                "inputs": {
                    "samples": ["3", 0],
                    "upscale_method": "bicubic",
                    "width": width,
                    "height": height,
                    "crop": "disabled",
                },
            },
            "11": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": refine_seed,
                    "steps": plan.highres.steps,
                    "cfg": plan.cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": plan.highres.denoise,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["10", 0],
                },
            },
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "trl_quality_hires", "images": ["8", 0]}},
        }

    def patch(self, workflow: dict, plan: ImageGenerationPlan, checkpoint: str, seed: int) -> dict:
        patched = copy.deepcopy(workflow)
        positive_id = self.node_mapping.get("positive_prompt")
        negative_id = self.node_mapping.get("negative_prompt")
        sampler_id = self.node_mapping.get("sampler")
        checkpoint_id = self.node_mapping.get("checkpoint")
        latent_id = self.node_mapping.get("latent") or self.node_mapping.get("width_height")
        if positive_id in patched:
            patched[positive_id].setdefault("inputs", {})["text"] = plan.positive_prompt
        if negative_id in patched:
            patched[negative_id].setdefault("inputs", {})["text"] = plan.negative_prompt
        if checkpoint_id in patched:
            patched[checkpoint_id].setdefault("inputs", {})["ckpt_name"] = checkpoint
        if sampler_id in patched:
            inputs = patched[sampler_id].setdefault("inputs", {})
            inputs.update(
                {
                    "seed": seed,
                    "steps": plan.steps,
                    "cfg": plan.cfg,
                    "sampler_name": SAMPLER_ALIASES.get(plan.sampler, plan.sampler or "euler"),
                    "scheduler": SCHEDULER_ALIASES.get(plan.scheduler, plan.scheduler or "normal"),
                }
            )
        if latent_id in patched:
            width, height = parse_size(plan.size)
            inputs = patched[latent_id].setdefault("inputs", {})
            inputs.update({"width": width, "height": height, "batch_size": plan.batch_size})
        return patched


class ComfyUIWorkflowRunner:
    """Submits workflows to ComfyUI and copies generated files locally."""

    def __init__(self, patcher: ComfyUIWorkflowPatcher | None = None) -> None:
        self.patcher = patcher or ComfyUIWorkflowPatcher()

    def generate(self, plan: ImageGenerationPlan, workflow: dict | None = None) -> list[dict]:
        # The runner only returns copied artifacts served by this backend. It
        # never exposes ComfyUI's internal output path directly to the frontend.
        checkpoints = _api_checkpoints()
        if not checkpoints:
            raise RuntimeError("No checkpoint model found in ComfyUI.")
        preferred_checkpoint = plan.model or get_comfyui_checkpoint()
        checkpoint = preferred_checkpoint if preferred_checkpoint in checkpoints else checkpoints[0]
        seed = int(time.time() * 1000) % 2_147_483_647 if plan.seed == "random" else int(plan.seed)
        payload_workflow = (
            self.patcher.patch(workflow, plan, checkpoint=checkpoint, seed=seed)
            if workflow
            else self.patcher.build_basic_workflow(checkpoint, plan, seed=seed)
        )
        queued = _post_json("/prompt", {"prompt": payload_workflow, "client_id": f"trl-quality-{uuid4().hex}"}, timeout=30)
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return a prompt_id: {queued}")
        timeout_at = time.time() + get_comfyui_timeout_seconds()
        while time.time() < timeout_at:
            history = _url_json(f"/history/{prompt_id}", timeout=10)
            item = history.get(prompt_id)
            if item:
                artifacts = []
                for output in item.get("outputs", {}).values():
                    for image_info in output.get("images", []):
                        artifacts.append(copy_comfy_image(image_info, plan))
                if artifacts:
                    return artifacts
            time.sleep(1)
        raise TimeoutError("ComfyUI generation timed out before history output was available.")


def parse_size(size: str) -> tuple[int, int]:
    try:
        width_text, height_text = (size or "768x512").lower().split("x", 1)
        return round_to_multiple(max(256, int(width_text)), 8), round_to_multiple(max(256, int(height_text)), 8)
    except Exception:
        return 768, 512


def round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def image_url_for_name(filename: str) -> str:
    return f"/generated-images/{filename}"


def copy_comfy_image(image_info: dict, plan: ImageGenerationPlan) -> dict:
    query = urllib.parse.urlencode(
        {
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
    )
    with urllib.request.urlopen(f"{get_comfyui_url()}/view?{query}", timeout=60) as response:
        data = response.read()
    output_dir = get_generated_images_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(str(image_info.get("filename", "image.png"))).suffix or ".png"
    image_id = uuid4().hex
    target = output_dir / f"quality_{time.strftime('%Y%m%d_%H%M%S')}_{image_id}{ext}"
    target.write_bytes(data)
    width, height = parse_size(plan.size)
    return {
        "id": image_id,
        "url": image_url_for_name(target.name),
        "path": str(target),
        "mime_type": "image/png",
        "width": width,
        "height": height,
        "size": plan.size,
        "prompt": plan.positive_prompt,
        "negative_prompt": plan.negative_prompt,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
