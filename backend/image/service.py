from __future__ import annotations

import html
import json
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from backend.settings import (
    comfyui_enabled,
    get_comfyui_dir,
    get_comfyui_url,
    get_generated_images_dir,
    get_image_generator_command,
    get_image_provider,
)
from backend.image.generation_service import ImageGenerationService


IMAGE_MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth"}


def _artifact_from_legacy_result(result: dict, prompt: str, size: str = "1024x1024", quality: str = "auto", image_format: str = "png") -> dict:
    """把旧版图片生成返回值统一转换成前端可展示的 artifact 结构。"""
    url = str(result.get("url") or "")
    path = str(result.get("path") or "")
    if not url:
        return result
    suffix = Path(path or url).suffix.lower()
    mime_type = {
        ".svg": "image/svg+xml",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    final_format = "svg" if suffix == ".svg" else image_format
    image_id = Path(path or url).stem or uuid4().hex
    result.setdefault("type", "image_result")
    result.setdefault("provider", str(result.get("mode") or "local"))
    result.setdefault("model", str(result.get("mode") or "local"))
    result.setdefault(
        "images",
        [
            {
                "id": image_id,
                "url": url,
                "path": path,
                "mime_type": mime_type,
                "size": size,
                "quality": quality or "auto",
                "format": final_format,
                "prompt": prompt,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        ],
    )
    return result


def _safe_svg_text(text: str, limit: int = 260) -> str:
    value = " ".join((text or "").split())[:limit]
    return html.escape(value or "Untitled image prompt")


def _write_prompt_svg(prompt: str, detail: str = "") -> Path:
    output_dir = get_generated_images_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.svg"
    prompt_text = _safe_svg_text(prompt)
    detail_text = _safe_svg_text(detail or "配置 ComfyUI 模型后可生成真实图片。", limit=180)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f9fbff"/>
      <stop offset="0.52" stop-color="#eef4ff"/>
      <stop offset="1" stop-color="#fff0f7"/>
    </linearGradient>
  </defs>
  <rect width="1024" height="1024" rx="56" fill="url(#bg)"/>
  <rect x="80" y="92" width="864" height="840" rx="42" fill="#ffffff" stroke="#dfe7ff" stroke-width="3"/>
  <circle cx="836" cy="188" r="54" fill="#ff81ba" opacity="0.22"/>
  <circle cx="184" cy="824" r="72" fill="#5c7cff" opacity="0.16"/>
  <text x="116" y="174" font-family="Microsoft YaHei, Segoe UI, sans-serif" font-size="34" font-weight="700" fill="#2a2f45">Image Generation Draft</text>
  <text x="116" y="232" font-family="Microsoft YaHei, Segoe UI, sans-serif" font-size="22" fill="#70789a">{detail_text}</text>
  <foreignObject x="116" y="302" width="792" height="430">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Microsoft YaHei, Segoe UI, sans-serif; font-size: 36px; line-height: 1.45; color: #2a2f45; font-weight: 650;">
      {prompt_text}
    </div>
  </foreignObject>
  <text x="116" y="850" font-family="Microsoft YaHei, Segoe UI, sans-serif" font-size="22" fill="#70789a">ComfyUI: {html.escape(str(get_comfyui_dir()))}</text>
</svg>
"""
    target.write_text(svg, encoding="utf-8")
    return target


def _url_json(path: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(f"{get_comfyui_url()}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(path: str, payload: dict, timeout: int = 30) -> dict:
    request = urllib.request.Request(
        f"{get_comfyui_url()}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _local_checkpoints() -> list[str]:
    checkpoints_dir = get_comfyui_dir() / "models" / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    return [
        path.name
        for path in checkpoints_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_MODEL_SUFFIXES
    ]


def _api_checkpoints() -> list[str]:
    try:
        info = _url_json("/object_info", timeout=5)
        ckpt = info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [])
        if isinstance(ckpt, list) and ckpt and isinstance(ckpt[0], list):
            return [str(item) for item in ckpt[0]]
    except Exception:
        return []
    return []


def comfyui_status() -> dict:
    """检查 ComfyUI 是否启用、服务是否可达、当前有哪些 checkpoint。"""
    online = False
    detail = ""
    try:
        _url_json("/system_stats", timeout=3)
        online = True
        detail = f"ComfyUI is reachable at {get_comfyui_url()}"
    except Exception as error:
        detail = f"ComfyUI is not reachable at {get_comfyui_url()}: {error}"
    checkpoints = _api_checkpoints() if online else _local_checkpoints()
    return {
        "enabled": comfyui_enabled(),
        "online": online,
        "url": get_comfyui_url(),
        "directory": str(get_comfyui_dir()),
        "checkpoints": checkpoints,
        "ready": online and bool(checkpoints),
        "detail": detail if checkpoints else f"{detail}. No checkpoint model found in models/checkpoints.",
    }


def _basic_workflow(prompt: str, checkpoint: str, width: int = 768, height: int = 768, steps: int = 20) -> dict:
    seed = int(time.time() * 1000) % 2_147_483_647
    return {
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, blurry, distorted, watermark, text", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "trl_assistant", "images": ["8", 0]}},
    }


def _copy_comfy_image(image_info: dict) -> Path:
    query = urllib.parse.urlencode(
        {
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
    )
    with urllib.request.urlopen(f"{get_comfyui_url()}/view?{query}", timeout=30) as response:
        data = response.read()
    output_dir = get_generated_images_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(str(image_info.get("filename", "image.png"))).suffix or ".png"
    target = output_dir / f"comfy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}{ext}"
    target.write_bytes(data)
    return target


def _generate_with_comfyui(prompt: str, model: str = "", size: str = "768x768") -> dict:
    """通过 ComfyUI 工作流生成图片，并把输出文件复制到 public generated-images 目录。"""
    status = comfyui_status()
    if not status["online"]:
        raise RuntimeError(status["detail"])
    checkpoints = status["checkpoints"]
    if not checkpoints:
        raise RuntimeError(status["detail"])
    checkpoint = model if model in checkpoints else checkpoints[0]
    try:
        width_text, height_text = (size or "768x768").lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except Exception:
        width, height = 768, 768
    workflow = _basic_workflow(prompt, checkpoint=checkpoint, width=width, height=height)
    queued = _post_json("/prompt", {"prompt": workflow, "client_id": f"trl-{uuid4().hex}"}, timeout=30)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {queued}")
    for _ in range(240):
        history = _url_json(f"/history/{prompt_id}", timeout=10)
        item = history.get(prompt_id)
        if item:
            outputs = item.get("outputs", {})
            for output in outputs.values():
                images = output.get("images", [])
                if images:
                    target = _copy_comfy_image(images[0])
                    return {
                        "ok": True,
                        "mode": "comfyui",
                        "path": str(target),
                        "url": f"/generated-images/{target.name}",
                        "detail": f"ComfyUI generated image with checkpoint {checkpoint}",
                    }
        time.sleep(1)
    raise RuntimeError("ComfyUI generation timed out while waiting for history output.")


def _generate_with_command(prompt: str, model: str = "", size: str = "1024x1024") -> dict:
    """通过外部命令生成图片，适合接入自定义脚本或本地生成器。"""
    command = get_image_generator_command()
    output_dir = get_generated_images_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.png"
    payload = {
        "prompt": prompt,
        "model": model,
        "size": size,
        "output": str(target),
    }
    completed = subprocess.run(
        command,
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        shell=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if completed.returncode == 0 and target.exists():
        return {"ok": True, "mode": "external", "path": str(target), "url": f"/generated-images/{target.name}", "detail": completed.stdout.strip()}
    detail = completed.stderr.strip() or completed.stdout.strip() or "Image generator command did not create an output file."
    raise RuntimeError(detail)


# 核心3：简单图片生成路径，供 `/api/images/generate` 使用，也作为高级 Agent 图片路径之外的备用通道。
def generate_image(
    prompt: str,
    model: str = "",
    size: str = "768x768",
    quality: str = "",
    format: str = "",
    background: str = "",
    n: int = 1,
    style_notes: str = "",
    user_visible_prompt: str = "",
) -> dict:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Image prompt cannot be empty.")

    if get_image_provider() == "openai":
        result = ImageGenerationService(provider_name="openai", model=model or None).generate_image(
            prompt=prompt,
            size=size,
            quality=quality,
            format=format,
            background=background,
            n=n,
            style_notes=style_notes,
            user_visible_prompt=user_visible_prompt,
        )
        if result.get("images"):
            first = result["images"][0]
            result.update({"path": first.get("path", ""), "url": first.get("url", ""), "detail": "OpenAI image generated."})
        return result

    if comfyui_enabled():
        try:
            return _artifact_from_legacy_result(_generate_with_comfyui(prompt, model=model, size=size), prompt=prompt, size=size)
        except Exception as error:
            target = _write_prompt_svg(prompt, detail=f"ComfyUI 暂不可用：{error}")
            return _artifact_from_legacy_result({
                "ok": True,
                "mode": "svg-fallback",
                "path": str(target),
                "url": f"/generated-images/{target.name}",
                "detail": f"ComfyUI 暂不可用，已生成占位图：{error}",
            }, prompt=prompt, size=size, image_format="svg")

    if get_image_generator_command():
        return _artifact_from_legacy_result(_generate_with_command(prompt, model=model, size=size), prompt=prompt, size=size)

    target = _write_prompt_svg(prompt)
    return _artifact_from_legacy_result({
        "ok": True,
        "mode": "svg-fallback",
        "path": str(target),
        "url": f"/generated-images/{target.name}",
        "detail": "未配置真实图片模型，已生成提示词 SVG 图像卡片。",
    }, prompt=prompt, size=size, image_format="svg")
