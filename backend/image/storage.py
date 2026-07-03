from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from backend.image.generation_types import ImageArtifact, MIME_BY_FORMAT
from backend.settings import get_generated_images_dir, get_image_public_base_url


def image_url_for_name(filename: str) -> str:
    base_url = get_image_public_base_url()
    if base_url:
        return f"{base_url}/{filename}"
    return f"/generated-images/{filename}"


def save_image_bytes(
    data: bytes,
    image_format: str,
    size: str,
    quality: str,
    prompt: str,
) -> ImageArtifact:
    output_dir = get_generated_images_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_id = uuid4().hex
    safe_format = "jpg" if image_format == "jpeg" else image_format
    filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_id}.{safe_format}"
    target = output_dir / filename
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        partial.write_bytes(data)
        partial.replace(target)
    except Exception:
        if partial.exists():
            partial.unlink(missing_ok=True)
        if target.exists():
            target.unlink(missing_ok=True)
        raise
    created_at = datetime.now().isoformat(timespec="seconds")
    return ImageArtifact(
        id=image_id,
        url=image_url_for_name(filename),
        path=str(target),
        mime_type=MIME_BY_FORMAT[image_format],
        size=size,
        quality=quality,
        format=image_format,
        prompt=prompt,
        created_at=created_at,
    )
