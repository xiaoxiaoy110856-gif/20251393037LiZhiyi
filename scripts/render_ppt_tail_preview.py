from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "outputs" / "ppt_tail_rl_specs.json"
OUT_DIR = ROOT / "outputs" / "ppt_tail_preview"
W, H = 1920, 1080
SCALE = 144


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), max(8, int(size * 1.35)))
    return ImageFont.load_default()


def xywh(el: dict) -> tuple[int, int, int, int]:
    x = int(float(el["x"]) * SCALE)
    y = int(float(el["y"]) * SCALE)
    w = int(float(el["w"]) * SCALE)
    h = int(float(el["h"]) * SCALE)
    return x, y, w, h


def draw_wrapped(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fnt: ImageFont.FreeTypeFont, fill: str, align: str) -> None:
    x, y, w, h = box
    paras = text.split("\n")
    lines: list[str] = []
    for para in paras:
        current = ""
        for ch in para:
            test = current + ch
            if draw.textbbox((0, 0), test, font=fnt)[2] <= w - 20 or not current:
                current = test
            else:
                lines.append(current)
                current = ch
        lines.append(current)
    line_h = int((draw.textbbox((0, 0), "国Ag", font=fnt)[3] - draw.textbbox((0, 0), "国Ag", font=fnt)[1]) * 1.25)
    total_h = line_h * len(lines)
    yy = y + max(0, (h - total_h) // 2)
    for line in lines:
        tw = draw.textbbox((0, 0), line, font=fnt)[2]
        if align == "c":
            xx = x + (w - tw) // 2
        elif align == "r":
            xx = x + w - tw - 10
        else:
            xx = x + 10
        draw.text((xx, yy), line, font=fnt, fill="#" + fill)
        yy += line_h


def render_slide(spec: dict, index: int) -> Path:
    img = Image.new("RGB", (W, H), "#FDF7FA")
    draw = ImageDraw.Draw(img)
    for el in spec["elements"]:
        x, y, w, h = xywh(el)
        if el["kind"] == "rect":
            fill = "#" + str(el.get("fill", "FFFFFF"))
            line = "#" + str(el.get("line")) if el.get("line") else None
            radius = int(min(w, h) * 0.18) if el.get("radius") else 0
            if radius:
                draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=line, width=3 if line else 1)
            else:
                draw.rectangle([x, y, x + w, y + h], fill=fill, outline=line, width=3 if line else 1)
        elif el["kind"] == "text":
            fill = el.get("fill")
            line = el.get("line")
            radius = int(min(w, h) * 0.18) if el.get("radius") else 0
            if fill:
                if radius:
                    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill="#" + fill, outline="#" + line if line else None, width=3 if line else 1)
                else:
                    draw.rectangle([x, y, x + w, y + h], fill="#" + fill, outline="#" + line if line else None, width=3 if line else 1)
            fnt = font(int(el.get("size", 18)), bool(el.get("bold")))
            draw_wrapped(draw, (x + 6, y + 4, w - 12, h - 8), str(el.get("text", "")), fnt, str(el.get("color", "17231F")), str(el.get("align", "l")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"slide{index}.png"
    img.save(path)
    return path


def make_montage(paths: list[Path]) -> Path:
    thumbs = [Image.open(path).resize((480, 270)) for path in paths]
    rows = (len(thumbs) + 1) // 2
    montage = Image.new("RGB", (960, 270 * rows), "#FFFFFF")
    for i, thumb in enumerate(thumbs):
        x = (i % 2) * 480
        y = (i // 2) * 270
        montage.paste(thumb, (x, y))
    out = OUT_DIR / "tail_montage.png"
    montage.save(out)
    return out


def main() -> None:
    spec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SPEC_PATH
    raw_specs = json.loads(spec_path.read_text(encoding="utf-8"))
    items = []
    for index, item in enumerate(raw_specs):
        if isinstance(item, dict) and "slide_no" in item and "spec" in item:
            items.append((int(item["slide_no"]), item["spec"]))
        else:
            items.append((50 + index, item))
    paths = [render_slide(spec, slide_no) for slide_no, spec in items]
    montage = make_montage(paths)
    print(json.dumps({"slides": [str(path) for path in paths], "montage": str(montage)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
