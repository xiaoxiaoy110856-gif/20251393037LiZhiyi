from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from scripts.add_linucb_explanation_slides import THEME, get_specs

OUT = Path("outputs/ppt_previews/linucb_addendum_png")
W, H = 13.333333, 7.5


def color(value: str) -> str:
    return "#" + value.replace("#", "")


def wrap_text(value: str, width: int) -> str:
    lines = []
    for para in str(value).split("\n"):
        lines.append("\n".join(textwrap.wrap(para, width=width, break_long_words=True) or [""]))
    return "\n".join(lines)


def add_text(ax, x, y, w, h, value, size=14, c=None, bold=False, align="center", font=None):
    chars = max(4, int(w * 7.2 / max(size / 14, 0.1)))
    ha = "center" if align in {"ctr", "c", "center"} else ("right" if align in {"r", "right"} else "left")
    tx = x + w / 2 if ha == "center" else (x + w - 0.08 if ha == "right" else x + 0.08)
    ax.text(
        tx,
        y + h / 2,
        wrap_text(value, chars),
        va="center",
        ha=ha,
        fontsize=size,
        color=color(c or THEME["ink"]),
        fontproperties=font,
        fontweight="bold" if bold else "normal",
        linespacing=1.2,
    )


def render(spec: dict, idx: int, font) -> Path:
    fig, ax = plt.subplots(figsize=(W, H), dpi=144)
    fig.patch.set_facecolor(color(spec.get("bg", THEME["bg"])))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, 0.16, color=color(THEME["green"])))
    ax.add_patch(Rectangle((0, 0.16), W, 0.07, color=color(THEME["pink"])))
    add_text(ax, 0.62, 0.34, 9.3, 0.62, spec["title"], 27, THEME["ink"], True, "left", font)
    add_text(ax, 0.64, 0.94, 9.6, 0.42, spec.get("subtitle", ""), 12, THEME["muted"], False, "left", font)
    ax.add_patch(
        FancyBboxPatch(
            (10.35, 0.48),
            2.25,
            0.38,
            boxstyle="round,pad=0.015,rounding_size=.12",
            facecolor=color(THEME["green2"]),
            edgecolor=color(THEME["green"]),
            linewidth=1.2,
        )
    )
    add_text(ax, 10.35, 0.48, 2.25, 0.38, spec.get("tag", "RL-RAG 答辩页"), 11, THEME["line"], True, "center", font)
    for obj in spec["objects"]:
        if obj["kind"] == "box":
            ax.add_patch(
                FancyBboxPatch(
                    (obj["x"], obj["y"]),
                    obj["w"],
                    obj["h"],
                    boxstyle="round,pad=0.015,rounding_size=.12",
                    facecolor=color(obj.get("fill", THEME["white"])),
                    edgecolor=color(obj.get("outline", THEME["green"])),
                    linewidth=1.4,
                )
            )
            add_text(ax, obj["x"], obj["y"], obj["w"], obj["h"], obj.get("text", ""), obj.get("font", 15), obj.get("color", THEME["ink"]), obj.get("bold", False), obj.get("align", "ctr"), font)
        elif obj["kind"] == "text":
            add_text(ax, obj["x"], obj["y"], obj["w"], obj["h"], obj.get("text", ""), obj.get("font", 16), obj.get("color", THEME["ink"]), obj.get("bold", False), obj.get("align", "l"), font)
        elif obj["kind"] == "line":
            ax.add_patch(
                FancyArrowPatch(
                    (obj["x1"], obj["y1"]),
                    (obj["x2"], obj["y2"]),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=obj.get("width", 2),
                    color=color(obj.get("color", THEME["line"])),
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"slide{idx}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out


def main() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    for idx, spec in enumerate(get_specs(), 58):
        print(render(spec, idx, font))


if __name__ == "__main__":
    main()
