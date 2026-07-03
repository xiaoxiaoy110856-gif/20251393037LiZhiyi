from __future__ import annotations

import os
import re
import shutil
import site
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape


P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
APP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"

for prefix, uri in {
    "p": P_NS,
    "a": A_NS,
    "r": R_NS,
    "": CT_NS,
}.items():
    ET.register_namespace(prefix, uri)

EMU = 914400
SLIDE_W_IN = 13.333333
SLIDE_H_IN = 7.5
PX_PER_IN = 144

PPTX_PATH = Path(os.environ.get("PPTX_PATH", r"C:\Users\Lenovo\Desktop\软件过程管理.pptx"))
PREVIEW_DIR = Path("outputs") / "ppt_previews" / "linucb_addendum"


THEME = {
    "bg": "FFF5FA",
    "ink": "17312B",
    "muted": "58746B",
    "pink": "FF77B7",
    "pink2": "FFD6EA",
    "green": "35C780",
    "green2": "DDF8E9",
    "mint": "A7F3D0",
    "white": "FFFFFF",
    "line": "245B49",
    "yellow": "FFE082",
    "purple": "9B7CFF",
}


def qn(tag: str) -> str:
    prefix, name = tag.split(":", 1)
    return f"{{{ {'p': P_NS, 'a': A_NS, 'r': R_NS}[prefix] }}}{name}"


def emu(value_in: float) -> str:
    return str(int(round(value_in * EMU)))


def rgb(value: str) -> str:
    return value.replace("#", "").upper()


def add(parent: ET.Element, tag: str, attrs: dict | None = None) -> ET.Element:
    node = ET.SubElement(parent, qn(tag), attrs or {})
    return node


def solid_fill(parent: ET.Element, color: str, alpha: int | None = None) -> None:
    fill = add(parent, "a:solidFill")
    clr = add(fill, "a:srgbClr", {"val": rgb(color)})
    if alpha is not None:
        add(clr, "a:alpha", {"val": str(alpha)})


def line_fill(parent: ET.Element, color: str, width_pt: float = 1.0) -> None:
    ln = add(parent, "a:ln", {"w": str(int(width_pt * 12700))})
    solid_fill(ln, color)


def no_line(parent: ET.Element) -> None:
    ln = add(parent, "a:ln")
    add(ln, "a:noFill")


def add_xfrm(sp_pr: ET.Element, x: float, y: float, w: float, h: float) -> None:
    xfrm = add(sp_pr, "a:xfrm")
    add(xfrm, "a:off", {"x": emu(x), "y": emu(y)})
    add(xfrm, "a:ext", {"cx": emu(w), "cy": emu(h)})


def text_body(parent: ET.Element, text: str, font_size: int, color: str, bold: bool = False, align: str = "ctr") -> None:
    tx = add(parent, "p:txBody")
    body_pr = add(tx, "a:bodyPr", {"wrap": "square", "rtlCol": "0", "anchor": "mid", "lIns": "91440", "rIns": "91440", "tIns": "45720", "bIns": "45720"})
    add(body_pr, "a:normAutofit", {"fontScale": "85000", "lnSpcReduction": "10000"})
    add(tx, "a:lstStyle")
    paragraphs = [line for line in text.split("\n")]
    for line in paragraphs:
        p = add(tx, "a:p")
        ppr = add(p, "a:pPr", {"algn": align})
        ln_spc = add(ppr, "a:lnSpc")
        add(ln_spc, "a:spcPct", {"val": "93000"})
        r = add(p, "a:r")
        rpr_attrs = {"lang": "zh-CN", "sz": str(font_size * 100)}
        if bold:
            rpr_attrs["b"] = "1"
        rpr = add(r, "a:rPr", rpr_attrs)
        solid_fill(rpr, color)
        add(rpr, "a:latin", {"typeface": "Microsoft YaHei"})
        add(rpr, "a:ea", {"typeface": "Microsoft YaHei"})
        t = add(r, "a:t")
        t.text = line
    if not paragraphs:
        add(tx, "a:p")


def shape(sp_tree: ET.Element, sid: int, name: str, x: float, y: float, w: float, h: float, text: str = "", *,
          fill: str = "FFFFFF", outline: str | None = None, prst: str = "roundRect", font_size: int = 18,
          color: str = "17312B", bold: bool = False, align: str = "ctr") -> int:
    sp = add(sp_tree, "p:sp")
    nv = add(sp, "p:nvSpPr")
    add(nv, "p:cNvPr", {"id": str(sid), "name": name})
    add(nv, "p:cNvSpPr")
    add(nv, "p:nvPr")
    sp_pr = add(sp, "p:spPr")
    add_xfrm(sp_pr, x, y, w, h)
    geom = add(sp_pr, "a:prstGeom", {"prst": prst})
    add(geom, "a:avLst")
    solid_fill(sp_pr, fill)
    if outline:
        line_fill(sp_pr, outline, 1.2)
    else:
        no_line(sp_pr)
    if text:
        text_body(sp, text, font_size, color, bold=bold, align=align)
    return sid + 1


def textbox(sp_tree: ET.Element, sid: int, name: str, x: float, y: float, w: float, h: float, text: str, *,
            font_size: int = 24, color: str = "17312B", bold: bool = False, align: str = "l") -> int:
    sp = add(sp_tree, "p:sp")
    nv = add(sp, "p:nvSpPr")
    add(nv, "p:cNvPr", {"id": str(sid), "name": name})
    add(nv, "p:cNvSpPr", {"txBox": "1"})
    add(nv, "p:nvPr")
    sp_pr = add(sp, "p:spPr")
    add_xfrm(sp_pr, x, y, w, h)
    geom = add(sp_pr, "a:prstGeom", {"prst": "rect"})
    add(geom, "a:avLst")
    add(sp_pr, "a:noFill")
    no_line(sp_pr)
    text_body(sp, text, font_size, color, bold=bold, align={"l": "l", "c": "ctr", "r": "r"}.get(align, align))
    return sid + 1


def connector(sp_tree: ET.Element, sid: int, name: str, x1: float, y1: float, x2: float, y2: float, *,
              color: str = "245B49", width_pt: float = 2.0) -> int:
    cxn = add(sp_tree, "p:cxnSp")
    nv = add(cxn, "p:nvCxnSpPr")
    add(nv, "p:cNvPr", {"id": str(sid), "name": name})
    add(nv, "p:cNvCxnSpPr")
    add(nv, "p:nvPr")
    sp_pr = add(cxn, "p:spPr")
    attrs = {}
    if x2 < x1:
        attrs["flipH"] = "1"
    if y2 < y1:
        attrs["flipV"] = "1"
    xfrm = add(sp_pr, "a:xfrm", attrs)
    add(xfrm, "a:off", {"x": emu(min(x1, x2)), "y": emu(min(y1, y2))})
    add(xfrm, "a:ext", {"cx": emu(abs(x2 - x1) or 0.01), "cy": emu(abs(y2 - y1) or 0.01)})
    geom = add(sp_pr, "a:prstGeom", {"prst": "straightConnector1"})
    add(geom, "a:avLst")
    ln = add(sp_pr, "a:ln", {"w": str(int(width_pt * 12700)), "cap": "round"})
    solid_fill(ln, color)
    add(ln, "a:headEnd", {"type": "triangle", "w": "med", "len": "med"})
    return sid + 1


def build_slide(spec: dict) -> bytes:
    sld = ET.Element(qn("p:sld"))
    c_sld = add(sld, "p:cSld")
    sp_tree = add(c_sld, "p:spTree")
    nv_grp = add(sp_tree, "p:nvGrpSpPr")
    add(nv_grp, "p:cNvPr", {"id": "1", "name": ""})
    add(nv_grp, "p:cNvGrpSpPr")
    add(nv_grp, "p:nvPr")
    grp_pr = add(sp_tree, "p:grpSpPr")
    xfrm = add(grp_pr, "a:xfrm")
    add(xfrm, "a:off", {"x": "0", "y": "0"})
    add(xfrm, "a:ext", {"cx": "0", "cy": "0"})
    add(xfrm, "a:chOff", {"x": "0", "y": "0"})
    add(xfrm, "a:chExt", {"cx": "0", "cy": "0"})

    sid = 2
    sid = shape(sp_tree, sid, "background", 0, 0, SLIDE_W_IN, SLIDE_H_IN, fill=spec.get("bg", THEME["bg"]), prst="rect")
    sid = shape(sp_tree, sid, "top-ribbon", 0, 0, SLIDE_W_IN, 0.16, fill=THEME["green"], prst="rect")
    sid = shape(sp_tree, sid, "pink-ribbon", 0, 0.16, SLIDE_W_IN, 0.07, fill=THEME["pink"], prst="rect")
    sid = textbox(sp_tree, sid, "slide-title", 0.62, 0.42, 8.8, 0.58, spec["title"], font_size=27, color=THEME["ink"], bold=True, align="l")
    sid = textbox(sp_tree, sid, "slide-subtitle", 0.64, 1.02, 9.4, 0.36, spec.get("subtitle", ""), font_size=12, color=THEME["muted"], bold=False, align="l")
    sid = shape(sp_tree, sid, "section-tag", 10.35, 0.48, 2.25, 0.38, spec.get("tag", "RL-RAG 答辩页"), fill=THEME["green2"], outline=THEME["green"], font_size=11, color=THEME["line"], bold=True)

    for obj in spec["objects"]:
        kind = obj["kind"]
        if kind == "box":
            sid = shape(sp_tree, sid, obj.get("name", "box"), obj["x"], obj["y"], obj["w"], obj["h"], obj.get("text", ""),
                        fill=obj.get("fill", THEME["white"]), outline=obj.get("outline", THEME["green"]), prst=obj.get("prst", "roundRect"),
                        font_size=obj.get("font", 15), color=obj.get("color", THEME["ink"]), bold=obj.get("bold", False), align=obj.get("align", "ctr"))
        elif kind == "text":
            sid = textbox(sp_tree, sid, obj.get("name", "text"), obj["x"], obj["y"], obj["w"], obj["h"], obj.get("text", ""),
                          font_size=obj.get("font", 16), color=obj.get("color", THEME["ink"]), bold=obj.get("bold", False), align=obj.get("align", "l"))
        elif kind == "line":
            sid = connector(sp_tree, sid, obj.get("name", "line"), obj["x1"], obj["y1"], obj["x2"], obj["y2"],
                            color=obj.get("color", THEME["line"]), width_pt=obj.get("width", 2.0))

    clr = add(sld, "p:clrMapOvr")
    add(clr, "a:masterClrMapping")
    return ET.tostring(sld, encoding="utf-8", xml_declaration=True)


def slide_relationship_xml(layout_target: str) -> bytes:
    root = ET.Element("Relationships", {"xmlns": PKG_REL_NS})
    ET.SubElement(root, "Relationship", {
        "Id": "rId1",
        "Type": f"{R_NS}/slideLayout",
        "Target": layout_target,
    })
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def get_specs() -> list[dict]:
    return [
        {
            "title": "策略分层模型图：Qwen + RAG + LinUCB",
            "subtitle": "LinUCB 做检索动作选择；PPO/DPO/ORPO 不替代它，而是在 Agent 行为层和答案偏好层继续增强。",
            "tag": "模型图",
            "objects": [
                {"kind": "box", "x": 0.75, "y": 1.65, "w": 1.6, "h": 0.66, "text": "用户问题\nQuery", "fill": THEME["pink2"], "outline": THEME["pink"], "font": 15, "bold": True},
                {"kind": "line", "x1": 2.35, "y1": 1.98, "x2": 2.78, "y2": 1.98},
                {"kind": "box", "x": 2.78, "y": 1.55, "w": 1.75, "h": 0.86, "text": "Qwen Planner\n理解意图\n规划工具", "fill": THEME["white"], "outline": THEME["green"], "font": 12, "bold": True},
                {"kind": "line", "x1": 4.53, "y1": 1.98, "x2": 5.0, "y2": 1.98},
                {"kind": "box", "x": 5.0, "y": 1.4, "w": 2.0, "h": 1.1, "text": "Context\n压缩历史\n组装证据槽", "fill": "F3FFF7", "outline": THEME["green"], "font": 12, "bold": True},
                {"kind": "line", "x1": 7.0, "y1": 1.98, "x2": 7.45, "y2": 1.98},
                {"kind": "box", "x": 7.45, "y": 1.32, "w": 2.15, "h": 1.25, "text": "LinUCB\n选择检索 action", "fill": THEME["green"], "outline": THEME["line"], "font": 16, "color": "FFFFFF", "bold": True},
                {"kind": "line", "x1": 9.6, "y1": 1.98, "x2": 10.05, "y2": 1.98},
                {"kind": "box", "x": 10.05, "y": 1.55, "w": 1.9, "h": 0.86, "text": "RAG 证据\n论文 / 主题 / 段落", "fill": THEME["white"], "outline": THEME["green"], "font": 13, "bold": True},
                {"kind": "line", "x1": 10.95, "y1": 2.42, "x2": 10.95, "y2": 3.16},
                {"kind": "box", "x": 9.2, "y": 3.16, "w": 2.95, "h": 0.78, "text": "Qwen 最终回答：基于证据组织语言", "fill": THEME["pink2"], "outline": THEME["pink"], "font": 14, "bold": True},
                {"kind": "box", "x": 0.9, "y": 4.45, "w": 3.35, "h": 1.32, "text": "PPO\n多步 Agent 行为策略\n是否检索 / 是否读文件\n是否再搜", "fill": "FFFFFF", "outline": THEME["purple"], "font": 12, "bold": True},
                {"kind": "box", "x": 4.9, "y": 4.45, "w": 3.35, "h": 1.32, "text": "DPO\n答案偏好优化\n更重证据 / 少幻觉 / 结构清楚", "fill": "FFFFFF", "outline": THEME["pink"], "font": 13, "bold": True},
                {"kind": "box", "x": 8.9, "y": 4.45, "w": 3.35, "h": 1.32, "text": "ORPO\n轻量偏好对照\n低成本验证回答偏好收益", "fill": "FFFFFF", "outline": THEME["green"], "font": 13, "bold": True},
                {"kind": "text", "x": 1.05, "y": 6.05, "w": 11.1, "h": 0.62, "text": "结论：LinUCB 是检索层主策略；PPO/DPO/ORPO 是增强层。\n它们证明 Agent 不只会搜，还能更会决策、更会回答。", "font": 14, "bold": True, "color": THEME["line"], "align": "c"},
            ],
        },
        {
            "title": "LinUCB 是什么：单步检索策略选择器",
            "subtitle": "每来一个问题，先抽取 query 特征，再在多个 retrieval action 里选一个最有希望命中证据的动作。",
            "tag": "LinUCB",
            "objects": [
                {"kind": "box", "x": 0.85, "y": 1.58, "w": 3.0, "h": 1.1, "text": "上下文 x\n关键词、意图、领域、历史摘要", "fill": THEME["pink2"], "outline": THEME["pink"], "font": 14, "bold": True},
                {"kind": "line", "x1": 3.85, "y1": 2.13, "x2": 4.55, "y2": 2.13},
                {"kind": "box", "x": 4.55, "y": 1.42, "w": 3.9, "h": 1.42, "text": "UCB(a) = 预测收益 + 不确定性奖励\n既利用已知好动作，也保留探索", "fill": THEME["white"], "outline": THEME["green"], "font": 15, "bold": True},
                {"kind": "line", "x1": 8.45, "y1": 2.13, "x2": 9.15, "y2": 2.13},
                {"kind": "box", "x": 9.15, "y": 1.58, "w": 3.0, "h": 1.1, "text": "输出 action\n向量 / 关键词 / 混合 / 扩 top_k", "fill": THEME["green2"], "outline": THEME["green"], "font": 14, "bold": True},
                {"kind": "text", "x": 0.85, "y": 3.35, "w": 3.1, "h": 0.36, "text": "为什么适合本项目", "font": 18, "bold": True, "color": THEME["line"]},
                {"kind": "box", "x": 0.85, "y": 3.85, "w": 3.1, "h": 1.45, "text": "1. 检索是单步选择\n2. action 离散明确\n3. 奖励可直接来自命中率\n4. 可解释，方便答辩", "fill": "FFFFFF", "outline": THEME["green"], "font": 13, "align": "l"},
                {"kind": "text", "x": 4.35, "y": 3.35, "w": 3.4, "h": 0.36, "text": "它学到什么", "font": 18, "bold": True, "color": THEME["line"]},
                {"kind": "box", "x": 4.35, "y": 3.85, "w": 3.4, "h": 1.45, "text": "如果问题像“轨迹压缩”\n就偏向论文主题检索；\n如果问题像“某方法定义”\n就偏向精确关键词检索。", "fill": "FFFFFF", "outline": THEME["pink"], "font": 13, "align": "l"},
                {"kind": "text", "x": 8.25, "y": 3.35, "w": 3.9, "h": 0.36, "text": "和 PPO / DDQN 的区别", "font": 18, "bold": True, "color": THEME["line"]},
                {"kind": "box", "x": 8.25, "y": 3.85, "w": 3.9, "h": 1.45, "text": "LinUCB：轻量、稳定、上线优先\nDDQN：神经网络策略对照\nPPO：适合多步工具链，不是单步最优", "fill": "FFFFFF", "outline": THEME["purple"], "font": 13, "align": "l"},
            ],
        },
        {
            "title": "LinUCB 最优后，PPO/DPO/ORPO 的作用",
            "subtitle": "有效，但分工不同：它们不抢 LinUCB 的检索主策略位置，而是优化 Agent 的行为链和答案偏好。",
            "tag": "分工",
            "objects": [
                {"kind": "box", "x": 0.75, "y": 1.58, "w": 2.55, "h": 1.15, "text": "检索策略层\nLinUCB", "fill": THEME["green"], "outline": THEME["line"], "font": 18, "color": "FFFFFF", "bold": True},
                {"kind": "box", "x": 3.72, "y": 1.58, "w": 2.55, "h": 1.15, "text": "神经网络对照\nDDQN", "fill": THEME["green2"], "outline": THEME["green"], "font": 16, "bold": True},
                {"kind": "box", "x": 6.69, "y": 1.58, "w": 2.55, "h": 1.15, "text": "多步工具策略\nPPO", "fill": "F2EFFF", "outline": THEME["purple"], "font": 16, "bold": True},
                {"kind": "box", "x": 9.66, "y": 1.58, "w": 2.55, "h": 1.15, "text": "答案偏好\nDPO / ORPO", "fill": THEME["pink2"], "outline": THEME["pink"], "font": 16, "bold": True},
                {"kind": "line", "x1": 3.3, "y1": 2.15, "x2": 3.7, "y2": 2.15},
                {"kind": "line", "x1": 6.27, "y1": 2.15, "x2": 6.67, "y2": 2.15},
                {"kind": "line", "x1": 9.24, "y1": 2.15, "x2": 9.64, "y2": 2.15},
                {"kind": "box", "x": 0.75, "y": 3.18, "w": 2.55, "h": 1.5, "text": "上线主线\n给 query 选 action\n提高 Source / Topic Hit", "fill": "FFFFFF", "outline": THEME["green"], "font": 12},
                {"kind": "box", "x": 3.72, "y": 3.18, "w": 2.55, "h": 1.5, "text": "实验对照\n证明神经网络策略能学\n但训练成本更高", "fill": "FFFFFF", "outline": THEME["green"], "font": 12},
                {"kind": "box", "x": 6.69, "y": 3.18, "w": 2.55, "h": 1.5, "text": "多轮任务\n检索 / 读文件 / 压缩\n链路更适合 PPO", "fill": "FFFFFF", "outline": THEME["purple"], "font": 12},
                {"kind": "box", "x": 9.66, "y": 3.18, "w": 2.55, "h": 1.5, "text": "回答质量\n偏好证据引用\n少胡编 / 结构清楚", "fill": "FFFFFF", "outline": THEME["pink"], "font": 12},
                {"kind": "text", "x": 1.0, "y": 5.42, "w": 11.0, "h": 0.72, "text": "答辩口径：LinUCB 解决“怎么搜”；PPO 解决“工具链怎么走”；\nDPO/ORPO 解决“怎么答得更可信”。", "font": 15, "bold": True, "color": THEME["line"], "align": "c"},
            ],
        },
        {
            "title": "Source Hit 和 Topic Hit 分别代表什么",
            "subtitle": "这两个指标评估的是检索质量，不是最终文本漂亮程度；它们用来证明 RL 检索策略是否真的找对证据。",
            "tag": "指标解释",
            "objects": [
                {"kind": "box", "x": 0.9, "y": 1.62, "w": 5.55, "h": 2.3, "text": "Source Hit\n命中正确论文 / 正确来源\n\n标准答案需要论文 A\nTop-K 结果出现论文 A\n则 Source Hit = 1", "fill": THEME["green2"], "outline": THEME["green"], "font": 14, "bold": True},
                {"kind": "box", "x": 6.9, "y": 1.62, "w": 5.55, "h": 2.3, "text": "Topic Hit\n命中正确研究主题 / 方向\n\n问题问 RL 轨迹压缩\n结果包含轨迹优化 / 路径规划\n则 Topic Hit = 1", "fill": THEME["pink2"], "outline": THEME["pink"], "font": 14, "bold": True},
                {"kind": "box", "x": 1.15, "y": 4.52, "w": 4.9, "h": 0.72, "text": "Source Hit = 正确来源命中数 / 总问题数", "fill": "FFFFFF", "outline": THEME["green"], "font": 12, "bold": True},
                {"kind": "box", "x": 7.15, "y": 4.52, "w": 4.9, "h": 0.72, "text": "Topic Hit = 正确主题命中数 / 总问题数", "fill": "FFFFFF", "outline": THEME["pink"], "font": 12, "bold": True},
                {"kind": "text", "x": 1.0, "y": 5.72, "w": 11.35, "h": 0.62, "text": "一句话区分：Source Hit 看“是不是那篇/那个来源”；\nTopic Hit 看“方向是不是对”。", "font": 15, "bold": True, "color": THEME["line"], "align": "c"},
            ],
        },
        {
            "title": "训练与评估闭环：为什么能证明 RL 有收益",
            "subtitle": "不是直接说 RL 更好，而是让每个策略在同一批问题、同一知识库、同一指标下对比。",
            "tag": "训练闭环",
            "objects": [
                {"kind": "box", "x": 0.7, "y": 1.55, "w": 1.75, "h": 0.8, "text": "题库\n问题 + 标准主题", "fill": THEME["pink2"], "outline": THEME["pink"], "font": 13, "bold": True},
                {"kind": "line", "x1": 2.45, "y1": 1.95, "x2": 2.88, "y2": 1.95},
                {"kind": "box", "x": 2.88, "y": 1.55, "w": 1.75, "h": 0.8, "text": "论文库\nPDF 切块 + 索引", "fill": THEME["green2"], "outline": THEME["green"], "font": 13, "bold": True},
                {"kind": "line", "x1": 4.63, "y1": 1.95, "x2": 5.06, "y2": 1.95},
                {"kind": "box", "x": 5.06, "y": 1.55, "w": 1.9, "h": 0.8, "text": "动作枚举\nretrieval actions", "fill": "FFFFFF", "outline": THEME["green"], "font": 13, "bold": True},
                {"kind": "line", "x1": 6.96, "y1": 1.95, "x2": 7.39, "y2": 1.95},
                {"kind": "box", "x": 7.39, "y": 1.55, "w": 1.9, "h": 0.8, "text": "离线打分\nSource / Topic Hit", "fill": "FFFFFF", "outline": THEME["pink"], "font": 13, "bold": True},
                {"kind": "line", "x1": 9.29, "y1": 1.95, "x2": 9.72, "y2": 1.95},
                {"kind": "box", "x": 9.72, "y": 1.55, "w": 2.25, "h": 0.8, "text": "训练策略\nLinUCB / DDQN / PPO", "fill": THEME["green"], "outline": THEME["line"], "font": 13, "color": "FFFFFF", "bold": True},
                {"kind": "box", "x": 0.95, "y": 3.25, "w": 3.45, "h": 1.45, "text": "Baseline\n固定检索策略\n作为对照线", "fill": "FFFFFF", "outline": THEME["muted"], "font": 15, "bold": True},
                {"kind": "box", "x": 4.95, "y": 3.25, "w": 3.45, "h": 1.45, "text": "RL Policy\n根据 query 选动作\n学习哪种策略更容易命中", "fill": THEME["green2"], "outline": THEME["green"], "font": 15, "bold": True},
                {"kind": "box", "x": 8.95, "y": 3.25, "w": 3.45, "h": 1.45, "text": "Evaluation\n同一批问题复测\n画 Source / Topic Hit 对比", "fill": THEME["pink2"], "outline": THEME["pink"], "font": 15, "bold": True},
                {"kind": "line", "x1": 4.4, "y1": 3.98, "x2": 4.9, "y2": 3.98},
                {"kind": "line", "x1": 8.4, "y1": 3.98, "x2": 8.9, "y2": 3.98},
                {"kind": "text", "x": 1.0, "y": 5.65, "w": 11.2, "h": 0.5, "text": "判定逻辑：如果 LinUCB 在同一题库上比 baseline 更高，就说明策略学习确实提升了检索命中。", "font": 17, "bold": True, "color": THEME["line"], "align": "c"},
            ],
        },
        {
            "title": "最终答辩口径：不是放弃 PPO/DPO/ORPO",
            "subtitle": "准确说法是：LinUCB 作为上线主策略；PPO/DPO/ORPO 作为更高层的强化学习增强和实验对照。",
            "tag": "答辩总结",
            "objects": [
                {"kind": "box", "x": 0.9, "y": 1.55, "w": 3.4, "h": 1.2, "text": "上线推荐\nQwen + RAG + LinUCB", "fill": THEME["green"], "outline": THEME["line"], "font": 19, "color": "FFFFFF", "bold": True},
                {"kind": "box", "x": 4.95, "y": 1.55, "w": 3.4, "h": 1.2, "text": "神经网络对照\nDDQN / Dueling DDQN", "fill": THEME["green2"], "outline": THEME["green"], "font": 17, "bold": True},
                {"kind": "box", "x": 9.0, "y": 1.55, "w": 3.4, "h": 1.2, "text": "偏好与行为增强\nPPO / DPO / ORPO", "fill": THEME["pink2"], "outline": THEME["pink"], "font": 17, "bold": True},
                {"kind": "box", "x": 0.9, "y": 3.15, "w": 3.4, "h": 1.75, "text": "为什么上线用 LinUCB\n单步动作选择最匹配\n训练快、稳定、可解释", "fill": "FFFFFF", "outline": THEME["green"], "font": 14, "align": "l"},
                {"kind": "box", "x": 4.95, "y": 3.15, "w": 3.4, "h": 1.75, "text": "为什么保留 DDQN\n动作离散，也适合学习\n但需要更多样本和调参", "fill": "FFFFFF", "outline": THEME["green"], "font": 14, "align": "l"},
                {"kind": "box", "x": 9.0, "y": 3.15, "w": 3.4, "h": 1.75, "text": "为什么保留 PPO/DPO/ORPO\nPPO 优化多步工具链\nDPO/ORPO 优化回答偏好", "fill": "FFFFFF", "outline": THEME["pink"], "font": 14, "align": "l"},
                {"kind": "text", "x": 0.95, "y": 5.55, "w": 11.5, "h": 0.66, "text": "一句话：LinUCB 让 Agent 找对资料；PPO/DPO/ORPO 让 Agent 更会行动、更会基于证据回答。", "font": 19, "bold": True, "color": THEME["line"], "align": "c"},
            ],
        },
    ]


def parse_existing_titles(z: zipfile.ZipFile) -> set[str]:
    ns = {"p": P_NS, "a": A_NS, "r": R_NS}
    pres = ET.fromstring(z.read("ppt/presentation.xml"))
    rels = ET.fromstring(z.read("ppt/_rels/presentation.xml.rels"))
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    titles = set()
    for sid in pres.find("p:sldIdLst", ns).findall("p:sldId", ns):
        rid = sid.attrib.get(f"{{{R_NS}}}id")
        target = relmap.get(rid, "")
        path = "ppt/" + target if not target.startswith("/") else target[1:]
        if path in z.namelist():
            root = ET.fromstring(z.read(path))
            texts = [t.text.strip() for t in root.findall(".//a:t", ns) if t.text and t.text.strip()]
            if texts:
                titles.add(texts[0])
    return titles


def slide_title_map(entries: dict[str, bytes]) -> dict[str, int]:
    ns = {"p": P_NS, "a": A_NS, "r": R_NS}
    pres = ET.fromstring(entries["ppt/presentation.xml"])
    rels = ET.fromstring(entries["ppt/_rels/presentation.xml.rels"])
    relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    mapping: dict[str, int] = {}
    for sid in pres.find("p:sldIdLst", ns).findall("p:sldId", ns):
        rid = sid.attrib.get(f"{{{R_NS}}}id")
        target = relmap.get(rid, "")
        match = re.fullmatch(r"slides/slide(\d+)\.xml", target)
        if not match:
            continue
        path = "ppt/" + target
        if path not in entries:
            continue
        root = ET.fromstring(entries[path])
        texts = [t.text.strip() for t in root.findall(".//a:t", ns) if t.text and t.text.strip()]
        if texts:
            mapping[texts[0]] = int(match.group(1))
    return mapping


def modify_pptx() -> tuple[Path, Path, list[Path]]:
    if not PPTX_PATH.exists():
        raise FileNotFoundError(PPTX_PATH)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup = PPTX_PATH.with_name(f"{PPTX_PATH.stem}_backup_before_linucb_addendum_{timestamp}{PPTX_PATH.suffix}")
    shutil.copy2(PPTX_PATH, backup)

    specs = get_specs()
    with zipfile.ZipFile(PPTX_PATH, "r") as zin:
        existing_titles = parse_existing_titles(zin)
        entries = {name: zin.read(name) for name in zin.namelist()}

    pres = ET.fromstring(entries["ppt/presentation.xml"])
    pres_rels = ET.fromstring(entries["ppt/_rels/presentation.xml.rels"])
    content_types = ET.fromstring(entries["[Content_Types].xml"])
    sld_id_lst = pres.find(f"{{{P_NS}}}sldIdLst")
    existing_slide_ids = sld_id_lst.findall(f"{{{P_NS}}}sldId")
    slide_nums = []
    for name in entries:
        match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
        if match:
            slide_nums.append(int(match.group(1)))
    next_slide_num = max(slide_nums) + 1
    max_slide_id = max(int(el.attrib["id"]) for el in existing_slide_ids)
    rid_nums = [int(rel.attrib["Id"][3:]) for rel in pres_rels if rel.attrib.get("Id", "").startswith("rId") and rel.attrib["Id"][3:].isdigit()]
    next_rid = max(rid_nums) + 1

    layout_target = "../slideLayouts/slideLayout1.xml"
    rel57 = entries.get("ppt/slides/_rels/slide57.xml.rels")
    if rel57:
        rel_root = ET.fromstring(rel57)
        for rel in rel_root:
            if rel.attrib.get("Type") == f"{R_NS}/slideLayout":
                layout_target = rel.attrib.get("Target", layout_target)
                break

    if specs[0]["title"] in existing_titles:
        mapping = slide_title_map(entries)
        ordered_nums = []
        ns = {"p": P_NS, "r": R_NS}
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in pres_rels}
        for sid_el in pres.find("p:sldIdLst", ns).findall("p:sldId", ns):
            target = relmap.get(sid_el.attrib.get(f"{{{R_NS}}}id"), "")
            match = re.fullmatch(r"slides/slide(\d+)\.xml", target)
            if match:
                ordered_nums.append(int(match.group(1)))
        fallback_nums = ordered_nums[-len(specs):]
        preview_paths = []
        for idx, spec in enumerate(specs):
            slide_num = mapping.get(spec["title"]) or (fallback_nums[idx] if len(fallback_nums) == len(specs) else None)
            if not slide_num:
                raise RuntimeError(f"Cannot find existing slide for title: {spec['title']}")
            entries[f"ppt/slides/slide{slide_num}.xml"] = build_slide(spec)
            rel_name = f"ppt/slides/_rels/slide{slide_num}.xml.rels"
            entries[rel_name] = slide_relationship_xml(layout_target)
            preview = PREVIEW_DIR / f"slide{slide_num}_{safe_name(spec['title'])}.svg"
            render_preview(spec, preview)
            preview_paths.append(preview)
        temp = PPTX_PATH.with_suffix(".tmp.pptx")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, data in entries.items():
                zout.writestr(name, data)
        with zipfile.ZipFile(temp, "r") as zcheck:
            bad = zcheck.testzip()
            if bad:
                raise RuntimeError(f"Corrupt pptx entry: {bad}")
        temp.replace(PPTX_PATH)
        return PPTX_PATH, backup, preview_paths

    preview_paths = []
    for i, spec in enumerate(specs):
        slide_num = next_slide_num + i
        rid = f"rId{next_rid + i}"
        slide_name = f"ppt/slides/slide{slide_num}.xml"
        entries[slide_name] = build_slide(spec)
        entries[f"ppt/slides/_rels/slide{slide_num}.xml.rels"] = slide_relationship_xml(layout_target)

        ET.SubElement(pres_rels, "Relationship", {
            "Id": rid,
            "Type": f"{R_NS}/slide",
            "Target": f"slides/slide{slide_num}.xml",
        })
        sld = ET.SubElement(sld_id_lst, f"{{{P_NS}}}sldId", {"id": str(max_slide_id + i + 1)})
        sld.set(f"{{{R_NS}}}id", rid)
        ET.SubElement(content_types, f"{{{CT_NS}}}Override", {
            "PartName": f"/ppt/slides/slide{slide_num}.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
        })
        preview = PREVIEW_DIR / f"slide{slide_num}_{safe_name(spec['title'])}.svg"
        render_preview(spec, preview)
        preview_paths.append(preview)

    entries["ppt/presentation.xml"] = ET.tostring(pres, encoding="utf-8", xml_declaration=True)
    entries["ppt/_rels/presentation.xml.rels"] = ET.tostring(pres_rels, encoding="utf-8", xml_declaration=True)
    entries["[Content_Types].xml"] = ET.tostring(content_types, encoding="utf-8", xml_declaration=True)
    if "docProps/app.xml" in entries:
        app = ET.fromstring(entries["docProps/app.xml"])
        slides_el = app.find(f"{{{APP_NS}}}Slides")
        if slides_el is not None:
            slides_el.text = str(len(existing_slide_ids) + len(specs))
            entries["docProps/app.xml"] = ET.tostring(app, encoding="utf-8", xml_declaration=True)

    temp = PPTX_PATH.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in entries.items():
            zout.writestr(name, data)
    with zipfile.ZipFile(temp, "r") as zcheck:
        bad = zcheck.testzip()
        if bad:
            raise RuntimeError(f"Corrupt pptx entry: {bad}")
    temp.replace(PPTX_PATH)
    return PPTX_PATH, backup, preview_paths


def safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isascii() and ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "_", "-"}:
            keep.append("_")
    name = "".join(keep).strip("_")
    return name[:32] or "preview"


def pi(v: float) -> int:
    return int(round(v * PX_PER_IN))


def wrap_text_plain(text: str, max_chars: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for ch in paragraph:
            test = current + ch
            if len(test) <= max_chars or not current:
                current = test
            else:
                lines.append(current)
                current = ch
        lines.append(current)
    return lines


def svg_text(x: int, y: int, w: int, h: int, text: str, size: int, color: str, bold: bool = False, align: str = "middle") -> str:
    max_chars = max(4, int(w / max(size * 0.58, 1)))
    lines = wrap_text_plain(text, max_chars)
    line_h = int(size * 1.25)
    total_h = line_h * len(lines)
    cy = y + max(line_h, (h - total_h) // 2 + line_h)
    anchor = "middle" if align in {"ctr", "c", "center", "middle"} else ("end" if align in {"r", "right"} else "start")
    tx = x + w // 2 if anchor == "middle" else (x + w - 16 if anchor == "end" else x + 16)
    weight = "700" if bold else "400"
    out = [f'<text x="{tx}" y="{cy}" fill="#{rgb(color)}" font-family="Microsoft YaHei, Segoe UI, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">']
    for line in lines:
        out.append(f'<tspan x="{tx}" dy="0">{escape(line)}</tspan>')
        out.append(f'<tspan x="{tx}" dy="{line_h}"></tspan>')
    out.append("</text>")
    return "".join(out)


def render_preview(spec: dict, path: Path) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">',
        f'<rect width="1920" height="1080" fill="#{rgb(spec.get("bg", THEME["bg"]))}"/>',
        f'<rect x="0" y="0" width="1920" height="{pi(0.16)}" fill="#{rgb(THEME["green"])}"/>',
        f'<rect x="0" y="{pi(0.16)}" width="1920" height="{pi(0.07)}" fill="#{rgb(THEME["pink"])}"/>',
        svg_text(pi(0.62), pi(0.34), pi(9.3), pi(0.62), spec["title"], 40, THEME["ink"], True, "left"),
        svg_text(pi(0.64), pi(0.94), pi(9.6), pi(0.42), spec.get("subtitle", ""), 18, THEME["muted"], False, "left"),
        f'<rect x="{pi(10.35)}" y="{pi(0.48)}" width="{pi(2.25)}" height="{pi(0.38)}" rx="18" fill="#{rgb(THEME["green2"])}" stroke="#{rgb(THEME["green"])}" stroke-width="2"/>',
        svg_text(pi(10.35), pi(0.48), pi(2.25), pi(0.38), spec.get("tag", "RL-RAG 答辩页"), 16, THEME["line"], True),
    ]
    for obj in spec["objects"]:
        if obj["kind"] == "box":
            x, y, w, h = pi(obj["x"]), pi(obj["y"]), pi(obj["w"]), pi(obj["h"])
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#{rgb(obj.get("fill", THEME["white"]))}" stroke="#{rgb(obj.get("outline", THEME["green"]))}" stroke-width="3"/>')
            parts.append(svg_text(x, y, w, h, obj.get("text", ""), int(obj.get("font", 15) * 1.45), obj.get("color", THEME["ink"]), obj.get("bold", False), obj.get("align", "ctr")))
        elif obj["kind"] == "text":
            x, y, w, h = pi(obj["x"]), pi(obj["y"]), pi(obj["w"]), pi(obj["h"])
            parts.append(svg_text(x, y, w, h, obj.get("text", ""), int(obj.get("font", 16) * 1.45), obj.get("color", THEME["ink"]), obj.get("bold", False), obj.get("align", "l")))
        elif obj["kind"] == "line":
            parts.append(f'<line x1="{pi(obj["x1"])}" y1="{pi(obj["y1"])}" x2="{pi(obj["x2"])}" y2="{pi(obj["y2"])}" stroke="#{rgb(obj.get("color", THEME["line"]))}" stroke-width="4" stroke-linecap="round"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    deck, backup, previews = modify_pptx()
    print(f"UPDATED={deck}")
    print(f"BACKUP={backup}")
    print("PREVIEWS=")
    for preview in previews:
        print(preview)
