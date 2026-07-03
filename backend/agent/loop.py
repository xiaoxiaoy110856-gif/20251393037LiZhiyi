from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from backend.llm.service import run_messages
from backend.settings import get_agent_max_turns, get_top_k
from backend.agent.registry import TOOL_REGISTRY, execute_tool, tool_catalog_text
from backend.agent.tools import render_tool_result

# Agent loop contract:
# - normal tasks go through a small JSON tool-calling loop;
# - clear image-generation intents bypass the LLM loop and call
#   generate_image_advanced directly, so the user always gets an actual image
#   attempt instead of a textual explanation.


AGENT_SYSTEM_PROMPT = """You are a local AI agent for trajectory and reinforcement learning research.

Your job is to help with RL and trajectory work by:
1. understanding the user task,
2. deciding when local tools can improve grounding,
3. answering clearly and directly,
4. citing useful evidence when available,
5. avoiding fabricated claims.

You must follow this response protocol:
- If you need a tool next, output ONLY a JSON object like:
  {"type":"tool_use","id":"tool_1","name":"search_project_docs","input":{"query":"...","top_k":4}}
- If you already have enough information, output ONLY a JSON object like:
  {"type":"final","answer":"..."}

Available tools:
1. list_files: inspect workspace directory structure without reading file content.
2. read_file: read a workspace text file or a line range.
3. search_text: search workspace text by keyword, function name, error, or config value.
4. get_file_metadata: inspect file metadata.
5. build_file_index: build a project-level file manifest.
6. search_project_docs: search the RL/trajectory knowledge base.
7. summarize_experiment_text: summarize logs or experiment snippets.
8. generate_image_advanced: generate a higher-quality image from text using prompt rewriting, negative prompt, batch candidates, scoring, and retry.
9. generate_image: legacy simple image generation fallback.

Rules:
- Reuse the exact tool id when reasoning about a tool result.
- Use at most one tool per turn.
- If tool results are available, prefer grounded answers over generic ones.
- If evidence is weak, say so plainly in the final answer.
- When writing RL transition chains, use display LaTeX such as \\[s_t \\xrightarrow{\\;\\pi_\\theta\\;} a_t \\xrightarrow{\\;\\mathcal{E}\\;} (r_{t+1}, s_{t+1})\\], not plain arrow text or code fences.
- Do not assume file contents. Use tools to inspect files before making claims.
- For folder/project analysis, call list_files or build_file_index first.
- For locating a feature, call search_text first, then read_file only around relevant lines.
- Do not read many files at once. For large files, read only relevant line ranges.
- Cite workspace paths and line numbers in final answers whenever possible.
- Never access paths outside the configured workspace root.
- Do not read .env files, keys, certs, tokens, private keys, or other sensitive files unless the user explicitly confirms the risk.
- If search_text returns too many results, narrow the search before reading files.
- When the user clearly asks to generate an image, draw, make a poster/icon/logo draft/product mockup, or visualize a scene, call generate_image_advanced instead of only describing it.
- Do not pass the raw user prompt directly to ComfyUI. Use a structured image plan with subject count, scene, composition, camera, lighting, physical constraints, quality constraints, and negative prompt.
- For fragile subjects such as cars, hands, mechanical objects, and architecture, add structure constraints and negative prompt terms.
- If the user is only discussing how image generation works or how to write prompts, do not call generate_image.
- If the user asks to edit an existing image, explain that the current version only supports text-to-image unless a future edit_image provider is configured.
- If the user asks to generate an image of themself and no reference image is present, ask for a reference image first.
- For image generation prompts, make the prompt concrete: subject, scene, style, composition, light, color, aspect, and details.
- After image generation, return the image and a short note, not a long process explanation.
"""


@dataclass
class ToolTrace:
    id: str
    name: str
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


TOOL_DEFINITIONS = TOOL_REGISTRY


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON 工具调用对象；模型输出不规范时返回 None。"""
    candidate = (text or "").strip()
    if not candidate:
        return None
    for blob in [candidate, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.S)]:
        blob = blob.strip()
        if not blob.startswith("{"):
            continue
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _build_context(traces: list[ToolTrace]) -> str:
    """把已经执行过的工具结果整理成上下文，供 Qwen 决定下一步或生成最终回答。"""
    if not traces:
        return ""
    blocks = []
    for index, trace in enumerate(traces, start=1):
        blocks.append(
            f"[Tool {index}: {trace.name} | id={trace.id}]\n"
            f"input={json.dumps(trace.input, ensure_ascii=False)}\n"
            f"{render_tool_result(trace.output)}"
        )
    return "\n\n".join(blocks)


def _tool_catalog_block(top_k: int) -> str:
    """生成工具说明文本，让模型知道当前 Agent 可以调用哪些工具。"""
    return f"{tool_catalog_text(top_k)}\nYou may call tools repeatedly until you can answer."


def _parse_model_action(text: str) -> tuple[str, dict[str, Any] | None]:
    """解析模型动作：如果是 tool_call 就执行工具，如果是 final 就返回最终答案。"""
    payload = _extract_json_object(text)
    if not payload:
        return "final", {"answer": text.strip()}
    action_type = str(payload.get("type", "")).strip().lower()
    if action_type == "tool_use":
        return "tool_use", payload
    if action_type == "final":
        return "final", payload
    return "final", {"answer": text.strip()}


IMAGE_ACTION_RE = re.compile(
    r"(生成|画|绘制|做一张|制作|设计|海报|图标|logo|产品图|场景图|视觉化|render|draw|generate|create|poster|icon|logo|mockup)",
    re.IGNORECASE,
)
IMAGE_DISCUSSION_RE = re.compile(r"(怎么|如何|方法|教程|原理|prompt|提示词|为什么|what|how|guide)", re.IGNORECASE)
IMAGE_EDIT_RE = re.compile(r"(编辑|修改|局部|mask|抠图|换背景|edit|inpaint|outpaint)", re.IGNORECASE)
SELF_IMAGE_RE = re.compile(r"(我本人|我的人像|把我|我这张脸|me\b|my portrait)", re.IGNORECASE)


# 核心3：图片意图识别。命中后直接进入图片工具，避免模型只用文字描述而不真正生成图。
def should_generate_image(query: str) -> bool:
    """True only for creation requests, not prompt-writing discussions."""
    text = query or ""
    return bool(IMAGE_ACTION_RE.search(text)) and not bool(IMAGE_DISCUSSION_RE.search(text))


def is_image_edit_request(query: str) -> bool:
    """判断用户是不是在要求修图/局部编辑；当前没有参考图时会给出提示。"""
    return bool(IMAGE_EDIT_RE.search(query or ""))


def _execute_tool(tool_use: ToolUse, query: str, top_k: int) -> dict[str, Any]:
    """执行模型选中的工具，并补齐检索 query、top_k、待总结文本等默认参数。"""
    tool_input = dict(tool_use.input)
    if tool_use.name == "search_project_docs":
        tool_input["query"] = str(tool_input.get("query", "")).strip() or query
        tool_input["top_k"] = int(tool_input.get("top_k", top_k) or top_k)
    if tool_use.name == "summarize_experiment_text" and not str(tool_input.get("text", "")).strip():
        tool_input["text"] = query
    return execute_tool(tool_use.name, tool_input)


def agent_chat(query: str, history: list[dict[str, str]] | None = None, top_k: int | None = None, model_id: str | None = None) -> dict[str, Any]:
    """Agent 主循环：识别图片任务、构造工具目录、调用 Qwen 决策工具、整合工具结果并输出答案。"""
    history = history or []
    effective_top_k = top_k or get_top_k()
    tool_traces: list[ToolTrace] = []

    if should_generate_image(query):
        # 核心3：明确画图请求直接走高级图片生成链路，不再让模型反复判断是否调用工具。
        # Image generation is deterministic at the orchestration layer: the
        # Agent does not need to ask the LLM whether to call the tool.
        if is_image_edit_request(query):
            return {
                "answer": "当前版本只支持文生图，图片编辑和 mask 局部修改接口已经预留，但还没有启用。",
                "sources": [],
                "tool_traces": [],
                "context_preview": "",
            }
        if SELF_IMAGE_RE.search(query):
            return {
                "answer": "如果要基于你本人的人像生成图片，请先上传参考图。当前文生图不会凭空生成“你本人”。",
                "sources": [],
                "tool_traces": [],
                "context_preview": "",
            }
        tool_input = {
            "prompt": query,
            "quality_mode": "high",
            "batch_size": 1,
            "allow_retry": True,
            "use_highres_fix": True,
        }
        tool_output = execute_tool("generate_image_advanced", tool_input)
        trace = ToolTrace(id="tool_1", name="generate_image_advanced", input=tool_input, output=tool_output)
        tool_traces.append(trace)
        images = tool_output.get("images", []) if isinstance(tool_output, dict) else []
        if tool_output.get("error"):
            answer = f"图片生成失败：{tool_output['error']}"
        elif images:
            lines = ["已生成图片。"]
            for image in images:
                url = image.get("url", "")
                if url:
                    lines.append(f"![generated image]({url})")
                    lines.append(f"图片链接：{url}")
            answer = "\n\n".join(lines)
        else:
            answer = "图片生成失败：provider 没有返回图片。"
        return {
            "answer": answer,
            "sources": [],
            "tool_traces": [{"id": trace.id, "name": trace.name, "input": trace.input, "output": trace.output}],
            "context_preview": _build_context(tool_traces)[:3000],
        }

    runtime_messages: list[dict[str, str]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    runtime_messages.extend(history[-8:])
    runtime_messages.append({"role": "system", "content": _tool_catalog_block(effective_top_k)})
    runtime_messages.append({"role": "user", "content": query})

    final_answer = ""
    max_turns = max(1, get_agent_max_turns())
    for turn_index in range(max_turns):
        tool_context = _build_context(tool_traces)
        if tool_context:
            prompt_messages = runtime_messages + [
                {
                    "role": "system",
                    "content": f"Tool observations so far:\n\n{tool_context}",
                }
            ]
        else:
            prompt_messages = runtime_messages

        # 核心1：工具观察结果组装完成后，Qwen/Ollama 会继续输出下一次工具调用 JSON 或最终回答。
        raw_response = run_messages(prompt_messages, query=query, context_block=tool_context, model_id=model_id)
        action_type, payload = _parse_model_action(raw_response)

        if action_type == "final":
            final_answer = str((payload or {}).get("answer", "")).strip() or raw_response.strip()
            runtime_messages.append({"role": "assistant", "content": json.dumps(payload or {"type": "final", "answer": final_answer}, ensure_ascii=False)})
            break

        payload = payload or {}
        tool_id = str(payload.get("id", f"tool_{turn_index + 1}")).strip() or f"tool_{turn_index + 1}"
        tool_name = str(payload.get("name", "")).strip()
        tool_input = payload.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}

        tool_use = ToolUse(id=tool_id, name=tool_name, input=tool_input)
        runtime_messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"type": "tool_use", "id": tool_use.id, "name": tool_use.name, "input": tool_use.input},
                    ensure_ascii=False,
                ),
            }
        )
        tool_output = _execute_tool(tool_use, query=query, top_k=effective_top_k)
        trace = ToolTrace(id=tool_use.id, name=tool_use.name, input=tool_use.input, output=tool_output)
        tool_traces.append(trace)
        runtime_messages.append(
            {
                "role": "tool",
                "content": json.dumps(
                    {"type": "tool_result", "id": tool_use.id, "name": tool_use.name, "output": tool_output},
                    ensure_ascii=False,
                ),
            }
        )
    else:
        final_answer = "我已经尝试了多轮工具调用，但这一轮还没有稳定收束。我先把当前证据整理给你。"

    if not final_answer:
        final_answer = "我还没有得到稳定的最终回答，但已经整理了可用证据。"

    tool_context = _build_context(tool_traces)

    sources: list[dict[str, Any]] = []
    for trace in tool_traces:
        if trace.name == "search_project_docs":
            sources.extend(trace.output.get("results", []))

    return {
        "answer": final_answer,
        "sources": sources,
        "tool_traces": [
            {
                "id": trace.id,
                "name": trace.name,
                "input": trace.input,
                "output": trace.output,
            }
            for trace in tool_traces
        ],
        "context_preview": tool_context[:3000],
    }
