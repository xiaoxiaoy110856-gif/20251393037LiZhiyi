from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.settings import rag_enabled
from backend.image.service import generate_image
from backend.image.quality import generate_image_advanced
from backend.agent.tools import read_local_file, search_project_docs, summarize_experiment_text
from backend.workspace.tools import build_file_index, get_file_metadata, list_files, read_file, search_text


ToolCallable = Callable[..., dict[str, Any]]

# Central registry for every callable Agent tool. Keeping registration here
# avoids scattering tool names across app.py, agent_loop.py, and feature modules.


@dataclass(frozen=True)
class ToolSpec:
    """Agent 工具描述：包含工具名、自然语言说明、输入 schema 和真正执行函数。"""
    name: str
    description: str
    input_schema: dict[str, str]
    handler: ToolCallable


def _search_project_docs(query: str = "", top_k: int = 4, **_: Any) -> dict[str, Any]:
    """工具包装：搜索项目知识库，内部会经过 RL 检索策略选择。"""
    if not rag_enabled():
        return {"tool": "search_project_docs", "query": query, "count": 0, "results": [], "warning": "RAG disabled"}
    return search_project_docs(query, top_k=top_k)


def _read_local_file(path: str = "", **_: Any) -> dict[str, Any]:
    """工具包装：兼容旧提示词的本地文件读取接口。"""
    return read_local_file(path)


def _summarize_experiment_text(text: str = "", **_: Any) -> dict[str, Any]:
    """工具包装：总结实验日志、训练结果或评测文本。"""
    return summarize_experiment_text(text)


def _list_files(**kwargs: Any) -> dict[str, Any]:
    """工具包装：列出工作区文件结构，不直接读取文件正文。"""
    return list_files(**kwargs)


def _read_file(**kwargs: Any) -> dict[str, Any]:
    """工具包装：读取工作区文件，可按行号和最大字节数限制输出。"""
    return read_file(**kwargs)


def _search_text(**kwargs: Any) -> dict[str, Any]:
    """工具包装：在工作区内搜索关键词、函数名、错误信息或配置项。"""
    return search_text(**kwargs)


def _get_file_metadata(path: str = ".", **_: Any) -> dict[str, Any]:
    """工具包装：读取文件大小、类型、修改时间等元信息。"""
    return get_file_metadata(path)


def _build_file_index(**kwargs: Any) -> dict[str, Any]:
    """工具包装：生成工作区文件清单和语言统计。"""
    return build_file_index(**kwargs)


def _generate_image(**kwargs: Any) -> dict[str, Any]:
    """工具包装：调用简单图片生成路径。"""
    return generate_image(**kwargs)


def _generate_image_advanced(**kwargs: Any) -> dict[str, Any]:
    """工具包装：调用高级图片生成质量控制链路。"""
    return generate_image_advanced(**kwargs)


# 核心工具表：Agent 能调用的能力都注册在这里；agent_loop.py 只会执行本字典里的工具。
TOOL_REGISTRY: dict[str, ToolSpec] = {
    "search_project_docs": ToolSpec(
        name="search_project_docs",
        description="Search the local RL/trajectory knowledge base, not the workspace source tree.",
        input_schema={"query": "string", "top_k": "integer"},
        handler=_search_project_docs,
    ),
    "read_local_file": ToolSpec(
        name="read_local_file",
        description="Legacy local file reader for old prompts. Prefer read_file for workspace files.",
        input_schema={"path": "string"},
        handler=_read_local_file,
    ),
    "summarize_experiment_text": ToolSpec(
        name="summarize_experiment_text",
        description="Summarize logs, experiment snippets, or result descriptions.",
        input_schema={"text": "string"},
        handler=_summarize_experiment_text,
    ),
    "list_files": ToolSpec(
        name="list_files",
        description="List workspace directory structure and metadata without reading file content.",
        input_schema={"path": "string", "max_depth": "integer", "include_glob": "string", "exclude_glob": "string", "include_hidden": "boolean"},
        handler=_list_files,
    ),
    # 核心4/文件：Agent 分析本地文件时使用的工作区文件读取工具。
    "read_file": ToolSpec(
        name="read_file",
        description="Read a workspace text file, optionally with start_line/end_line and max_bytes. Returns line-numbered content.",
        input_schema={"path": "string", "start_line": "integer", "end_line": "integer", "max_bytes": "integer"},
        handler=_read_file,
    ),
    # 核心4/文件：用于定位代码、日志、配置项的全文搜索工具。
    "search_text": ToolSpec(
        name="search_text",
        description="Search workspace text files for keywords, function names, errors, and config values. Returns paths and line numbers.",
        input_schema={"query": "string", "path": "string", "glob": "string", "regex": "boolean", "case_sensitive": "boolean", "max_results": "integer"},
        handler=_search_text,
    ),
    "get_file_metadata": ToolSpec(
        name="get_file_metadata",
        description="Return workspace file metadata such as size, type, modified time, extension, binary/sensitive flags.",
        input_schema={"path": "string"},
        handler=_get_file_metadata,
    ),
    "build_file_index": ToolSpec(
        name="build_file_index",
        description="Build a workspace file manifest with language, size, line counts, hash, and language stats.",
        input_schema={"path": "string", "include_hidden": "boolean", "limit": "integer"},
        handler=_build_file_index,
    ),
    "generate_image": ToolSpec(
        name="generate_image",
        description=(
            "Use when the user asks to create, generate, draw, design, visualize, render, make an illustration, "
            "poster, icon, logo draft, product mockup, or image from text. Do not use for ordinary text answers."
        ),
        input_schema={
            "prompt": "string",
            "size": "string",
            "quality": "string",
            "format": "string",
            "background": "string",
            "n": "integer",
            "style_notes": "string",
            "user_visible_prompt": "string",
        },
        handler=_generate_image,
    ),
    # 核心3：聊天中生成图片请求使用的主工具。
    "generate_image_advanced": ToolSpec(
        name="generate_image_advanced",
        description=(
            "Use when the user asks to generate, create, draw, render, design, visualize, or produce an image. "
            "This tool rewrites prompts, adds negative prompts, generates multiple ComfyUI candidates, evaluates quality, "
            "retries if needed, and returns the best image."
        ),
        input_schema={
            "prompt": "string",
            "style": "string",
            "preset": "string",
            "size": "string",
            "batch_size": "integer",
            "quality_mode": "fast | balanced | high",
            "allow_retry": "boolean",
            "use_highres_fix": "boolean",
            "reference_image_id": "string",
            "notes": "string",
        },
        handler=_generate_image_advanced,
    ),
}


def tool_catalog_text(top_k: int) -> str:
    """把工具注册表转换成模型可读的工具说明文本。"""
    lines = ["Available tool definitions:"]
    for spec in TOOL_REGISTRY.values():
        schema = ", ".join(f"{key}: {value}" for key, value in spec.input_schema.items())
        lines.append(f"- {spec.name}({schema}): {spec.description}")
    lines.append(f"Default top_k for search_project_docs is {top_k}.")
    return "\n".join(lines)


def execute_tool(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    # 核心工具入口：所有 Agent 工具调用都经过这里，便于统一处理工具名、输入校验和异常。
    spec = TOOL_REGISTRY.get(name)
    if not spec:
        return {"tool": name, "error": f"Unsupported tool: {name}"}
    try:
        result = spec.handler(**tool_input)
    except TypeError as error:
        return {"tool": name, "error": f"Invalid tool input: {error}", "input": tool_input}
    except Exception as error:
        return {"tool": name, "error": str(error), "input": tool_input}
    result.setdefault("tool", name)
    return result
