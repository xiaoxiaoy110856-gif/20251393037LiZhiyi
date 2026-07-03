from __future__ import annotations

import json
import re
import subprocess
import difflib
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from typing import Any
from datetime import datetime

from backend.llm.service import chat_reply
from backend.retrieval.knowledge_store import get_cached_kb, search_knowledge
from backend.retrieval.policy import choose_retrieval_action
from backend.settings import ROOT, get_file_backups_dir, get_file_write_root, get_raw_kb_dir, get_repos_dir, rag_enabled


WORKSPACE_ROOT = ROOT.resolve()
ALLOWED_READ_ROOTS = {
    WORKSPACE_ROOT,
    get_raw_kb_dir().resolve(),
    get_repos_dir().resolve(),
    Path.home().resolve(),
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".log",
    ".sh",
    ".ps1",
    ".vue",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
    ".scss",
    ".html",
}

READABLE_BINARY_SUFFIXES = {
    ".pdf",
}


def _safe_path(path_text: str) -> Path:
    candidate = Path(path_text.strip().strip("\"'"))
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not any(root == candidate or root in candidate.parents for root in ALLOWED_READ_ROOTS):
        raise ValueError(f"Path is outside the allowed workspace: {candidate}")
    return candidate


def _safe_write_path(path_text: str) -> Path:
    root = get_file_write_root().resolve()
    candidate = Path(path_text.strip().strip("\"'"))
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not (root == candidate or root in candidate.parents):
        raise ValueError(f"Write path is outside LOCAL_FILE_WRITE_ROOT: {candidate}")
    return candidate


def _read_text_file_for_edit(path: Path, max_chars: int) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError(f"Only text files can be edited by this tool. Unsupported suffix: {path.suffix or '<none>'}")
    content = path.read_text(encoding="utf-8", errors="ignore")
    if len(content) > max_chars:
        raise ValueError(f"File is too large for a safe one-pass edit ({len(content)} chars > {max_chars}).")
    return content


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("The model did not return valid JSON for the edit proposal.")


def _unified_diff(old: str, new: str, path: Path) -> str:
    lines = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        fromfile=f"{path.name} (current)",
        tofile=f"{path.name} (proposed)",
        lineterm="",
    )
    return "\n".join(lines)


def _slugify_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "repo"


def extract_candidate_paths(text: str) -> list[str]:
    matches = re.findall(r"([A-Za-z]:\\[^\s\"']+|(?:[\w.\-_/]+/)+[\w.\-_/]+|[\w.\-_/]+\.(?:py|md|txt|json|jsonl|yaml|yml|toml|csv|log))", text)
    seen: list[str] = []
    for item in matches:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen


def search_project_docs(query: str, top_k: int = 4) -> dict[str, Any]:
    # 核心5：Agent 的 RAG 检索入口。先由训练好的策略选择检索动作，再用调整后的 query/top_k 查询知识库。
    policy = choose_retrieval_action(query, requested_top_k=top_k)
    retrieval_query = str(policy.get("retrieval_query") or query)
    retrieval_top_k = int(policy.get("top_k") or top_k)
    if rag_enabled():
        results = search_knowledge(retrieval_query, top_k=retrieval_top_k)
    else:
        results = []
        payload = get_cached_kb()
        terms = [term for term in re.split(r"\s+", retrieval_query.lower()) if term]
        for doc in payload.get("documents", []):
            text = f"{doc.get('title', '')} {doc.get('preview', '')}".lower()
            score = sum(1 for term in terms if term in text)
            if score <= 0:
                continue
            results.append(
                {
                    "title": doc.get("title", ""),
                    "topics": doc.get("topics", []),
                    "path": doc.get("path", ""),
                    "snippet": doc.get("preview", "")[:500],
                    "score": float(score),
                }
            )
        results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        results = results[:top_k]
    return {
        "tool": "search_project_docs",
        "query": query,
        "retrieval_query": retrieval_query,
        "policy": policy,
        "count": len(results),
        "results": results,
    }


def read_local_file(path_text: str, max_chars: int = 6000) -> dict[str, Any]:
    """读取本地工作区文件的旧接口，主要用于兼容早期 Agent 提示词。"""
    path = _safe_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        entries = sorted(item.name for item in path.iterdir())[:80]
        return {
            "tool": "read_local_file",
            "path": str(path),
            "kind": "directory",
            "entries": entries,
        }
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return {
            "tool": "read_local_file",
            "path": str(path),
            "kind": "binary_or_unsupported",
            "detail": f"Suffix {path.suffix or '<none>'} is not treated as a text file.",
        }
    content = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "tool": "read_local_file",
        "path": str(path),
        "kind": "text",
        "content": content[:max_chars],
        "truncated": len(content) > max_chars,
    }


def propose_file_edit(path_text: str, instruction: str, max_chars: int = 24000, model_id: str | None = None) -> dict[str, Any]:
    """根据用户自然语言指令生成修改后的完整文件内容和 diff，但不直接落盘。"""
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("Edit instruction cannot be empty.")

    path = _safe_write_path(path_text)
    content = _read_text_file_for_edit(path, max_chars=max_chars)
    prompt = (
        "You are a careful local code editing assistant. Modify the file according to the user instruction. "
        "Return JSON only, with no Markdown and no explanation. The JSON shape must be "
        '{"summary":"one sentence summary","new_content":"the complete modified file content"}. '
        "Do not omit content and do not return a diff."
    )
    context_block = (
        f"File path: {path}\n"
        f"User edit instruction:\n{instruction}\n\n"
        "Current complete file content:\n"
        "-----BEGIN FILE-----\n"
        f"{content}\n"
        "-----END FILE-----"
    )
    raw = chat_reply(query=prompt, history=[], context_block=context_block, model_id=model_id)
    parsed = _extract_json_object(raw)
    new_content = str(parsed.get("new_content", ""))
    if not new_content:
        raise ValueError("The edit proposal did not include new_content.")
    summary = str(parsed.get("summary", "")).strip() or "Generated a file edit proposal."
    return {
        "tool": "propose_file_edit",
        "path": str(path),
        "writeRoot": str(get_file_write_root().resolve()),
        "instruction": instruction,
        "summary": summary,
        "originalContent": content,
        "newContent": new_content,
        "sha256Before": _sha256_text(content),
        "sha256After": _sha256_text(new_content),
        "changed": content != new_content,
        "diff": _unified_diff(content, new_content, path),
    }


def apply_file_edit(path_text: str, new_content: str, sha256_before: str = "", instruction: str = "") -> dict[str, Any]:
    """把 propose 阶段生成的新内容写入文件，并在写入前做 hash 校验和备份。"""
    path = _safe_write_path(path_text)
    current = _read_text_file_for_edit(path, max_chars=2_000_000)
    current_hash = _sha256_text(current)
    if sha256_before and sha256_before != current_hash:
        raise ValueError("File changed after the proposal was generated. Please create a new proposal before applying.")

    backups_dir = get_file_backups_dir().resolve()
    backups_dir.mkdir(parents=True, exist_ok=True)
    root = get_file_write_root().resolve()
    try:
        relative = path.relative_to(root)
        backup_stem = "__".join(relative.parts)
    except ValueError:
        backup_stem = path.name
    backup_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", backup_stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backups_dir / f"{backup_stem}.{timestamp}.bak"
    backup_path.write_text(current, encoding="utf-8")

    path.write_text(new_content, encoding="utf-8")
    return {
        "tool": "apply_file_edit",
        "path": str(path),
        "backupPath": str(backup_path),
        "instruction": instruction.strip(),
        "sha256Before": current_hash,
        "sha256After": _sha256_text(new_content),
        "bytesWritten": len(new_content.encode("utf-8")),
    }


def analyze_local_path(path_text: str, prompt: str = "", max_chars: int = 12000) -> dict[str, Any]:
    """分析单个文件或目录，返回内容预览、文件清单和可供 Agent 使用的上下文。"""
    path = _safe_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    read_result = read_local_file(str(path), max_chars=max_chars)
    question = prompt.strip() or "请概括这个文件或目录的内容、用途、关键结构，以及值得注意的地方。"

    if read_result["kind"] == "directory":
        content_block = "目录内容如下：\n" + "\n".join(f"- {entry}" for entry in read_result.get("entries", []))
    elif read_result["kind"] == "text":
        content_block = f"文件路径：{read_result['path']}\n\n文件内容片段：\n{read_result['content']}"
    else:
        content_block = (
            f"文件路径：{read_result['path']}\n"
            f"文件类型：{path.suffix or '<none>'}\n"
            "这是一个当前不直接按文本读取的文件，请基于路径和类型说明它可能是什么，并建议下一步如何处理。"
        )

    analysis = chat_reply(
        query=question,
        history=[],
        context_block=content_block[: max_chars + 1000],
    )
    return {
        "tool": "analyze_local_path",
        "path": str(path),
        "read": read_result,
        "question": question,
        "analysis": analysis,
    }


def list_repo_directories(limit: int = 80) -> dict[str, Any]:
    repos_dir = get_repos_dir().resolve()
    repos_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for item in sorted(repos_dir.iterdir(), key=lambda entry: entry.name.lower()):
        if item.is_dir():
            items.append(
                {
                    "name": item.name,
                    "path": str(item),
                }
            )
        if len(items) >= limit:
            break
    return {"tool": "list_repo_directories", "root": str(repos_dir), "items": items}


def clone_github_repo(repo_url: str, branch: str = "", target_name: str = "") -> dict[str, Any]:
    url = repo_url.strip()
    if not url:
        raise ValueError("Repository URL cannot be empty.")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https repository URLs are supported.")
    if "github.com" not in parsed.netloc.lower():
        raise ValueError("Only GitHub repository URLs are supported in this tool.")

    repo_name = Path(parsed.path.rstrip("/")).name
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    final_name = _slugify_name(target_name or repo_name)

    repos_dir = get_repos_dir().resolve()
    repos_dir.mkdir(parents=True, exist_ok=True)
    target_dir = (repos_dir / final_name).resolve()
    if target_dir.exists():
        raise FileExistsError(f"Target directory already exists: {target_dir}")

    command = ["git", "clone", "--depth", "1"]
    if branch.strip():
        command.extend(["--branch", branch.strip()])
    command.extend([url, str(target_dir)])

    completed = subprocess.run(command, capture_output=True, text=True, cwd=repos_dir)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git clone failed").strip()
        raise RuntimeError(detail)

    return {
        "tool": "clone_github_repo",
        "url": url,
        "branch": branch.strip() or "",
        "target": str(target_dir),
        "stdout": (completed.stdout or "").strip(),
    }


def summarize_experiment_text(text: str, max_chars: int = 2200) -> dict[str, Any]:
    """把实验日志或训练结果压缩成短摘要，便于 Agent 在回答中引用。"""
    compact = re.sub(r"\s+", " ", text or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()][:20]
    keywords = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", compact):
        lowered = token.lower()
        if lowered not in keywords:
            keywords.append(lowered)
        if len(keywords) >= 16:
            break
    return {
        "tool": "summarize_experiment_text",
        "summary_hint": compact[:max_chars],
        "lines": lines,
        "keywords": keywords,
    }


def render_tool_result(result: dict[str, Any]) -> str:
    tool = result.get("tool", "tool")
    return json.dumps({"tool": tool, "payload": result}, ensure_ascii=False, indent=2)
