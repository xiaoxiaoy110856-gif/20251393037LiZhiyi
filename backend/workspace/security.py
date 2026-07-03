from __future__ import annotations

import fnmatch
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.settings import get_workspace_root

# 沙盒安全边界：所有工作区文件工具都必须先经过 resolve_safe_path，
# 这样可以阻止 ../ 路径穿越和指向工作区外部的符号链接。


IGNORED_DIRS = {
    ".git",
    ".dbvendor",
    ".pythonlibs",
    ".vendorlibs",
    "node_modules",
    "dist",
    "build",
    "venv",
    ".venv",
    "__pycache__",
    ".next",
    "coverage",
    ".cache",
    "dbvendor",
    "dbvendor_manual",
    "wheelhouse",
}

SENSITIVE_GLOBS = {
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.crt",
    "*.p12",
    "*.pfx",
    "*token*",
    "*secret*",
    "*private*key*",
    "id_rsa",
    "id_ed25519",
}


def workspace_root() -> Path:
    """返回允许读写的工作区根目录，是本项目文件沙盒的边界。"""
    return get_workspace_root().expanduser().resolve()


def resolve_safe_path(relative_path: str = ".") -> Path:
    # 核心4/沙盒：所有文件工具都必须经过这里解析路径，确认最终路径仍在 LOCAL_WORKSPACE_ROOT 内。
    # 这里会先解析绝对路径和符号链接，再做包含关系检查，从而阻止 ../ 和外部链接逃逸。
    root = workspace_root()
    raw = (relative_path or ".").strip().strip("\"'")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.expanduser().resolve()
    if not (resolved == root or root in resolved.parents):
        raise ValueError(f"Path is outside workspace root: {relative_path}")
    return resolved


def to_workspace_relative(path: Path) -> str:
    """把绝对路径转换成相对工作区路径，方便前端和日志展示。"""
    resolved = path.expanduser().resolve()
    return resolved.relative_to(workspace_root()).as_posix()


def is_hidden_path(path: Path) -> bool:
    """判断路径中是否包含隐藏目录或隐藏文件。"""
    try:
        rel = path.resolve().relative_to(workspace_root())
    except ValueError:
        rel = path
    return any(part.startswith(".") for part in rel.parts if part not in {"."})


def is_ignored_path(path: Path) -> bool:
    """判断路径是否属于 node_modules、dist、缓存目录等默认忽略范围。"""
    try:
        rel = path.resolve().relative_to(workspace_root())
    except ValueError:
        return True
    return any(part in IGNORED_DIRS for part in rel.parts)


def is_sensitive_path(path: Path) -> bool:
    """判断文件名是否命中 .env、密钥、token 等敏感文件模式。"""
    name = path.name.lower()
    rel = to_workspace_relative(path).lower()
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern) for pattern in SENSITIVE_GLOBS)


def is_binary_file(path: Path) -> bool:
    """通过文件头判断是否是二进制文件，避免把图片/模型等二进制内容当文本读取。"""
    if not path.is_file():
        return False
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return True
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def get_file_metadata(path: Path | str) -> dict[str, Any]:
    resolved = resolve_safe_path(str(path))
    stat = resolved.stat()
    kind = "directory" if resolved.is_dir() else "file"
    metadata: dict[str, Any] = {
        "path": to_workspace_relative(resolved),
        "type": kind,
        "size": stat.st_size,
        "extension": resolved.suffix.lower(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "is_binary": is_binary_file(resolved) if resolved.is_file() else False,
        "sensitive": is_sensitive_path(resolved) if resolved.is_file() else False,
    }
    return metadata
