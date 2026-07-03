from __future__ import annotations

import fnmatch
import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from backend.settings import get_workspace_index_max_file_bytes, get_workspace_max_read_bytes
from backend.workspace.security import (
    IGNORED_DIRS,
    get_file_metadata as secure_file_metadata,
    is_binary_file,
    is_hidden_path,
    is_ignored_path,
    is_sensitive_path,
    resolve_safe_path,
    to_workspace_relative,
    workspace_root,
)


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".vue": "vue",
    ".css": "css",
    ".scss": "scss",
    ".html": "html",
    ".md": "markdown",
    ".json": "json",
    ".jsonl": "jsonl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".csv": "csv",
    ".txt": "text",
    ".ps1": "powershell",
    ".sh": "shell",
}


def _matches_glob(rel_path: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    return fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(Path(rel_path).name, pattern)


def _should_skip(path: Path, include_hidden: bool = False) -> bool:
    if is_ignored_path(path):
        return True
    if not include_hidden and is_hidden_path(path):
        return True
    return False


def _iter_files(base: Path, include_hidden: bool = False):
    for child in sorted(base.iterdir(), key=lambda item: item.name.lower()):
        if _should_skip(child, include_hidden=include_hidden):
            continue
        if child.is_dir():
            yield from _iter_files(child, include_hidden=include_hidden)
        else:
            yield child


def _line_count(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for total, _ in enumerate(handle, start=1):
            pass
    return total


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_files(
    path: str = ".",
    max_depth: int = 3,
    include_glob: str = "",
    exclude_glob: str = "",
    include_hidden: bool = False,
    limit: int = 1000,
) -> dict[str, Any]:
    root = resolve_safe_path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not root.is_dir():
        metadata = secure_file_metadata(root)
        metadata["depth"] = 0
        return {"tool": "list_files", "root": to_workspace_relative(root), "items": [metadata], "truncated": False}

    items: list[dict[str, Any]] = []
    truncated = False
    base_depth = len(root.parts)
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name.lower(), reverse=True)
        except OSError:
            continue
        for child in children:
            if _should_skip(child, include_hidden=include_hidden):
                continue
            depth = len(child.parts) - base_depth
            if depth > max_depth:
                continue
            rel = to_workspace_relative(child)
            if include_glob and not child.is_dir() and not _matches_glob(rel, include_glob):
                continue
            if exclude_glob and _matches_glob(rel, exclude_glob):
                continue
            metadata = secure_file_metadata(child)
            metadata["depth"] = depth
            items.append(metadata)
            if len(items) >= limit:
                truncated = True
                stack.clear()
                break
            if child.is_dir() and depth < max_depth:
                stack.append(child)
        if truncated:
            break

    items.sort(key=lambda item: (item["path"].count("/"), item["path"]))
    return {
        "tool": "list_files",
        "root": to_workspace_relative(root) if root != workspace_root() else ".",
        "items": items,
        "count": len(items),
        "truncated": truncated,
        "ignored_dirs": sorted(IGNORED_DIRS),
    }


def read_file(path: str, start_line: int = 1, end_line: int | None = None, max_bytes: int | None = None) -> dict[str, Any]:
    resolved = resolve_safe_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {path}")
    metadata = secure_file_metadata(resolved)
    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if is_sensitive_path(resolved):
        return {"tool": "read_file", "ok": False, "error": "sensitive_file_blocked", "metadata": metadata}
    if is_binary_file(resolved):
        return {"tool": "read_file", "ok": False, "error": "binary_file_not_read", "metadata": metadata}

    byte_limit = max(1, int(max_bytes or get_workspace_max_read_bytes()))
    start = max(1, int(start_line or 1))
    end = int(end_line) if end_line else None
    if end is not None and end < start:
        raise ValueError("end_line must be greater than or equal to start_line.")

    selected: list[str] = []
    total_lines = 0
    used_bytes = 0
    truncated = False
    with resolved.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_lines = line_number
            if line_number < start:
                continue
            if end is not None and line_number > end:
                continue
            rendered = f"{line_number}: {line.rstrip()}"
            rendered_bytes = len((rendered + "\n").encode("utf-8", errors="replace"))
            if used_bytes + rendered_bytes > byte_limit:
                truncated = True
                continue
            selected.append(rendered)
            used_bytes += rendered_bytes

    actual_end = start + len(selected) - 1 if selected else start
    return {
        "tool": "read_file",
        "ok": True,
        "path": metadata["path"],
        "start_line": start,
        "end_line": actual_end,
        "total_lines": total_lines,
        "size": metadata["size"],
        "content": "\n".join(selected),
        "truncated": truncated or (end is not None and end < total_lines),
        "metadata": metadata,
    }


def _fallback_search(
    query: str,
    base: Path,
    glob: str,
    regex: bool,
    case_sensitive: bool,
    max_results: int,
) -> dict[str, Any]:
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query if regex else re.escape(query), flags)
    matches: list[dict[str, Any]] = []
    total = 0
    truncated = False
    for file_path in _iter_files(base):
        rel = to_workspace_relative(file_path)
        if glob and not _matches_glob(rel, glob):
            continue
        if is_sensitive_path(file_path) or is_binary_file(file_path):
            continue
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    found = pattern.search(line)
                    if not found:
                        continue
                    total += 1
                    if len(matches) >= max_results:
                        truncated = True
                        continue
                    matches.append(
                        {
                            "path": rel,
                            "line": line_number,
                            "column": found.start() + 1,
                            "preview": line.strip()[:300],
                        }
                    )
        except OSError:
            continue
    return {"matches": matches, "total_matches": total, "truncated": truncated}


def search_text(
    query: str,
    path: str = ".",
    glob: str = "",
    regex: bool = False,
    case_sensitive: bool = False,
    max_results: int = 100,
) -> dict[str, Any]:
    if not query:
        raise ValueError("query cannot be empty.")
    base = resolve_safe_path(path)
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    limit = max(1, int(max_results or 100))
    rg = shutil.which("rg")
    if rg:
        command = [rg, "--line-number", "--column", "--no-heading", "--color", "never"]
        for ignored in sorted(IGNORED_DIRS):
            command.extend(["--glob", f"!{ignored}/**"])
        for pattern in ["!.env", "!.env.*", "!*.pem", "!*.key", "!*.p12", "!*.pfx"]:
            command.extend(["--glob", pattern])
        if not case_sensitive:
            command.append("--ignore-case")
        if not regex:
            command.append("--fixed-strings")
        if glob:
            command.extend(["--glob", glob])
        command.extend([query, str(base)])
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        matches: list[dict[str, Any]] = []
        total = 0
        for line in completed.stdout.splitlines():
            match = re.match(r"^(.*?):(\d+):(\d+):(.*)$", line)
            if not match:
                continue
            file_name, line_text, column_text, preview = match.groups()
            try:
                file_path = resolve_safe_path(file_name)
            except ValueError:
                continue
            if is_ignored_path(file_path) or is_sensitive_path(file_path):
                continue
            total += 1
            if len(matches) >= limit:
                continue
            matches.append(
                {
                    "path": to_workspace_relative(file_path),
                    "line": int(line_text or 0),
                    "column": int(column_text or 0),
                    "preview": preview.strip()[:300],
                }
            )
        return {"tool": "search_text", "query": query, "matches": matches, "total_matches": total, "truncated": total > len(matches)}

    fallback = _fallback_search(query, base, glob, regex, case_sensitive, limit)
    return {"tool": "search_text", "query": query, **fallback}


def get_file_metadata(path: str) -> dict[str, Any]:
    metadata = secure_file_metadata(resolve_safe_path(path))
    return {"tool": "get_file_metadata", "metadata": metadata}


def build_file_index(path: str = ".", include_hidden: bool = False, limit: int = 5000) -> dict[str, Any]:
    base = resolve_safe_path(path)
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    files: list[dict[str, Any]] = []
    stats = {"total_files": 0, "total_size": 0, "languages": {}}
    truncated = False
    max_file_bytes = get_workspace_index_max_file_bytes()
    iterator = _iter_files(base, include_hidden=include_hidden) if base.is_dir() else iter([base])
    for file_path in iterator:
        if len(files) >= limit:
            truncated = True
            break
        if is_sensitive_path(file_path):
            continue
        metadata = secure_file_metadata(file_path)
        language = LANGUAGE_BY_EXTENSION.get(file_path.suffix.lower(), "binary" if metadata["is_binary"] else "text")
        lines = 0
        digest = ""
        if not metadata["is_binary"] and metadata["size"] <= max_file_bytes:
            try:
                lines = _line_count(file_path)
                digest = _hash_file(file_path)
            except OSError:
                continue
        elif metadata["size"] <= max_file_bytes:
            digest = _hash_file(file_path)
        files.append(
            {
                "path": metadata["path"],
                "language": language,
                "extension": metadata["extension"],
                "size": metadata["size"],
                "lines": lines,
                "hash": digest,
                "is_binary": metadata["is_binary"],
            }
        )
        stats["total_files"] += 1
        stats["total_size"] += int(metadata["size"])
        languages = stats["languages"]
        languages[language] = int(languages.get(language, 0)) + 1
    return {
        "tool": "build_file_index",
        "root": to_workspace_relative(base) if base != workspace_root() else ".",
        "files": files,
        "stats": stats,
        "truncated": truncated,
        "ignored_dirs": sorted(IGNORED_DIRS),
    }
