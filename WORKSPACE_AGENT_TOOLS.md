# Workspace Agent Tools

This project now gives the local Agent a Codex-like, folder-level workspace toolset without allowing access to the whole disk.

## Configure Workspace Root

Set the workspace root before starting the backend:

```powershell
$env:LOCAL_WORKSPACE_ROOT="C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\trl"
$env:LOCAL_WORKSPACE_MAX_READ_BYTES="200000"
$env:LOCAL_WORKSPACE_INDEX_MAX_FILE_BYTES="2000000"
```

If `LOCAL_WORKSPACE_ROOT` is not set, the project root is used.

## Tools

Tool registration lives in:

```text
backend/tool_registry.py
```

Agent routing lives in:

```text
backend/agent_loop.py
```

Workspace safety and file operations are implemented in:

```text
backend/workspace_security.py
backend/workspace_tools.py
```

### `list_files`

Lists workspace files and directories without reading content.

Parameters:

- `path`: default `"."`
- `max_depth`: default `3`
- `include_glob`: optional file glob
- `exclude_glob`: optional file glob
- `include_hidden`: default `false`

Returns path, type, size, extension, modified time, depth, and truncation status.

### `read_file`

Reads a text file or a line range with line numbers.

Parameters:

- `path`: required
- `start_line`: default `1`
- `end_line`: optional
- `max_bytes`: optional, defaults to `LOCAL_WORKSPACE_MAX_READ_BYTES`

Returns path, line range, total lines, file size, line-numbered content, metadata, and `truncated`.

### `search_text`

Searches workspace text files.

Parameters:

- `query`: required
- `path`: default `"."`
- `glob`: optional
- `regex`: default `false`
- `case_sensitive`: default `false`
- `max_results`: default `100`

Uses `rg` when available, otherwise falls back to Python recursive search. Results include path, line, column, and preview.

### `get_file_metadata`

Returns metadata for one workspace path: type, size, extension, modified time, binary flag, and sensitive flag.

### `build_file_index`

Builds a project manifest with language, extension, size, line count, hash, binary flag, and language statistics.

Parameters:

- `path`: default `"."`
- `include_hidden`: default `false`
- `limit`: default `5000`

## Default Ignored Directories

The workspace tools skip:

`.git`, `node_modules`, `dist`, `build`, `venv`, `.venv`, `__pycache__`, `.next`, `coverage`, `.cache`

## Security Limits

- All paths go through `resolve_safe_path`.
- `..` traversal and symlinks escaping the workspace are blocked after `Path.resolve()`.
- Binary files are not read as text.
- Sensitive files are blocked for content reads and skipped by search/index by default.
- Sensitive patterns include `.env`, `.env.*`, `*.pem`, `*.key`, `*.crt`, `*.p12`, `*.pfx`, `*token*`, `*secret*`, `*private*key*`, `id_rsa`, `id_ed25519`.
- Large reads are truncated by byte limit.
- The Agent is instructed to search first, then read targeted line ranges.

## Agent Workflow Examples

For “分析这个项目结构”:

1. `build_file_index({"path": ".", "limit": 5000})`
2. `list_files({"path": ".", "max_depth": 2})`
3. Answer with module summary and cite paths from the manifest.

For “登录逻辑在哪里实现”:

1. `search_text({"query": "login", "path": ".", "glob": "*", "max_results": 50})`
2. `read_file({"path": "matched/file.py", "start_line": 40, "end_line": 100})`
3. Answer with the matching file path and line numbers.

## Document Parser Extension

The current minimum implementation focuses on text/code files. To add document parsing, create a parser service with:

- `parse_document(path)`
- `read_document_page(path, page_number)`
- `list_document_tables(path)`
- `read_document_table(path, table_id)`

PDF can use the existing RAG stack dependency `pypdf` when installed. DOCX/XLSX/PPTX should be added behind optional imports so the assistant still starts without those packages.

## Embedding / Vector Search Extension

The project already has RAG code in `backend/knowledge_store.py`. The workspace manifest can be connected later by converting indexed files into chunks and feeding them into the existing index build path. The current version intentionally avoids adding new vector dependencies.
