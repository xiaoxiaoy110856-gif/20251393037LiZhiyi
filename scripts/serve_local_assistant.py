from __future__ import annotations

import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths
from backend.app import (
    apply_file_edit_payload,
    chat_payload,
    clear_session_payload,
    clone_repo_payload,
    create_session_payload,
    health_payload,
    image_generation_payload,
    knowledge_payload,
    session_detail_payload,
    local_file_analysis_payload,
    local_file_read_payload,
    policy_evaluation_payload,
    propose_file_edit_payload,
    repo_list_payload,
    rebuild_knowledge_payload,
    retrieval_training_payload,
    save_trajectory_payload,
    search_payload,
    sessions_payload,
    trajectory_runs_payload,
)
from backend.settings import get_generated_images_dir, get_host, get_port, get_ui_dir

ensure_python_paths()


HOST = get_host()
PORT = get_port()


def _read_static(path: str) -> tuple[bytes, str] | None:
    if path.startswith("/generated-images/"):
        target = get_generated_images_dir() / Path(path).name
        if not target.exists() or not target.is_file():
            return None
        guessed, _ = mimetypes.guess_type(str(target))
        return target.read_bytes(), guessed or "image/svg+xml"
    target = get_ui_dir() / path.lstrip("/")
    if path in {"", "/"}:
        target = get_ui_dir() / "index.html"
    if not target.exists() or not target.is_file():
        return None
    guessed, _ = mimetypes.guess_type(str(target))
    content_type = guessed or "text/plain"
    if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
        content_type = f"{content_type}; charset=utf-8"
    return target.read_bytes(), content_type


class LocalAssistantHandler(BaseHTTPRequestHandler):
    server_version = "LocalAssistantHTTP/0.1"

    def _set_headers(self, status: int = HTTPStatus.OK, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        self._set_headers(status)
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._set_headers(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                return self._write_json(health_payload())
            if parsed.path == "/api/knowledge":
                return self._write_json(knowledge_payload())
            if parsed.path == "/api/sessions":
                return self._write_json(sessions_payload())
            if parsed.path.startswith("/api/sessions/"):
                session_id = parsed.path.split("/")[-1]
                return self._write_json(session_detail_payload(session_id))
            if parsed.path == "/api/retrieval-training/latest":
                return self._write_json(retrieval_training_payload())
            if parsed.path == "/api/policy-evaluation":
                return self._write_json(policy_evaluation_payload())
            if parsed.path == "/api/repos":
                return self._write_json(repo_list_payload())
            if parsed.path == "/api/trajectories":
                params = parse_qs(parsed.query)
                limit = int((params.get("limit", ["20"])[0] or "20"))
                return self._write_json(trajectory_runs_payload(limit))
            if parsed.path == "/api/search":
                params = parse_qs(parsed.query)
                query = (params.get("query", [""])[0] or "").strip()
                top_k = int((params.get("top_k", ["4"])[0] or "4"))
                return self._write_json(search_payload(query, top_k))

            static = _read_static(parsed.path)
            if static:
                content, content_type = static
                self._set_headers(HTTPStatus.OK, content_type)
                self.wfile.write(content)
                return
            return self._write_json({"ok": False, "detail": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # pragma: no cover
            return self._write_json({"ok": False, "detail": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            if self.path == "/api/sessions":
                return self._write_json(create_session_payload(payload.get("title")))
            if self.path == "/api/knowledge/rebuild":
                return self._write_json(rebuild_knowledge_payload())
            if self.path == "/api/repos/clone":
                return self._write_json(
                    clone_repo_payload(
                        repo_url=str(payload.get("repo_url", "")).strip(),
                        branch=str(payload.get("branch", "")).strip(),
                        target_name=str(payload.get("target_name", "")).strip(),
                    )
                )
            if self.path == "/api/local-file/analyze":
                return self._write_json(
                    local_file_analysis_payload(
                        path=str(payload.get("path", "")).strip(),
                        prompt=str(payload.get("prompt", "")).strip(),
                        max_chars=int(payload.get("max_chars", 12000) or 12000),
                    )
                )
            if self.path == "/api/files/read":
                raw_end_line = payload.get("end_line")
                return self._write_json(
                    local_file_read_payload(
                        path=str(payload.get("path", "")).strip(),
                        start_line=int(payload.get("start_line", 1) or 1),
                        end_line=int(raw_end_line) if raw_end_line not in {None, ""} else None,
                        max_bytes=int(payload.get("max_bytes", 200000) or 200000),
                    )
                )
            if self.path == "/api/files/propose-edit":
                return self._write_json(
                    propose_file_edit_payload(
                        path=str(payload.get("path", "")).strip(),
                        instruction=str(payload.get("instruction", "")).strip(),
                        max_chars=int(payload.get("max_chars", 24000) or 24000),
                        model_id=str(payload.get("model_id", "")).strip(),
                    )
                )
            if self.path == "/api/files/apply-edit":
                return self._write_json(
                    apply_file_edit_payload(
                        path=str(payload.get("path", "")).strip(),
                        new_content=str(payload.get("new_content", "")),
                        sha256_before=str(payload.get("sha256_before", "")).strip(),
                        instruction=str(payload.get("instruction", "")).strip(),
                    )
                )
            if self.path == "/api/chat":
                return self._write_json(
                    chat_payload(
                        query=str(payload.get("query", "")).strip(),
                        session_id=payload.get("session_id"),
                        top_k=int(payload.get("top_k", 4) or 4),
                        attachment_name=str(payload.get("attachment_name", "")).strip(),
                        attachment_text=str(payload.get("attachment_text", "")).strip(),
                        model_id=str(payload.get("model_id", "")).strip(),
                    )
                )
            if self.path == "/api/images/generate":
                return self._write_json(
                    image_generation_payload(
                        prompt=str(payload.get("prompt", "")).strip(),
                        model=str(payload.get("model", "")).strip(),
                        size=str(payload.get("size", "1024x1024")).strip() or "1024x1024",
                        quality=str(payload.get("quality", "")).strip(),
                        format=str(payload.get("format", "")).strip(),
                        background=str(payload.get("background", "")).strip(),
                        n=int(payload.get("n", 1) or 1),
                        style_notes=str(payload.get("style_notes", "")).strip(),
                        preset=str(payload.get("preset", "")).strip(),
                        quality_mode=str(payload.get("quality_mode", "high")).strip() or "high",
                        batch_size=int(payload.get("batch_size", 1) or 1),
                        allow_retry=bool(payload.get("allow_retry", True)),
                        use_highres_fix=payload.get("use_highres_fix", True),
                    )
                )
            if self.path == "/api/trajectories":
                return self._write_json(
                    save_trajectory_payload(
                        trajectory_type=str(payload.get("trajectory_type", "")).strip() or "manual",
                        scenario_id=str(payload.get("scenario_id", "")).strip(),
                        scenario_label=str(payload.get("scenario_label", "")).strip(),
                        rl_method=str(payload.get("rl_method", "")).strip(),
                        compression_method=str(payload.get("compression_method", "")).strip(),
                        map_provider=str(payload.get("map_provider", "")).strip() or "OpenStreetMap",
                        route_provider=str(payload.get("route_provider", "")).strip() or "OSRM",
                        start=list(payload.get("start", []) or []),
                        end=list(payload.get("end", []) or []),
                        distance_km=float(payload.get("distance_km", 0) or 0),
                        duration_min=float(payload.get("duration_min", 0) or 0),
                        route_geometry=list(payload.get("route_geometry", []) or []),
                        compression=dict(payload.get("compression", {}) or {}),
                        metadata=dict(payload.get("metadata", {}) or {}),
                    )
                )
            return self._write_json({"ok": False, "detail": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # pragma: no cover
            return self._write_json({"ok": False, "detail": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            if not self.path.startswith("/api/sessions/"):
                return self._write_json({"ok": False, "detail": "Not found"}, HTTPStatus.NOT_FOUND)
            session_id = self.path.split("/")[-1]
            return self._write_json(clear_session_payload(session_id))
        except Exception as error:  # pragma: no cover
            return self._write_json({"ok": False, "detail": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), LocalAssistantHandler)
    print(f"Local assistant running on http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
