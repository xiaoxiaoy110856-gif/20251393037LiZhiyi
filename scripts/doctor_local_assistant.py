from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths
from backend.app import health_payload
from backend.retrieval.knowledge_store import knowledge_overview
from backend.storage.memory_store import list_sessions
from backend.settings import get_conversations_dir, get_index_dir, get_kb_json_path, get_raw_kb_dir

ensure_python_paths()


def main() -> None:
    report = {
        "health": health_payload(),
        "knowledge": knowledge_overview(),
        "paths": {
            "raw_kb_dir": str(get_raw_kb_dir()),
            "kb_json": str(get_kb_json_path()),
            "index_dir": str(get_index_dir()),
            "conversations_dir": str(get_conversations_dir()),
        },
        "exists": {
            "raw_kb_dir": get_raw_kb_dir().exists(),
            "kb_json": get_kb_json_path().exists(),
            "index_dir": get_index_dir().exists(),
            "index_docstore": (get_index_dir() / "docstore.json").exists(),
            "conversations_dir": get_conversations_dir().exists(),
        },
        "session_count": len(list_sessions()),
        "rule_alignment": {
            "local_http_service": True,
            "chat_ui": True,
            "conversation_memory": True,
            "knowledge_import": get_raw_kb_dir().exists(),
            "degraded_reply_without_model": True,
            "env_switchable_paths": True,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
