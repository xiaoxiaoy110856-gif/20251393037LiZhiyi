from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths
from backend.retrieval.knowledge_store import build_index, build_kb

ensure_python_paths()


def main() -> None:
    kb = build_kb()
    index = build_index()
    print(json.dumps({"kb": kb["source"], "index": index}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
