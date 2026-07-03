from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.retrieval.knowledge_store import build_kb, knowledge_overview


def main() -> None:
    payload = build_kb()
    overview = knowledge_overview()
    summary = {
        "source": payload["source"],
        "documents": overview["documentCount"],
        "chunks": overview["chunkCount"],
        "topics": overview["topics"],
        "sampleTitles": overview["sampleTitles"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
