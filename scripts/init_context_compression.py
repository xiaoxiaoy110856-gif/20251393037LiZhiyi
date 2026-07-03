from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.bootstrap import ensure_python_paths

ensure_python_paths()

from backend.storage.db import database_status, init_db


def main() -> None:
    ok = init_db()
    print(json.dumps({"ok": ok, "status": database_status(), "migration": "context_compression_tables"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
