from __future__ import annotations

import site
import sys
import os
from pathlib import Path


def ensure_python_paths() -> None:
    root = Path(__file__).resolve().parent.parent
    vendor_roots = [root / ".pythonlibs", root / ".vendorlibs"]
    vendor_texts = {str(path) for path in vendor_roots}
    sys.path[:] = [entry for entry in sys.path if entry not in vendor_texts]
    try:
        site.addsitedir(site.getusersitepackages())
    except Exception:
        pass

    use_project_pythonlibs = os.getenv("LOCAL_USE_PROJECT_PYTHONLIBS", "0").strip().lower() in {"1", "true", "yes"}
    if not use_project_pythonlibs:
        return

    for candidate in vendor_roots:
        try:
            ready_files = [
                candidate / "pypdf" / "__init__.py",
                candidate / "llama_index" / "__init__.py",
                candidate / "llama_index" / "core" / "__init__.py",
            ]
            looks_ready = any(path.exists() for path in ready_files)
            if looks_ready:
                readable = next((path for path in ready_files if path.exists()), None)
                if readable:
                    with readable.open("rb"):
                        pass
        except Exception:
            looks_ready = False

        if not looks_ready:
            continue

        text = str(candidate)
        if text not in sys.path:
            site.addsitedir(text)
