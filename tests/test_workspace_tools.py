from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from backend.tool_registry import TOOL_REGISTRY
from backend.workspace_security import is_binary_file, resolve_safe_path
from backend.workspace_tools import build_file_index, list_files, read_file, search_text


class WorkspaceToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_root = os.environ.get("LOCAL_WORKSPACE_ROOT")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.environ["LOCAL_WORKSPACE_ROOT"] = str(self.root)
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("def login():\n    return 'ok'\n\ndef logout():\n    return 'bye'\n", encoding="utf-8")
        (self.root / "README.md").write_text("# Demo\nThis project has login logic.\n", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "ignored.js").write_text("login should not be found\n", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("secret\n", encoding="utf-8")
        (self.root / "binary.bin").write_bytes(b"\x00\x01\x02")
        (self.root / "large.txt").write_text("\n".join(f"line {index}" for index in range(200)), encoding="utf-8")

    def tearDown(self) -> None:
        if self._old_root is None:
            os.environ.pop("LOCAL_WORKSPACE_ROOT", None)
        else:
            os.environ["LOCAL_WORKSPACE_ROOT"] = self._old_root
        self.temp.cleanup()

    def test_safe_path_rejects_outside_absolute_path(self) -> None:
        with self.assertRaises(ValueError):
            resolve_safe_path(str(self.root.parent / "outside.txt"))

    def test_safe_path_rejects_parent_traversal(self) -> None:
        with self.assertRaises(ValueError):
            resolve_safe_path("../outside.txt")

    def test_list_files_ignores_common_directories(self) -> None:
        result = list_files(".", max_depth=3)
        paths = {item["path"] for item in result["items"]}
        self.assertIn("src/app.py", paths)
        self.assertNotIn("node_modules/ignored.js", paths)
        self.assertNotIn(".git/config", paths)

    def test_read_file_line_range(self) -> None:
        result = read_file("src/app.py", start_line=1, end_line=2)
        self.assertEqual(result["start_line"], 1)
        self.assertEqual(result["end_line"], 2)
        self.assertIn("1: def login():", result["content"])
        self.assertIn("2:     return 'ok'", result["content"])

    def test_read_file_truncates_large_content(self) -> None:
        result = read_file("large.txt", max_bytes=40)
        self.assertTrue(result["truncated"])
        self.assertLess(len(result["content"].encode("utf-8")), 80)

    def test_search_text_returns_path_and_line(self) -> None:
        result = search_text("login", max_results=10)
        matches = {(item["path"], item["line"]) for item in result["matches"]}
        self.assertIn(("src/app.py", 1), matches)
        self.assertIn(("README.md", 2), matches)

    def test_search_text_ignores_common_directories(self) -> None:
        result = search_text("should not be found", max_results=10)
        self.assertEqual(result["matches"], [])

    def test_binary_file_is_not_read_as_text(self) -> None:
        self.assertTrue(is_binary_file(self.root / "binary.bin"))
        result = read_file("binary.bin")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "binary_file_not_read")

    def test_build_file_index_returns_stats_and_hash(self) -> None:
        result = build_file_index(".")
        files = {item["path"]: item for item in result["files"]}
        self.assertIn("src/app.py", files)
        self.assertEqual(files["src/app.py"]["language"], "python")
        self.assertGreater(files["src/app.py"]["lines"], 0)
        self.assertTrue(files["src/app.py"]["hash"])
        self.assertGreaterEqual(result["stats"]["languages"]["python"], 1)

    def test_agent_tool_registry_contains_workspace_tools(self) -> None:
        for name in ["list_files", "read_file", "search_text", "get_file_metadata", "build_file_index"]:
            self.assertIn(name, TOOL_REGISTRY)


if __name__ == "__main__":
    unittest.main()
