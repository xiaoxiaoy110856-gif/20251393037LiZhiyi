from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from backend.settings import ROOT
from backend.tools import apply_file_edit
from backend.workspace_tools import read_file


class FileSandboxWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / "outputs" / "sandbox_smoke" / "agent_file_tool_test.txt"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("alpha\n", encoding="utf-8")

    def test_read_and_apply_file_edit_inside_workspace(self) -> None:
        read_result = read_file("outputs/sandbox_smoke/agent_file_tool_test.txt")
        self.assertTrue(read_result["ok"])
        self.assertIn("alpha", read_result["content"])

        before = self.path.read_text(encoding="utf-8")
        digest = hashlib.sha256(before.encode("utf-8")).hexdigest()
        apply_result = apply_file_edit(str(self.path), "alpha\nbeta\n", sha256_before=digest, instruction="append beta")

        self.assertEqual(apply_result["path"], str(self.path.resolve()))
        self.assertGreater(apply_result["bytesWritten"], 0)
        self.assertTrue(Path(apply_result["backupPath"]).exists())
        reread = read_file("outputs/sandbox_smoke/agent_file_tool_test.txt")
        self.assertIn("beta", reread["content"])

    def test_path_escape_is_blocked(self) -> None:
        with self.assertRaises(ValueError):
            read_file("../outside.txt")


if __name__ == "__main__":
    unittest.main()
