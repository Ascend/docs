"""Contract for the migrated VBench Ascend Quick Start."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "sources" / "vbench"
DOC = SOURCE / "quick_start.md"
WORKFLOW = ROOT / ".github" / "workflows" / "vbench-quick-start.yml"
TEST = ROOT / "tests" / "vbench" / "test_quick_start_ascend.py"
HELPER_REF = "abaa5c1610c57f34eeaa73208d91b6a9b1573a79"


class TestVBenchContract(unittest.TestCase):
    def test_quick_start_preserves_the_public_human_action_path(self) -> None:
        self.assertTrue(DOC.is_file())
        text = DOC.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# VBench\n"))
        self.assertEqual(
            set(re.findall(r'#test id="([^"]+)"', text)),
            {"check-python", "install-torch", "checkout-and-patch", "install-vbench",
             "prepare-assets", "human-action-cli", "human-action-api"},
        )
        self.assertEqual(
            set(re.findall(r'#test-result id="([^"]+)"', text)),
            set(re.findall(r'#test id="([^"]+)"', text)),
        )
        self.assertEqual(text.count("输出结果如下："), 7)
        self.assertIn("| VBench | xxx |", text)
        self.assertIn("git clone https://github.com/cosdt-ci-test/workflows.git", text)
        self.assertIn(HELPER_REF, text)
        self.assertIn("vbench evaluate", text)
        self.assertIn('```python #test id="human-action-api"', text)
        self.assertNotIn("runtime_checks", text)

    def test_project_has_one_reachable_page(self) -> None:
        self.assertTrue((SOURCE / "index.rst").is_file())
        self.assertEqual(
            (SOURCE / "index.rst").read_text(encoding="utf-8"),
            ".. include:: quick_start.md\n   :parser: myst_parser.sphinx_\n",
        )
        home = (ROOT / "index.rst").read_text(encoding="utf-8")
        self.assertIn('href="sources/vbench/index.html">快速上手', home)
        self.assertIn("sources/vbench/index.rst", home)

    def test_guard_uses_the_npu_runtime_and_pinned_helpers(self) -> None:
        self.assertTrue(WORKFLOW.is_file())
        self.assertTrue(TEST.is_file())
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", workflow)
        self.assertIn("project: vbench", workflow)
        self.assertIn("linux-aarch64-a2-1", workflow)
        self.assertIn("cann:9.1.0-910b-ubuntu22.04-py3.12", workflow)
        self.assertIn("/data/ci-cache/vbench:/root/.cache/vbench", workflow)
        self.assertIn("upstream_repo: Vchitect/VBench", workflow)
        self.assertIn("tests.vbench.test_quick_start_ascend", workflow)
        self.assertIn(HELPER_REF, TEST.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
