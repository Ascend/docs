"""Static contract for the migrated Ray Quick Start."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "sources" / "Ray"
DOC = SOURCE_DIR / "quick_start.md"
USAGE = SOURCE_DIR / "usage.rst"
WORKFLOW = ROOT / ".github" / "workflows" / "ray-quick-start.yml"


class TestProjectContract(unittest.TestCase):
    def test_quick_start_is_only_the_guarded_npu_path(self) -> None:
        self.assertFalse((SOURCE_DIR / "quick_start.rst").exists())
        text = DOC.read_text(encoding="utf-8")
        test_ids = set(re.findall(r'#test id="([^"]+)"', text))
        result_ids = set(re.findall(r'#test-result id="([^"]+)"', text))

        self.assertEqual(
            test_ids,
            {"check-py", "check-torch", "ray-install", "ray-detects-npus", "ray-isolates-npus"},
        )
        self.assertEqual(result_ids, test_ids)
        self.assertIn('resources={"NPU": 1}', text)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES", text)
        self.assertNotIn("ray job submit", text)
        self.assertTrue(text.startswith("# Ray\n"))
        self.assertNotIn("python - <<'PY'", text)
        for test_id in ("check-torch", "ray-detects-npus", "ray-isolates-npus"):
            self.assertIn(f'```python #test id="{test_id}"', text)
        self.assertEqual(text.count("输出结果如下：\n\n```text #test-result"), len(test_ids))
        self.assertIn("torch= 2.9.0+cpu", text)

    def test_legacy_general_usage_is_preserved_separately(self) -> None:
        text = USAGE.read_text(encoding="utf-8")

        self.assertIn("ray start --head", text)
        self.assertIn("ray job submit", text)
        self.assertIn("RAY_DEDUP_LOGS", text)
        self.assertIn("--dashboard-host", text)
        self.assertTrue(text.startswith(":orphan:\n"))

    def test_project_is_one_sidebar_page(self) -> None:
        project_index = (SOURCE_DIR / "index.rst").read_text(encoding="utf-8")
        homepage = (ROOT / "index.rst").read_text(encoding="utf-8")

        self.assertEqual(
            project_index,
            ".. include:: quick_start.md\n   :parser: myst_parser.sphinx_\n",
        )
        self.assertIn('href="sources/Ray/index.html">快速上手', homepage)

    def test_workflow_keeps_the_two_npu_runtime(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", text)
        self.assertIn("project: ray", text)
        self.assertIn("linux-aarch64-a2-2", text)
        self.assertIn("cann:9.1.0-910b-ubuntu22.04-py3.12", text)
        self.assertIn("upstream_repo: ray-project/ray", text)
        self.assertIn("tests.ray.test_quick_start_ascend", text)
        self.assertNotIn("ray-examples", text)


if __name__ == "__main__":
    unittest.main()
