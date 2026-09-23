"""Static contract for the migrated InternLM Quick Start."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "sources" / "internlm" / "quick_start.md"
WORKFLOW = ROOT / ".github" / "workflows" / "internlm-quick-start.yml"


class TestProjectContract(unittest.TestCase):
    def test_document_keeps_the_upstream_main_npu_inference(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        test_ids = set(re.findall(r'#test id="([^"]+)"', text))
        result_ids = set(re.findall(r'#test-result id="([^"]+)"', text))

        self.assertEqual(
            test_ids,
            {"check-python", "check-torch", "checkout-upstream", "install-deps", "download-model", "npu-inference"},
        )
        self.assertEqual(result_ids, test_ids)
        self.assertIn("ecosystem/README_npu.md", text)
        self.assertIn("internlm3-8b-instruct", text)
        self.assertIn(").npu()", text)
        self.assertNotIn("bitsandbytes", text)
        self.assertTrue(text.startswith("# InternLM\n"))
        self.assertNotIn("python - <<'PY'", text)
        for test_id in ("check-torch", "install-deps", "download-model", "npu-inference"):
            self.assertIn(f'```python #test id="{test_id}"', text)
        self.assertEqual(text.count("输出结果如下：\n\n```text #test-result"), len(test_ids))
        self.assertIn("torch: 2.9.0+cpu", text)

    def test_workflow_tracks_main_and_serializes_the_model_cache(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("group: internlm-quick-start", text)
        self.assertIn("project: internlm", text)
        self.assertIn("fixed_ref: main", text)
        self.assertIn("linux-aarch64-a2-1", text)
        self.assertIn("/data/ci-cache/modelscope/internlm", text)
        self.assertIn("timeout_minutes: 180", text)
        self.assertIn("upstream_repo: InternLM/InternLM", text)
        self.assertIn("tests.internlm.test_quick_start_ascend", text)

    def test_project_is_reachable_from_the_site_navigation(self) -> None:
        index = (ROOT / "index.rst").read_text(encoding="utf-8")
        project_index = ROOT / "sources" / "internlm" / "index.rst"

        self.assertIn("sources/internlm/index.rst", index)
        self.assertIn('href="sources/internlm/index.html">快速上手', index)
        self.assertEqual(
            project_index.read_text(encoding="utf-8"),
            ".. include:: quick_start.md\n   :parser: myst_parser.sphinx_\n",
        )


if __name__ == "__main__":
    unittest.main()
