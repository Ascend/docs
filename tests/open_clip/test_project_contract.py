"""Static contract for the migrated open_clip Quick Start."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "sources" / "open_clip"
DOC = SOURCE_DIR / "quick_start.md"
WORKFLOW = ROOT / ".github" / "workflows" / "open-clip-quick-start.yml"


class TestProjectContract(unittest.TestCase):
    def test_guarded_document_replaces_the_legacy_quick_start(self) -> None:
        self.assertFalse((SOURCE_DIR / "quick_start.rst").exists())
        text = DOC.read_text(encoding="utf-8")
        test_ids = set(re.findall(r'#test id="([^"]+)"', text))
        result_ids = set(re.findall(r'#test-result id="([^"]+)"', text))

        self.assertEqual(
            test_ids,
            {"check-python", "check-torch", "install-open-clip", "npu-inference", "npu-training"},
        )
        self.assertEqual(result_ids, test_ids)
        self.assertIn("pretrained=\"laion2b_s34b_b79k\"", text)
        self.assertIn("--dataset-type synthetic", text)
        self.assertIn("NPU training PASSED", text)

    def test_install_page_no_longer_conflicts_with_the_guarded_stack(self) -> None:
        text = (SOURCE_DIR / "install.rst").read_text(encoding="utf-8")

        self.assertNotIn("2.2.0", text)
        self.assertNotIn("open_cliop", text)
        self.assertIn("quick_start", text)

    def test_workflow_preserves_the_validated_runtime_and_cache(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", text)
        self.assertIn("project: open_clip", text)
        self.assertIn("linux-aarch64-a2-1", text)
        self.assertIn("cann:9.1.0-910b-ubuntu22.04-py3.12", text)
        self.assertIn("/data/ci-cache/huggingface/open-clip", text)
        self.assertIn("upstream_repo: mlfoundations/open_clip", text)
        self.assertIn("tests.open_clip.test_quick_start_ascend", text)


if __name__ == "__main__":
    unittest.main()
