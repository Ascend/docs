"""Static contract for the migrated flash-linear-attention Quick Start."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "sources" / "flash-linear-attention" / "quick_start.md"
WORKFLOW = ROOT / ".github" / "workflows" / "flash-linear-attention-quick-start.yml"


class TestProjectContract(unittest.TestCase):
    def test_document_keeps_the_validated_npu_path(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        test_ids = set(re.findall(r'#test id="([^"]+)"', text))
        result_ids = set(re.findall(r'#test-result id="([^"]+)"', text))

        self.assertEqual(
            test_ids,
            {"check-cann", "install-fla", "check-npu", "gdn-forward-backward"},
        )
        self.assertEqual(result_ids, test_ids)
        self.assertIn('python -m pip install -q ".[npu]"', text)
        self.assertIn("loss.backward()", text)
        self.assertIn("torch.npu.synchronize()", text)

    def test_workflow_preserves_the_validated_environment(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", text)
        self.assertIn("project: flash-linear-attention", text)
        self.assertIn("linux-aarch64-a2-1", text)
        self.assertIn("cann:9.0.0-910b-ubuntu22.04-py3.11", text)
        self.assertIn("container_options: '--shm-size=16g'", text)
        self.assertIn("upstream_repo: fla-org/flash-linear-attention", text)
        self.assertIn("tests.flash_linear_attention.test_quick_start_ascend", text)

    def test_project_is_reachable_from_the_site_navigation(self) -> None:
        index = (ROOT / "index.rst").read_text(encoding="utf-8")

        self.assertIn("sources/flash-linear-attention/index.md", index)
        self.assertIn("sources/flash-linear-attention/quick_start.html", index)


if __name__ == "__main__":
    unittest.main()
