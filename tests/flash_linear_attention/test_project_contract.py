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
        self.assertTrue(text.startswith("# flash-linear-attention\n"))
        self.assertNotIn("python - <<'PY'", text)
        self.assertIn('```python #test id="check-npu"', text)
        self.assertIn('```python #test id="gdn-forward-backward"', text)
        self.assertEqual(text.count("输出结果如下：\n\n```text #test-result"), len(test_ids))
        self.assertIn("torch xxx+cpu", text)

    def test_workflow_preserves_the_validated_environment(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", text)
        self.assertIn("project: flash-linear-attention", text)
        self.assertIn("linux-aarch64-a2-1", text)
        self.assertIn("cann:9.0.0-910b-ubuntu22.04-py3.11", text)
        self.assertIn("container_options: '--shm-size=16g'", text)
        self.assertIn("upstream_repo: fla-org/flash-linear-attention", text)
        self.assertIn("tests.flash_linear_attention.test_quick_start_ascend", text)

    def test_document_names_the_workflow_image_and_verified_npu_stack(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        document = DOC.read_text(encoding="utf-8")
        image_match = re.search(r"(?m)^\s+image: (\S+)$", workflow)
        self.assertIsNotNone(image_match)
        image = image_match.group(1)

        self.assertIn(f"`{image}`", document)
        cann_version = re.search(r"cann:(\d+\.\d+\.\d+)", image).group(1)
        self.assertIn(f"| CANN | {cann_version} |", document)
        self.assertIn("| flash-linear-attention | 最新 release |", document)
        for package in ("torch", "torch_npu", "torchvision", "triton-ascend"):
            self.assertRegex(document, rf"(?m)^\| {package} \| \d+\.\d+")
        self.assertRegex(
            document,
            r"https://github.com/fla-org/flash-linear-attention/blob/v\d+\.\d+\.\d+/pyproject\.toml",
        )

    def test_project_is_reachable_from_the_site_navigation(self) -> None:
        index = (ROOT / "index.rst").read_text(encoding="utf-8")
        project_index = ROOT / "sources" / "flash-linear-attention" / "index.rst"

        self.assertIn("sources/flash-linear-attention/index.rst", index)
        self.assertIn('href="sources/flash-linear-attention/"', index)
        self.assertEqual(
            project_index.read_text(encoding="utf-8"),
            ".. include:: quick_start.md\n   :parser: myst_parser.sphinx_\n",
        )


if __name__ == "__main__":
    unittest.main()
