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
            {"check-python", "install-torch", "checkout-vbench", "install-vbench",
             "prepare-assets", "human-action-cli", "human-action-api"},
        )
        self.assertEqual(
            set(re.findall(r'#test-result id="([^"]+)"', text)),
            set(re.findall(r'#test id="([^"]+)"', text)),
        )
        self.assertEqual(text.count("输出结果如下："), 7)
        prerequisites = text.split("## 前置条件", 1)[1].split("## 检查并安装运行环境", 1)[0]
        self.assertNotIn("| VBench |", prerequisites)
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

    def test_ascend_branch_note_is_next_to_the_checkout_command(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        prerequisites = text.split("## 前置条件", 1)[1].split("## 检查并安装运行环境", 1)[0]
        checkout = text.split("## 获取", 1)[1].split("## 安装 VBench", 1)[0]
        self.assertNotIn("xxx 表示最新的版本号", text)
        self.assertNotIn("不要直接输入尖括号", text)
        self.assertNotIn("`<ref>`", prerequisites)
        note = "`<ref>` 默认填写 `master`"
        self.assertIn(note, checkout)
        self.assertIn("https://github.com/Ascend/VBench.git", checkout)
        self.assertLess(checkout.index("```{admonition} Note"), checkout.index('```shell #test id="checkout-vbench"'))

    def test_guide_uses_native_support(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        for obsolete in ("apply_patches.py", "git apply", "VBENCH_INSTALL_PROFILE",
                         "patches applied", "最新 release", "最新正式 release"):
            with self.subTest(obsolete=obsolete):
                self.assertNotIn(obsolete, text)

    def test_quick_start_installs_human_action_dependencies_before_vbench(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        install = text.split("## 安装 VBench", 1)[1].split("## 准备官方", 1)[0]
        command = "uv pip install --no-build-isolation --no-deps -e ./vbench-src"
        self.assertIn(command, install)
        dependencies = install.split(command, 1)[0].split("bash scripts/install_decord.sh", 1)[1]
        self.assertIn("uv pip install", dependencies)
        for dependency in ("opencv-python", "timm", "tqdm", "scipy", "gdown==6.4.2"):
            self.assertIn(dependency, dependencies)
        self.assertRegex(dependencies, r'"clip @ git\+https://github\.com/openai/CLIP\.git@[0-9a-f]{40}"')
        self.assertNotIn("openai-clip", dependencies)

    def test_cli_result_does_not_require_the_removed_patch_log(self) -> None:
        text = DOC.read_text(encoding="utf-8")
        result = text.split('```text #test-result id="human-action-cli"', 1)[1].split("```", 1)[0]
        self.assertNotIn("UMT device:", result)
        self.assertIn("done", result)
        guard = TEST.read_text(encoding="utf-8")
        self.assertIn("with check_api(human_action", guard)
        self.assertIn("self._checked_api(command.cmd)", guard)

    def test_guard_uses_the_npu_runtime_and_pinned_helpers(self) -> None:
        self.assertTrue(WORKFLOW.is_file())
        self.assertTrue(TEST.is_file())
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", workflow)
        self.assertIn("project: vbench", workflow)
        self.assertIn("linux-aarch64-a2-1", workflow)
        self.assertIn("cann:9.1.0-910b-ubuntu22.04-py3.12", workflow)
        self.assertIn("/data/ci-cache/vbench:/root/.cache/vbench", workflow)
        self.assertIn("upstream_repo: Ascend/VBench", workflow)
        self.assertIn("fixed_ref: master", workflow)
        self.assertIn("tests.vbench.test_quick_start_ascend", workflow)
        self.assertIn(HELPER_REF, TEST.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
