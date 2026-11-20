"""Execute the VBench human_action guide with pinned helper scripts."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from textwrap import dedent, indent

from doc_test.base import MarkdownDocTestBase, TestCommand


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "sources" / "vbench" / "quick_start.md"
HELPER_REF = "abaa5c1610c57f34eeaa73208d91b6a9b1573a79"
HELPER_URL = "https://github.com/cosdt-ci-test/workflows.git"


def _e2e_enabled() -> bool:
    return os.environ.get("NPU_READY", "").strip().lower() == "true"


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = "Ascend/docs/vbench-quick-start"
    ERROR_MARKERS = (*MarkdownDocTestBase.ERROR_MARKERS, "ERR99999")
    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    _project_root: Path | None = None

    def pre_process(self) -> str:
        if not DOC.is_file():
            raise RuntimeError(f"VBench Quick Start not found: {DOC}")
        return DOC.read_text(encoding="utf-8")

    @staticmethod
    def _checked_api(body: str) -> str:
        """Keep device and result assertions outside the user's API example."""
        prefix = dedent('''\
            import os
            from pathlib import Path
            import torch_npu
            import vbench.human_action as human_action
            from scripts.prepare_assets import UMT_FILENAME
            from scripts.runtime_checks import check_api

            cache = Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache/vbench"))
            with check_api(human_action, cache / "umt_model" / UMT_FILENAME):
        ''')
        return prefix + indent(body, "    ")

    @classmethod
    def prepare_environment(cls) -> None:
        if not Path(cls._CANN_SET_ENV).is_file():
            raise RuntimeError(f"CANN environment script not found: {cls._CANN_SET_ENV}")
        merged = subprocess.run(
            ["bash", "-c", f"source {cls._CANN_SET_ENV} >/dev/null 2>&1; env"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                os.environ.setdefault(key, value)

        helper_root = Path(tempfile.mkdtemp(prefix="vbench-doc-helpers-")) / "workflows"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
             HELPER_URL, str(helper_root)],
            check=True,
        )
        for args in (
            ["sparse-checkout", "init", "--cone"],
            ["sparse-checkout", "set", "projects/vbench", "cache-seed/vbench", "scripts"],
            ["fetch", "--depth", "1", "origin", HELPER_REF],
            ["checkout", "--detach", "FETCH_HEAD"],
        ):
            subprocess.run(["git", "-C", str(helper_root), *args], check=True)
        cls._project_root = helper_root / "projects" / "vbench"
        if not (cls._project_root / "scripts" / "prepare_assets.py").is_file():
            raise RuntimeError("Pinned VBench helper checkout is incomplete")

        os.environ["PIP_CONSTRAINT"] = str(cls._project_root / "constraints-npu.txt")
        os.environ["UV_CONSTRAINT"] = str(cls._project_root / "constraints-npu.txt")
        os.environ["UV_SYSTEM_PYTHON"] = "1"
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
        os.environ.setdefault("VBENCH_CACHE_DIR", "/root/.cache/vbench")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    def test_document_contract(self) -> None:
        commands, results = self.parse(self.pre_process())
        self.assertEqual(len(results), 7)
        self.assertEqual(
            {command.id for command in commands if isinstance(command, TestCommand)},
            set(results),
        )

    @unittest.skipIf(not _e2e_enabled(), "requires an Ascend NPU runner; set NPU_READY=true")
    def test_runs_doc(self) -> None:
        if self._project_root is None:
            raise RuntimeError("VBench helpers were not prepared")
        commands, results = self.parse(self.pre_process())
        api = [command for command in commands if isinstance(command, TestCommand)
               and command.id == "human-action-api"]
        if len(api) != 1 or api[0].language != "python":
            raise ValueError("Expected exactly one public VBench Python API example")
        checked = [replace(command, cmd=self._checked_api(command.cmd))
                   if command is api[0] else command for command in commands]
        original_cwd = Path.cwd()
        try:
            os.chdir(self._project_root)
            self.execute(checked, results)
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
