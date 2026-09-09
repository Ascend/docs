"""End-to-end test for the open_clip Ascend Quick Start document."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase


def _e2e_enabled() -> bool:
    return os.environ.get("NPU_READY", "").strip().lower() == "true"


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = "Ascend/docs/open-clip-quick-start"
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        "applicaiton exception",
        "ERR99999",
    )

    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    _CLUSTER_INDEX = (
        "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
    )
    _ASCEND_EXTRA = "https://repo.huaweicloud.com/ascend/repos/pypi"
    _CONSTRAINTS_FILE = Path(__file__).resolve().parent / "constraints-npu.txt"

    def pre_process(self) -> str:
        doc_path = (
            Path(__file__).resolve().parents[2]
            / "sources"
            / "open_clip"
            / "quick_start.md"
        )
        if not doc_path.is_file():
            raise RuntimeError(f"Quick Start document not found: {doc_path}")
        return doc_path.read_text(encoding="utf-8")

    @classmethod
    def prepare_environment(cls) -> None:
        if not os.path.isfile(cls._CANN_SET_ENV):
            raise RuntimeError(f"CANN environment script not found: {cls._CANN_SET_ENV}")

        merged = subprocess.run(
            ["bash", "-c", f"source {cls._CANN_SET_ENV} >/dev/null 2>&1; env"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key, value)

        os.environ["PIP_CONSTRAINT"] = str(cls._CONSTRAINTS_FILE)
        os.environ["UV_CONSTRAINT"] = str(cls._CONSTRAINTS_FILE)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        subprocess.run(["python", "-m", "pip", "install", "uv"], check=True)

        probe = subprocess.run(
            [
                "python",
                "-c",
                "import torch, torch_npu; "
                "raise SystemExit(0 if "
                "torch.__version__.startswith('2.9.0') and "
                "torch_npu.__version__.startswith('2.9.0') else 1)",
            ],
            capture_output=True,
            check=False,
        )
        if probe.returncode != 0:
            print("setup: installing torch==2.9.0 torch_npu==2.9.0.post2")
            subprocess.run(
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--index-url",
                    cls._CLUSTER_INDEX,
                    "--extra-index-url",
                    cls._ASCEND_EXTRA,
                    "torch==2.9.0",
                    "torch_npu==2.9.0.post2",
                ],
                check=True,
            )
        else:
            print("setup: reusing torch 2.9 / torch_npu 2.9 stack")

        subprocess.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "torchvision==0.24.0",
            ],
            check=True,
        )

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        "end-to-end requires an Ascend NPU runner; set NPU_READY=true",
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == "__main__":
    unittest.main()
