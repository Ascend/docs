"""End-to-end test for the Ray Ascend NPU Quick Start."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase


def _e2e_enabled() -> bool:
    return os.environ.get("NPU_READY", "").strip().lower() == "true"


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1200
    USER_AGENT = "Ascend/docs/ray-quick-start"
    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    _CLUSTER_INDEX = (
        "http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
    )
    _ASCEND_EXTRA = "https://repo.huaweicloud.com/ascend/repos/pypi"
    _TORCH_VERSION = "2.9.0"
    _TORCH_NPU_VERSION = "2.9.0.post2"

    def pre_process(self) -> str:
        doc_path = (
            Path(__file__).resolve().parents[2]
            / "sources"
            / "Ray"
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

        os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1")
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        os.environ.setdefault("RAY_DEDUP_LOGS", "0")

        probe_script = (
            "import torch, torch_npu\n"
            "raise SystemExit(0 if "
            f"torch.__version__.startswith('{cls._TORCH_VERSION}') "
            f"and torch_npu.__version__.startswith('{cls._TORCH_NPU_VERSION}') "
            "else 1)"
        )
        probe = subprocess.run(
            ["python", "-c", probe_script],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            print(
                "setup: reusing torch "
                f"{cls._TORCH_VERSION} / torch_npu {cls._TORCH_NPU_VERSION}"
            )
        else:
            print(
                "setup: installing torch=="
                f"{cls._TORCH_VERSION} torch_npu=={cls._TORCH_NPU_VERSION}"
            )
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
                    f"torch=={cls._TORCH_VERSION}",
                    f"torch_npu=={cls._TORCH_NPU_VERSION}",
                ],
                check=True,
            )

        subprocess.run(
            [
                "python",
                "-c",
                "import torch, torch_npu; "
                "assert torch.npu.is_available(); "
                "assert torch.npu.device_count() >= 2; "
                "print(torch.__version__, torch_npu.__version__)",
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
