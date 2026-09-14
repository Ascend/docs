"""Quick-start-Ascend test: doc under test is sources/xdit/quick_start.md."""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase
from doc_test.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == "true"


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get("NPU_READY"))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """SD3 medium smoke on 1 card + 2-card Ulysses parallel (NPU/hccl)."""

    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = "cosdt-ci-test/quick-start"
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
    )

    _CUDA_CONSTRAINTS = (
        "cuda-toolkit<0", "cuda-python<0", "cuda-bindings<0", "cuda-core<0", "cuda-pathfinder<0",
        "flashinfer-python<0", "nvidia-cublas<0", "nvidia-cuda-runtime<0", "nvidia-cuda-nvrtc<0",
        "nvidia-cuda-cupti<0", "nvidia-cudnn<0", "nvidia-cudnn-frontend<0", "nvidia-cufft<0",
        "nvidia-curand<0", "nvidia-cusolver<0", "nvidia-cusparse<0", "nvidia-cutlass-dsl<0",
        "nvidia-cutlass-dsl-libs-base<0", "nvidia-cutlass-dsl-libs-core<0", "nvidia-cutlass-dsl-libs-cu12<0",
        "nvidia-ml-py<0", "nvidia-nccl<0", "nvidia-nvjitlink<0", "nvidia-nvtx<0",
        "nvidia-cublas-cu12<0", "nvidia-cuda-nvdisasm<0", "nvidia-cuda-runtime-cu12<0", "nvidia-cuda-nvrtc-cu12<0",
        "nvidia-cuda-cupti-cu12<0", "nvidia-cudnn-cu12<0", "nvidia-cufft-cu12<0", "nvidia-curand-cu12<0",
        "nvidia-cusolver-cu12<0", "nvidia-cusparse-cu12<0", "nvidia-cusparselt-cu12<0", "nvidia-nccl-cu12<0",
        "nvidia-nvjitlink-cu12<0", "nvidia-nvtx-cu12<0",
    )
    _CONSTRAINTS_FILE = "/tmp/xdit_npu_constraints.txt"
    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"

    def pre_process(self) -> str:
        doc = Path(__file__).resolve().parent.parent.parent / "sources" / "xdit" / "quick_start.md"
        return doc.read_text(encoding="utf-8")

    @classmethod
    def prepare_environment(cls) -> None:
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ["bash", "-c", f"source {cls._CANN_SET_ENV} >/dev/null 2>&1; env"],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key, value)
            print("setup: sourced CANN env from set_env.sh")
        else:
            print(f"setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)")

        with open(cls._CONSTRAINTS_FILE, "w", encoding="utf-8") as f:
            f.write(chr(10).join(cls._CUDA_CONSTRAINTS) + chr(10))
        os.environ["PIP_CONSTRAINT"] = cls._CONSTRAINTS_FILE
        os.environ["UV_CONSTRAINT"] = cls._CONSTRAINTS_FILE

        subprocess.run(["python", "-m", "pip", "install", "uv"], check=True)

        subprocess.run(["uv", "pip", "uninstall", "-y", "xfuser"],
            capture_output=True, text=True, check=False)

        project_root = Path(__file__).resolve().parent.parent.parent / "sources" / "xdit"
        os.chdir(str(project_root))
        print(f"setup: cwd -> {project_root}")

        ps = "import torch, torch_npu\nraise SystemExit(0 if torch.npu.is_available() else 1)\n"
        probe = subprocess.run(["python", "-c", ps], capture_output=True, check=False)
        if probe.returncode == 0:
            vs = subprocess.run(["python", "-c", "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"],
                capture_output=True, text=True, check=True)
            print(f"setup: reusing image torch stack ({vs.stdout.strip()})")
        else:
            print("setup: torch probe failed, doc install-torch will install the pinned stack")

        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

        ensure_safetensors()
        try:
            purge_modelscope_corrupt(resolve_modelscope_cache())
        except Exception as e:
            print(f"setup: cache purge skipped ({e})")

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        "end-to-end requires NPU runner; set NPU_READY=true",
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == "__main__":
    unittest.main()
