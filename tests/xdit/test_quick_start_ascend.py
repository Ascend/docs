"""Quick-start-Ascend test: doc under test is sources/xdit/quick_start.md."""

from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase, TestCommand
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


def _write_example_script(document: str) -> None:
    """Write the single reader-facing Python example into the test cwd."""
    blocks = re.findall(r"(?ms)^```python[ \t]*\r?\n(.*?)^```[ \t]*$", document)
    if len(blocks) != 1:
        raise AssertionError(f"expected one unlabeled Python example, found {len(blocks)}")
    script = blocks[0].rstrip() + "\n"
    compile(script, "sd3_npu.py", "exec")
    Path("sd3_npu.py").write_text(script, encoding="utf-8")


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
    _PROJECT_ROOT = Path("/root/xdit-test")
    _GENERATED_PNGS = {
        "xdit-sd3-smoke": Path("results/sd3_npu1_ulysses1.png"),
        "xdit-sd3-2card": Path("results/sd3_npu2_ulysses2.png"),
    }

    def _verify_generated_png(self, path: Path) -> None:
        """Keep the PNG integrity check in CI, not in the quick start."""
        if not path.is_file():
            raise AssertionError(f"generated image not found: {path}")
        image = path.read_bytes()
        if len(image) <= 50_000:
            raise AssertionError(
                "generated image is suspiciously small "
                f"({len(image)} bytes): {path}"
            )
        if image[:8] != b"\x89PNG\r\n\x1a\n":
            raise AssertionError(
                "generated image is not a PNG "
                f"(magic={image[:8]!r}): {path}"
            )
        self.log(
            f"[Step] verified generated PNG ({len(image)}B): "
            f"{path}"
        )

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if isinstance(cmd, TestCommand) and cmd.id in self._GENERATED_PNGS:
            super()._run_one(cmd, results, env, cwd, timeout, idx)
            self._verify_generated_png(self._GENERATED_PNGS[cmd.id])
            return
        return super()._run_one(cmd, results, env, cwd, timeout, idx)

    def pre_process(self) -> str:
        doc = Path(__file__).resolve().parent.parent.parent / "sources" / "xdit" / "quick_start.md"
        document = doc.read_text(encoding="utf-8")
        _write_example_script(document)
        return document

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

        # purge stale xfuser from the image so the doc install block really
        # installs the PyPI release instead of keeping a baked-in copy
        subprocess.run(["python", "-m", "pip", "uninstall", "-y", "xfuser"],
            capture_output=True, text=True, check=False)

        # Run the documented script and write results outside the repository.
        os.makedirs(cls._PROJECT_ROOT, exist_ok=True)
        os.chdir(cls._PROJECT_ROOT)
        print(f"setup: cwd -> {cls._PROJECT_ROOT}")

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
