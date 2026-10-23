"""Quick-start test: doc under test is ``sources/torchvision/quick_start.md``.

Run: ``python -m unittest tests.torchvision.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``torchvision-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).

Scope note: the doc body covers the smoke path for **stock torchvision**
running under ``torch_npu`` PrivateUse1 dispatch. ``torch`` /
``torch_npu`` / ``pillow`` are installed by the doc body via ``uv pip
install`` (versions per the
[Ascend PyTorch Compatibility Matrix](https://gitcode.com/Ascend/pytorch/blob/main/COMPATIBILITY.en.md):
torch==2.14.0+cpu / torch_npu==2.14.0 / CANN==9.1.0) — ``torch``'s CPU
build only exists in the Aliyun ``pytorch-wheels/cpu`` find-links source
(PyPI's linux wheels are all CUDA builds), ``torch_npu`` comes from the
default index, which in CI is the in-cluster PyPI cache gateway. The
Aliyun *pypi simple* route must not be used for ``torch_npu``: it serves
at ~0.1 MiB/s, so the install stalls for tens of minutes. ``torchvision``
is installed **from source**: ``git clone`` pytorch/vision at the latest
release tag + ``FORCE_CUDA=0 uv pip install -e .`` — PyPI's linux wheels
all link ``libcudart.so`` and therefore conflict with ``torch_npu``, so a
CPU-only source build is the only compatible route. No Stable ABI patch:
torch 2.14.0's stable API already ships ``torch::stable::permute``
(which torchvision v0.29.0's ``deform_conv2d_kernel.cpp`` uses), so the
earlier torch-2.12-era transpose-chain patch is no longer needed.

The doc verifies:

* package + sub-module imports (``torchvision``, ``torchvision.transforms``,
  ``torchvision.transforms.v2``, ``torchvision.io``)
* transforms.v2 pipeline (synthetic image -> TVTensor -> transform) with
  NPU dispatch (``tensor.to('npu:0')``), which fails fast if the C++
  extensions did not register NPU kernels

Model download / ModelScope cache is **not** in scope — torchvision
transforms work on synthetic images, no checkpoints are fetched.
"""

from __future__ import annotations

import os
import subprocess
import unittest

os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """End-to-end test: fetch doc -> validate contract -> run ``#test-setup``
    / ``#test`` in order -> compare against ``#test-result``.

    ``prepare_environment`` only does runner-side scaffolding (CANN env +
    CUDA exclusion list + ``uv`` bootstrap); ``torch`` / ``torch_npu`` /
    ``pillow`` / ``torchvision`` are all installed by the doc's
    ``#test-setup`` / ``#test`` blocks, so the test exercises the exact
    install path users get.
    """

    # 60 min per command. The doc's install blocks normally finish in
    # ~15 s (torch from the Aliyun pytorch-wheels find-links source,
    # torch_npu from the default index = in-cluster cache gateway), and
    # the torchvision source build takes ~2 min; 60 min leaves room for
    # a cold cache or a slow mirror instead of failing the guard.
    DEFAULT_COMMAND_TIMEOUT = 3600

    USER_AGENT = 'ascend-docs/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',
    )

    # Process-level CUDA exclusion list. Same rationale as the accelerate
    # test: write to /tmp and export, so subprocesses (subprocess.run
    # inherits parent env by default) see it. torchvision's own wheel
    # resolver can transitively drag in CUDA wheels without this
    # constraint, and we never want a ``nvidia-cublas-cu12`` etc. to
    # sneak into a NPU run.
    _CUDA_CONSTRAINTS = (
        'cuda-toolkit<0',
        'cuda-python<0',
        'cuda-bindings<0',
        'cuda-core<0',
        'cuda-pathfinder<0',
        'flashinfer-python<0',
        'nvidia-cublas<0',
        'nvidia-cuda-runtime<0',
        'nvidia-cuda-nvrtc<0',
        'nvidia-cuda-cupti<0',
        'nvidia-cudnn<0',
        'nvidia-cudnn-frontend<0',
        'nvidia-cufft<0',
        'nvidia-curand<0',
        'nvidia-cusolver<0',
        'nvidia-cusparse<0',
        'nvidia-cusparselt<0',
        'nvidia-cutlass-dsl<0',
        'nvidia-cutlass-dsl-libs-base<0',
        'nvidia-cutlass-dsl-libs-core<0',
        'nvidia-cutlass-dsl-libs-cu12<0',
        'nvidia-ml-py<0',
        'nvidia-nccl<0',
        'nvidia-nvjitlink<0',
        'nvidia-nvtx<0',
        'nvidia-cublas-cu12<0',
        'nvidia-cuda-nvdisasm<0',
        'nvidia-cuda-runtime-cu12<0',
        'nvidia-cuda-nvrtc-cu12<0',
        'nvidia-cuda-cupti-cu12<0',
        'nvidia-cudnn-cu12<0',
        'nvidia-cufft-cu12<0',
        'nvidia-curand-cu12<0',
        'nvidia-cusolver-cu12<0',
        'nvidia-cusparse-cu12<0',
        'nvidia-cusparselt-cu12<0',
        'nvidia-nccl-cu12<0',
        'nvidia-nvjitlink-cu12<0',
        'nvidia-nvtx-cu12<0',
    )
    _CONSTRAINTS_FILE = '/tmp/torchvision_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image
    # (``image:`` input of ``torchvision-quick-start.yml``).
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write the CUDA exclusion list + install uv.

        ``torch`` / ``torch_npu`` / ``pillow`` / ``torchvision`` are
        intentionally NOT pre-installed here: the doc's ``#test-setup``
        blocks install them, so a broken install block fails loudly
        instead of being masked by a pre-installed copy.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # Don't overwrite envs explicitly injected by the
                # workflow (jobs.env / steps.env); only fill in CANN
                # keys that are missing, to avoid conflicts.
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) CUDA exclusion list + process-level env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: the doc body's install steps are all ``uv pip install ...``,
        # which gives clean stdout (no pip's "Obtaining / Installing
        # collected packages" preamble) so the #test-result fuzzy match
        # isn't polluted.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) uv request-level logging. uv prints NOTHING until the
        # resolve finishes, so a stalled download shows up as a silent
        # hang in the CI log. With RUST_LOG=uv=debug each HTTP request /
        # candidate wheel is logged to stderr, and the runner dumps the
        # captured stderr on timeout — the last logged URL is then the
        # hop that stalled.
        os.environ['RUST_LOG'] = 'uv=debug'

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY`` is
        unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class
        ``run_template()`` runs the full ``pre_process`` -> ``parse`` ->
        ``execute`` -> ``post_process`` flow. ``prepare_environment`` is
        triggered by ``setUpClass`` once, not from ``run_template``."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
