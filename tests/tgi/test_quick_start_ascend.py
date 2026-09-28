"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``sources/tgi/quick_start.md``
(follows the label contract: every ``shell`` code block carries one of
the ``#test`` / ``#test-setup`` / ``#test-result`` labels plus ``id=`` /
``store=`` / ``load='x>>y'`` / ``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.tgi.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``tgi-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; bash reads ``$UPSTREAM_REF`` to get
                                  the latest release tag. The value is
                                  captured into ``captures`` via the
                                  ``#test-setup store="upstream_ref"`` block's
                                  stdout, then substituted into the doc
                                  command body where ``<ref>`` appears.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from doc_test.base import MarkdownDocTestBase
from doc_test.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``quick_start.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``."""

    # The doc builds TGI from source (cargo release-opt of launcher + router
    # on aarch64), which dominates the wall clock; every subprocess shares
    # this timeout. 60 min is the ceiling for the build block.
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the cosdt fork
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
        'EJ0003',  # HCCL port bind failure - leftover TGI/hccl processes on the NPU
    )

    # Process-level CUDA exclusion list. TGI's dependency tree pulls
    # nvidia-* / cuda-* metapackages (and flashinfer) as transitive deps on
    # the default index; the runners are aarch64 without CUDA, so those
    # either fail to install or are dead weight. Written to /tmp and
    # exported; subprocesses (subprocess.run inherits parent env by
    # default) see it the same way.
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
    _CONSTRAINTS_FILE = '/tmp/tgi_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source
    # (fallback install path; the doc's own block installs torch/torch_npu
    # from the ascend source with an explicit --index-url).
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Two candidate paths exist across ascendhub images (toolkit layout vs
    # cann layout); the first one present wins.
    _CANN_SET_ENV_CANDIDATES = (
        '/usr/local/Ascend/ascend-toolkit/set_env.sh',
        '/usr/local/Ascend/cann/set_env.sh',
    )

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + torch stack probe
    # + safetensors + modelscope cache purge + cargo bin on PATH
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + install uv + torch
        stack probe + safetensors + purge stale modelscope cache shards +
        put ``~/.cargo/bin`` on PATH for the doc's rustup block.

        The doc's ``### Python 依赖`` / ``### 安装 Rust 工具链`` /
        ``## 4. 构建并安装 TGI`` sections are the single source of truth
        for which packages get installed; this class only handles
        ``torch_npu`` here (probe the image stack, install when absent),
        the defensive ``safetensors`` install, the modelscope cache
        purge, and PATH plumbing for the Rust toolchain installed by the
        doc itself.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Not the same as ``unittest.TestCase.setUp`` —
        that lifecycle hook fires before every test method, which is
        wrong for a one-shot install.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ
        sourced = False
        for candidate in cls._CANN_SET_ENV_CANDIDATES:
            if not os.path.isfile(candidate):
                continue
            merged = subprocess.run(
                ['bash', '-c', f'source {candidate} >/dev/null 2>&1; env'],
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
            print(f'setup: sourced CANN env from {candidate}')
            sourced = True
            break
        if not sourced:
            print(
                f'setup: skipping CANN env source (none of '
                f'{cls._CANN_SET_ENV_CANDIDATES} present)'
            )

        # 1) CUDA exclusion list + process-level env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: the doc's build/install blocks call uv pip install.
        # Inherit ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST`` from the yml
        # job-level env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe + install: reuse the image's pre-installed
        # torch_npu when importable; otherwise install the official
        # current recommended pair from the Ascend source (the doc's own
        # block installs torch_npu the same way, unpinned).
        _PROBE_SCRIPT = 'import torch, torch_npu'
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            check=False,  # probe's success/failure is the branch signal — don't raise
        )
        if probe.returncode == 0:
            _VERSIONS_SCRIPT = (
                'import torch, torch_npu; '
                'print(torch.__version__, torch_npu.__version__)'
            )
            versions = subprocess.run(
                ['python', '-c', _VERSIONS_SCRIPT],
                capture_output=True, text=True, check=True,
            )
            print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            print('setup: installing torch_npu from the Ascend source')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch_npu',
                ],
                check=True,
            )

        # 4) safetensors: native loader used by the cache validation
        # step below. Pulled in transitively by torch on most images;
        # install defensively in case the CANN base ships without it.
        ensure_safetensors()

        # 5) Cache validation: persistent host-side bind mount can hold
        # truncated safetensors from interrupted runs. Walk every shard
        # under each model dir and purge it on failure; modelscope
        # will re-download cleanly on next access. Implementation
        # lives in doc_test.model_cache; see that module's
        # docstring for the full rationale.
        purge_modelscope_corrupt(resolve_modelscope_cache())

        # 6) Rust toolchain PATH plumbing: the doc's rustup block installs
        # under ``~/.cargo/bin``. Each doc block runs in a fresh bash -c
        # with os.environ copied in, so cargo/rustc must be reachable via
        # the pre-merged PATH (rustup itself does not need them here —
        # only the later cargo build blocks do).
        cargo_bin = os.path.expanduser('~/.cargo/bin')
        if cargo_bin not in os.environ.get('PATH', ''):
            os.environ['PATH'] = f"{cargo_bin}:{os.environ.get('PATH', '')}"

        # 7) modelscope / transformers / accelerate / kernels are NOT
        # installed here — the doc's own blocks install them in document
        # order (modelscope in §2; the rest via the fork's
        # server/requirements_ascend.txt in §4).

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints + uv +
        torch stack + safetensors + modelscope cache purge + cargo PATH.

        ``transformers`` / ``accelerate`` / ``modelscope`` / ``kernels`` are
        NOT installed here — see ``prepare_environment`` for why.

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
