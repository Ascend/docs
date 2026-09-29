"""Quick-start test: doc under test is ``sources/torchtune/quick_start.md``.

End-to-end case built on top of the ``MarkdownDocTestBase`` contract:
every ``shell`` / ``python`` code block carries one of the ``#test`` /
``#test-setup`` / ``#test-result`` labels plus ``id=`` / ``store=`` /
``load='x>>y'`` / ``fuzzy='xxx'`` parameters.

Run: ``python -m unittest tests.torchtune.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the ``quick-start-template.yml``
engine that ``torchtune-quick-start.yml`` delegates to):
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
    purge_huggingface_corrupt,
    report_huggingface_state,
    resolve_huggingface_cache,
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
    ``#test-result``.

    Scope: single-card LoRA finetune of Qwen2.5-0.5B-Instruct - install
    torchtune (binary + source paths), ``tune run lora_finetune_single_device``
    for 3 steps, then verify the LoRA adapter checkpoint on disk. The base
    model is downloaded from HuggingFace Hub by the doc's own
    ``snapshot_download`` block.

    The test subclass itself does not own any ``test_*`` method beyond the
    template-method entry; the doc body is the spec. ``prepare_environment``
    makes sure ``torch_npu`` is importable + the cluster ``pip`` mirror is
    bound before the framework starts executing doc commands (the doc
    body itself pins ``torch`` / ``torch_npu`` and runs ``uv pip install
    torchtune`` on a ``<ref>`` checkout, but the baseline stack still
    needs to be on the runner for the doc to do anything useful).
    """

    # 60 min per command: long enough for the ~1 GB Qwen2.5-0.5B
    # snapshot_download + a 3-step tune run (cold cache + first NPU
    # compile); short enough to fail fast on hangs.
    DEFAULT_COMMAND_TIMEOUT = 3600

    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. Same rationale as the other
    # projects' tests: write to /tmp and export, so subprocesses
    # (subprocess.run inherits parent env by default) see it. torchtune's
    # own dependency tree pulls `datasets` + `huggingface_hub` which can
    # transitively drag in CUDA wheels without this constraint.
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
    _CONSTRAINTS_FILE = '/tmp/torchtune_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # Base model pulled by the doc's snapshot_download block.
    _MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``torchtune-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv + torch stack probe
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + CUDA constraints + uv + torch stack probe.

        torchtune itself is intentionally NOT pre-installed here: the doc
        body runs ``uv pip install torchtune`` (binary path) and
        ``uv pip install .`` (source path) against the upstream release
        tag injected by the workflow, so the test exercises the exact
        install path users get.

        What this hook does pre-install:
            * CANN env (so torch_npu is importable);
            * CUDA exclusion list (so an accidental ``nvidia-cudnn`` pull
              doesn't shadow the NPU build);
            * torch / torch_npu, only if the image's pre-installed wheels
              don't already match the version matrix (the doc's own
              ``#test-setup`` blocks then pin the versions it needs);
            * HF hub cache sanity: report + purge corrupt safetensors so a
              dirty leftover cache self-heals instead of failing the run.
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
                # workflow; only fill in CANN keys that are missing.
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

        # 2) uv: the doc body's binary install path passes
        # ``--index-url https://mirrors.aliyun.com/pypi/simple`` explicitly
        # to dodge the cluster cache (the cluster PyPI mirror doesn't
        # always have a fresh torchtune wheel on release day), so the
        # ``PIP_INDEX_URL`` / ``UV_INDEX_URL`` job-level env in
        # ``quick-start-template.yml`` is overridden for that command and
        # doesn't apply here. uv's output is also cleaner than pip's
        # "Obtaining/Installing collected packages" preamble so the
        # #test-result fuzzy match against ``torchtune xxx`` isn't
        # polluted.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: when version matches the image's
        # pre-installed wheels, reuse them to avoid the cluster cache
        # triggering ``+cpu`` resolution. The doc's own #test-setup
        # blocks pin the final torch / torch_npu versions it needs.
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.9.0') "
            "and torch_npu.__version__.startswith('2.9.0') "
            "else 1)"
        )
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
            print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch==2.9.0', 'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        # 4) ``huggingface_hub`` / ``torchao`` / ``torchtune`` are NOT
        # pre-installed here: the doc's own labeled blocks install them in
        # document order (``install-deps`` for the hub + torchao,
        # ``torchtune-install-binary`` / ``torchtune-install-source`` for
        # torchtune itself), so a broken install block fails loudly as a
        # fuzzy mismatch instead of being masked by a pre-installed copy.

        # tqdm: required by huggingface_hub's download progress bars.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'tqdm'],
            check=True,
        )

        # 5) HF hub cache sanity: the doc downloads Qwen2.5-0.5B-Instruct
        # via ``huggingface_hub.snapshot_download`` into the default HF
        # cache. Report what's already there and purge model dirs holding
        # corrupt safetensors shards so the download self-heals.
        ensure_safetensors()
        report_huggingface_state(cls._MODEL_ID)
        purge_huggingface_corrupt(resolve_huggingface_cache())

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints + uv + torch stack.

        ``huggingface_hub`` / ``torchao`` / ``torchtune`` are NOT installed
        here — the doc's own labeled blocks install them in document order,
        so a broken install block fails loudly instead of being masked.

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
