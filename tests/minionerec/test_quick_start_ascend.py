"""Quick-start test: doc under test is ``sources/minionerec/quick_start.md``.

End-to-end case built on top of the ``MarkdownDocTestBase`` contract:
clone ``AkaliKong/MiniOneRec`` -> install the NPU torch stack and the
SFT / evaluation / RL dependencies -> verify the upstream ``npu_support``
shim maps ``torch.cuda`` onto NPU -> download Qwen2.5-0.5B and the
preprocessed Amazon Industrial_and_Scientific subset from HuggingFace ->
one SFT epoch -> constrained-decoding evaluation (CC must be 0) -> one
GRPO epoch.

Upstream PR #85 adds ``npu_support.py`` and imports it first in every
entry script, so no env-level injection or source edit is needed;
``rl.py`` picks ``adamw_torch`` on NPU and ``--sample 40`` caps every
dataset. The doc's ``git status --porcelain`` check asserts the checkout
stays clean.

At this scale every reward stays 0: ``rule_reward`` is exact match, one
SFT epoch never hits, and ``ndcg_rule_reward`` zeroes the whole group
when nothing hits - so the group advantage and the gradient are 0 too.
The RL stage therefore guards flow-through only: ``run-rl`` must exit 0
and ``rl.py``'s closing ``save_model`` / ``model.save_pretrained`` must
leave weights, config and tokenizer under ``rl_out/final_checkpoint``.

Run: ``python -m unittest tests.minionerec.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the engine
``quick-start-template.yml``, triggered by ``minionerec-quick-start.yml``):
    ``MONITORED_DOC_URL``         doc to fetch and execute
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU
                                  runner: local dev machines / normal ubuntu
                                  runners have no ``/dev/davinci*`` device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from pathlib import Path

import requests

from doc_test.base import MarkdownDocTestBase, SetupCommand


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``sources/minionerec/quick_start.md`` end-to-end test: fetch doc ->
    validate contract -> run ``#test-setup`` / ``#test`` in order ->
    compare against ``#test-result``.

    Working directory is pinned to ``/root/minionerec-test``: the doc is
    cwd-relative (clones to ``./MiniOneRec``, downloads into
    ``./MiniOneRec/data``), so the clone tree must not collide with the
    ``workflows`` checkout the engine runs the test from.
    """

    # SFT (~11 min) and RL (~14 min) dominate; 1h per command leaves
    # headroom for a cold HuggingFace download.
    DEFAULT_COMMAND_TIMEOUT = 3600

    USER_AGENT = 'ascend-docs/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a
    # CANN runtime failure surfaces a full stderr dump instead of a
    # head/tail window that hides the failing line.
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    _CANN_SET_ENV_CANDIDATES = (
        '/usr/local/Ascend/ascend-toolkit/set_env.sh',
        '/usr/local/Ascend/cann/set_env.sh',
    )

    # The A3 writable layer is small and quota-limited (the RL checkpoint
    # write dies with EDQUOT), while ``/root/.cache`` is the volume the A3
    # pool shares across runs. Keep this leg's workdir on the shared volume.
    _WORK_DIR = '/root/minionerec-test'
    _A3_WORK_DIR = '/root/.cache/minionerec-test'

    # Base model, fetched from the ModelScope mirror on A3 before the
    # document's download block runs.
    #
    # Why: hf-mirror serves nine of the ten ``Qwen/Qwen2.5-0.5B`` files itself,
    # but redirects the 988 MB ``model.safetensors`` to ``cas-bridge.xethub.hf.co``
    # (Xet). On the A3 pool that redirected HEAD comes back without a
    # ``Content-Length``, and ``snapshot_download`` refuses any file whose size
    # it cannot read - a hard failure that its own retries do not cover. With
    # the files already on disk, ``hf_hub_download`` falls back to them instead.
    # The document keeps the HuggingFace step: readers are unaffected.
    _MS_MODEL_API = 'https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B'
    _BASE_MODEL = 'Qwen/Qwen2.5-0.5B'

    # ----------------------------------------------------------
    # prepare_environment: PATH + CANN env + cwd
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """CANN env, ``git`` and the working directory.

        Everything business-facing is in the document: the dependency
        install, the shim, the model and data download, SFT, evaluation
        and RL are the steps a reader follows.
        """
        # Keep one visible card: the A3 runner exposes two.
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')
        # A3 runners reach Hugging Face only through the mirror (direct
        # access returns 503); A2 dials huggingface.co directly. Xet off on
        # A3: its data plane lives on cas-bridge.xethub.hf.co, outside
        # HF_ENDPOINT.
        if os.environ.get('MACHINE_TYPE') == 'a3':
            os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
            os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

        # 0) PATH: /usr/local/sbin for npu-smi.
        current_path = os.environ.get('PATH', '')
        prefix_dirs = '/usr/local/sbin:/usr/local/bin'
        if not current_path.startswith(prefix_dirs):
            os.environ['PATH'] = f'{prefix_dirs}:{current_path}'

        # 1) CANN env: source set_env.sh and merge the env stream into
        # os.environ so every doc block's subprocess inherits
        # LD_LIBRARY_PATH / ASCEND_* without re-sourcing.
        sourced = False
        for set_env in cls._CANN_SET_ENV_CANDIDATES:
            if os.path.isfile(set_env):
                merged = subprocess.run(
                    ['bash', '-c', f'source {set_env} >/dev/null 2>&1; env'],
                    capture_output=True, text=True, check=True,
                )
                for line in merged.stdout.splitlines():
                    if '=' not in line:
                        continue
                    key, _, value = line.partition('=')
                    # Don't overwrite what the workflow injected.
                    os.environ.setdefault(key, value)
                print(f'setup: sourced CANN env from {set_env}')
                sourced = True
                break
        if not sourced:
            print(
                'setup: no CANN set_env.sh found (expected '
                f'{cls._CANN_SET_ENV_CANDIDATES}) - every step will fail'
            )

        # 2) git: the engine bootstraps it, but a missing git turns the
        # clone into an opaque `command not found`.
        if shutil.which('git') is None:
            print('setup: installing git')
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'git'], check=True)

        # 3) Working directory: pinned before run_template() so the doc's
        # cwd-relative clone never lands in the guard checkout.
        work_dir = cls._WORK_DIR
        if os.environ.get('MACHINE_TYPE') == 'a3':
            # Leftovers from an earlier run survive on the shared volume and
            # the doc clones into a fresh directory, so wipe it first.
            work_dir = cls._A3_WORK_DIR
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        # The A3 writable layer is quota-limited, so print what is actually
        # available: a checkpoint write that dies with EDQUOT is otherwise
        # indistinguishable from a network or model failure.
        usage = shutil.disk_usage(work_dir)
        print(
            f'setup: working dir {work_dir}, '
            f'total {usage.total / 2 ** 30:.0f} GiB, free {usage.free / 2 ** 30:.1f} GiB'
        )

    # ----------------------------------------------------------
    # A3 only: seed the base model before the document downloads it
    # ----------------------------------------------------------

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        """Seed ``Qwen/Qwen2.5-0.5B`` ahead of the document's download step.

        ``#test-setup`` blocks carry no ``id``, so the block is recognised by
        what it does: pull the base model from the Hub.
        """
        if (
            os.environ.get('MACHINE_TYPE') == 'a3'
            and isinstance(cmd, SetupCommand)
            and 'snapshot_download' in cmd.cmd
            and self._BASE_MODEL in cmd.cmd
        ):
            self._seed_base_model(Path(cwd))
        return super()._run_one(cmd, results, env, cwd, timeout, idx)

    def _seed_base_model(self, cwd: Path) -> None:
        """Download the base model from ModelScope into the cloned tree."""
        target = cwd / 'MiniOneRec' / 'Qwen2.5-0.5B'
        listing = requests.get(
            f'{self._MS_MODEL_API}/repo/files',
            params={'Revision': 'master', 'Recursive': 'true'},
            timeout=120,
        )
        listing.raise_for_status()
        files = [
            entry['Path']
            for entry in listing.json()['Data']['Files']
            if entry.get('Type') == 'blob'
        ]
        print(f'seed: {len(files)} files from ModelScope -> {target}')
        for name in files:
            dest = target / name
            if dest.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(
                f'{self._MS_MODEL_API}/repo',
                params={'Revision': 'master', 'FilePath': name},
                stream=True,
                timeout=1800,
            ) as response:
                response.raise_for_status()
                with open(dest, 'wb') as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
            print(f'seed: {name} ({dest.stat().st_size} bytes)')

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class.

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
