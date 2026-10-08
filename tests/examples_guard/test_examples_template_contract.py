"""Contract tests for the shared examples guard engine and thin triggers.

The executed tests run the decide step's bash exactly as each event
would: a PR run that skips the matrix reads green without testing
anything, so the PR branch is pinned here.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / '.github' / 'workflows'
TEMPLATE = WORKFLOWS / 'examples-template.yml'
PEFT_TRIGGER = WORKFLOWS / 'peft-examples.yml'
ENGINE_USE = 'uses: ./.github/workflows/examples-template.yml'


def _load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


# Engine triggers that opt in to PR runs; PyYAML reads the bare `on`
# key as boolean True.
PR_TRIGGERS = sorted(
    path for path in WORKFLOWS.glob('*-examples.yml')
    if ENGINE_USE in path.read_text(encoding='utf-8')
    and 'pull_request' in _load_workflow(path)[True])

RENDER_MAP = {
    '${{ steps.monitor.outputs.need_to_run }}': '',
    '${{ steps.monitor.outputs.release_ref }}': '',
    '${{ steps.monitor.outputs.reason }}': '',
}

CURL_STUB = '''#!/bin/sh
out=
url=
while [ $# -gt 0 ]; do
  case "$1" in
    -o|-w|--retry) [ "$1" = -o ] && out="$2"; shift 2 ;;
    -H) printf '%s\\n' "$2" >> "$CURL_LOG"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
printf '%s\\n' "$url" >> "$CURL_LOG"
case "$url" in
  */releases/latest) body="$RELEASE_BODY"; code="$RELEASE_CODE" ;;
  */commits/*) body="$COMMIT_BODY"; code="$COMMIT_CODE" ;;
  *) body=; code=500 ;;
esac
printf '%s' "$body" > "$out"
echo "$code"
'''


def _modern_bash() -> str | None:
    """Return a bash >= 4, like the one on the Actions runner.

    macOS /bin/bash 3.2 lets a declared-but-unset local pass `set -u`
    where bash 5 aborts, so running the decide block under 3.2 hides
    that class of bug.
    """
    for candidate in (shutil.which('bash'), '/bin/bash'):
        if not candidate:
            continue
        probe = subprocess.run(
            [candidate, '-c', 'echo "${BASH_VERSINFO[0]}"'],
            capture_output=True, text=True, check=False)
        major = probe.stdout.strip()
        if major.isdigit() and int(major) >= 4:
            return candidate
    return None


BASH = _modern_bash()


def _decide_run_block() -> str:
    text = TEMPLATE.read_text(encoding='utf-8')
    step = text[text.index('      - name: Decide target'):]
    step = step[:step.index('\n  manifest-check:')]
    return step.split('        run: |\n', 1)[1]


def _render(text: str) -> str:
    for expression, value in RENDER_MAP.items():
        text = text.replace(expression, value)
    if '${{' in text:
        raise AssertionError('decide grew an unrendered expression')
    return text


class EngineContractTests(unittest.TestCase):

    def test_pull_request_events_force_the_full_matrix(self) -> None:
        text = TEMPLATE.read_text(encoding='utf-8')
        self.assertIn('elif [ "$EVENT_NAME" = "pull_request" ]', text)
        self.assertIn('trigger=pull_request', text)
        self.assertIn('need_to_run=true', text)

    def test_monitor_state_is_schedule_only(self) -> None:
        text = TEMPLATE.read_text(encoding='utf-8')
        # restore / verify / monitor steps plus the save-monitor-state job.
        self.assertEqual(text.count("github.event_name == 'schedule'"), 4)
        self.assertIn("github.event_name == 'schedule' &&", text)

    def test_results_publish_from_github_hosted_runners(self) -> None:
        job = _load_workflow(TEMPLATE)['jobs']['validate-results']
        self.assertEqual(job['runs-on'], 'ubuntu-latest')
        text = TEMPLATE.read_text(encoding='utf-8')
        self.assertIn('python3 workflows/scripts/write_example_result.py', text)
        self.assertIn(
            '${{ inputs.project }}-examples-${{ github.run_id }}'
            '-${{ strategy.job-index }}', text)

    def test_every_api_call_carries_the_job_token(self) -> None:
        text = TEMPLATE.read_text(encoding='utf-8')
        api_calls = text.count('"https://api.github.com/')
        self.assertGreater(api_calls, 0)
        self.assertEqual(
            text.count('-H "Authorization: Bearer $GH_TOKEN"'), api_calls)
        monitor_env = _load_workflow(TEMPLATE)['jobs']['monitor']['env']
        self.assertEqual(monitor_env['GH_TOKEN'], '${{ github.token }}')


class TriggerContractTests(unittest.TestCase):

    def test_peft_opts_in_to_pr_runs(self) -> None:
        self.assertIn(PEFT_TRIGGER, PR_TRIGGERS)

    def test_pull_request_concurrency_is_namespaced_per_workflow(self) -> None:
        for trigger in PR_TRIGGERS:
            with self.subTest(trigger=trigger.name):
                text = trigger.read_text(encoding='utf-8')
                self.assertIn(
                    f"format('{trigger.stem}-pr-{{0}}', "
                    'github.event.pull_request.number)', text)
                self.assertIn(
                    "cancel-in-progress: ${{ github.event_name == "
                    "'pull_request' }}", text)

    def test_pull_request_paths_are_project_scoped(self) -> None:
        for trigger in PR_TRIGGERS:
            with self.subTest(trigger=trigger.name):
                workflow = _load_workflow(trigger)
                pull_request = workflow[True]['pull_request']
                self.assertEqual(
                    pull_request['types'], ['opened', 'synchronize', 'reopened'])
                project = next(iter(workflow['jobs'].values()))['with']['project']
                own_trigger = f'.github/workflows/{trigger.name}'
                self.assertEqual(
                    sorted(pull_request['paths']),
                    sorted([f'examples-guards/{project}/**', own_trigger]))

    def test_peft_schedule_stays_disabled_during_bring_up(self) -> None:
        text = PEFT_TRIGGER.read_text(encoding='utf-8')
        self.assertNotIn('\n  schedule:\n', text)
        self.assertIn('# schedule:', text)


@unittest.skipUnless(BASH and shutil.which('jq'),
                     'needs bash >= 4 and jq, as on the Actions runner')
class DecideExecutionTests(unittest.TestCase):
    """Execute the decide run block under each event's conditions."""

    def _run_decide(self, env_overrides: dict) -> tuple[dict, str, int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub = root / 'curl'
            stub.write_text(CURL_STUB)
            stub.chmod(0o755)
            script = root / 'decide.sh'
            script.write_text(_render(_decide_run_block()))
            output = root / 'github_output'
            curl_log = root / 'curl.log'
            env = {
                'PATH': f'{root}:{os.environ.get("PATH", "")}',
                'TMPDIR': str(root),
                'EVENT_NAME': 'pull_request',
                'GITHUB_OUTPUT': str(output),
                'GH_TOKEN': 'test-token',
                'INPUT_REF': '',
                'INPUT_REPO': '',
                'DEFAULT_BRANCH': 'main',
                'UPSTREAM_REPO': 'huggingface/peft',
                'RELEASE_BODY': '{"tag_name": "v0.20.0"}',
                'RELEASE_CODE': '200',
                'COMMIT_BODY': '{"sha": "abc123"}',
                'COMMIT_CODE': '200',
                'CURL_LOG': str(curl_log),
            }
            env.update(env_overrides)
            result = subprocess.run(
                [BASH, str(script)], env=env,
                capture_output=True, text=True, check=False)
            outputs: dict = {}
            if output.exists():
                for line in output.read_text(
                        encoding='utf-8').splitlines():
                    if '=' in line:
                        key, value = line.split('=', 1)
                        outputs[key] = value
            calls = curl_log.read_text() if curl_log.exists() else ''
            return outputs, result.stdout, result.returncode, calls

    def test_pull_request_resolves_latest_release(self) -> None:
        outputs, stdout, code, calls = self._run_decide({})
        self.assertEqual(code, 0, stdout)
        self.assertEqual(outputs, {
            'need_to_run': 'true',
            'trigger': 'pull_request',
            'target_repo': 'huggingface/peft',
            'target_ref': 'v0.20.0',
        })
        self.assertIn('Authorization: Bearer test-token', calls)

    def test_pull_request_without_releases_tests_default_branch_head(self) -> None:
        outputs, stdout, code, calls = self._run_decide(
            {'RELEASE_BODY': '', 'RELEASE_CODE': '404'})
        self.assertEqual(code, 0, stdout)
        self.assertEqual(outputs['need_to_run'], 'true')
        self.assertEqual(outputs['target_ref'], 'abc123')
        self.assertIn('/commits/main', calls)

    def test_pull_request_api_failure_stops_before_the_matrix(self) -> None:
        outputs, stdout, code, calls = self._run_decide(
            {'RELEASE_BODY': '', 'RELEASE_CODE': '403'})
        self.assertEqual(code, 1)
        self.assertEqual(outputs, {})
        self.assertIn('::error::', stdout)
        self.assertNotIn('/commits/', calls)

    def test_pull_request_head_lookup_failure_stops(self) -> None:
        outputs, stdout, code, _ = self._run_decide({
            'RELEASE_BODY': '', 'RELEASE_CODE': '404',
            'COMMIT_BODY': '', 'COMMIT_CODE': '403'})
        self.assertEqual(code, 1)
        self.assertEqual(outputs, {})
        self.assertIn('::error::', stdout)

    def test_dispatch_explicit_ref_passes_through(self) -> None:
        outputs, stdout, code, calls = self._run_decide(
            {'EVENT_NAME': 'workflow_dispatch', 'INPUT_REF': 'v0.19.0'})
        self.assertEqual(code, 0, stdout)
        self.assertEqual(outputs['need_to_run'], 'true')
        self.assertEqual(outputs['trigger'], 'workflow_dispatch')
        self.assertEqual(outputs['target_ref'], 'v0.19.0')
        self.assertEqual(calls, '')

    def test_dispatch_api_failure_falls_back_to_default_branch(self) -> None:
        outputs, stdout, code, _ = self._run_decide({
            'EVENT_NAME': 'workflow_dispatch',
            'RELEASE_BODY': '', 'RELEASE_CODE': '403'})
        self.assertEqual(code, 0, stdout)
        self.assertEqual(outputs['target_ref'], 'main')
        self.assertIn('::warning::', stdout)

    def test_schedule_without_signal_skips_the_matrix(self) -> None:
        outputs, stdout, code, _ = self._run_decide(
            {'EVENT_NAME': 'schedule'})
        self.assertEqual(code, 0)
        self.assertEqual(outputs['trigger'], 'schedule')
        self.assertNotEqual(outputs.get('need_to_run'), 'true')
        self.assertIn('No upstream change', stdout)


if __name__ == '__main__':
    unittest.main()
