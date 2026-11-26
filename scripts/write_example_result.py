#!/usr/bin/env python3
"""Write result.json for one example from its GitHub Actions job conclusion."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REQUIRED_ENV = ('GH_TOKEN', 'GITHUB_REPOSITORY', 'GITHUB_RUN_ID',
                'EXPECTED_JOB_NAME', 'TRIGGER', 'TARGET_REPO',
                'TARGET_REF', 'EXAMPLE_PATH', 'IMAGE')


def conclusion_to_status(conclusion: str) -> str:
    """success / cancelled pass through; everything else is failure."""
    if conclusion in ('success', 'cancelled'):
        return conclusion
    return 'failure'


def job_matches(name: str, expected: str) -> bool:
    """Exact match, or suffix match for called-workflow name prefixes.

    Inside a reusable workflow github.job is unprefixed ('<name>'),
    but the Jobs API prefixes every job name with the caller's
    workflow name ('peft-examples / <name>'); chained reusable
    workflows nest further prefixes.
    """
    return name == expected or name.endswith(f' / {expected}')


def read_jobs_page(request: urllib.request.Request) -> dict:
    """Read one page, retrying temporary gateway failures and timeouts."""
    for delay in (1, 2, 4, None):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            if delay is None:
                raise
            if isinstance(exc, urllib.error.HTTPError):
                if exc.code not in (502, 503, 504):
                    raise
                exc.close()
            elif not isinstance(getattr(exc, 'reason', exc), TimeoutError):
                raise
            print(f'GitHub Jobs API request failed ({exc}). Retrying in {delay}s.',
                  file=sys.stderr)
            time.sleep(delay)


def _find_job(repo: str, run_id: str, job_name: str,
              token: str) -> tuple[bool, str | None]:
    """Page through the Jobs API for a name match.

    Returns (found, conclusion): found is True once a matching job is
    seen; conclusion is that job's conclusion, which may be None while
    the API is still propagating a freshly-finished job.
    """
    per_page = 20
    page = 1
    while True:
        url = (f'https://api.github.com/repos/{repo}/actions/runs/{run_id}'
               f'/jobs?per_page={per_page}&page={page}')
        request = urllib.request.Request(url, headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
        })
        data = read_jobs_page(request)
        jobs = data.get('jobs') or []
        for job in jobs:
            if job_matches(job.get('name') or '', job_name):
                return True, job.get('conclusion')
        if len(jobs) < per_page:
            return False, None
        page += 1


def fetch_conclusion(repo: str, run_id: str, job_name: str,
                     token: str, deadline: float = 180.0,
                     interval: float = 5.0) -> str | None:
    """Return the matching job's conclusion, waiting out API lag.

    The Jobs API is eventually consistent: for a few seconds after a
    job finishes, its conclusion can still read as null even though
    `needs` already fired the downstream job. Two cases are handled
    differently:

    * no name match across all pages -> permanent (retrying cannot
      help), return None immediately;
    * matched but conclusion still null -> poll until it resolves or
      deadline (seconds) expires.
    """
    started = time.monotonic()
    while True:
        found, conclusion = _find_job(repo, run_id, job_name, token)
        if not found:
            return None
        if conclusion:
            return conclusion
        if time.monotonic() - started >= deadline:
            print(f'timed out after {deadline:.0f}s waiting for '
                  f'conclusion of {job_name}', file=sys.stderr)
            return None
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', default='result.json',
        help='result JSON output path')
    args = parser.parse_args()

    missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        print(f'missing required environment: {missing}', file=sys.stderr)
        raise SystemExit(1)
    env = os.environ

    conclusion = fetch_conclusion(
        env['GITHUB_REPOSITORY'], env['GITHUB_RUN_ID'],
        env['EXPECTED_JOB_NAME'], env['GH_TOKEN'])
    if not conclusion:
        print(f"could not find completed job: {env['EXPECTED_JOB_NAME']}",
              file=sys.stderr)
        raise SystemExit(1)

    result = {
        'trigger': env['TRIGGER'],
        'target_repo': env['TARGET_REPO'],
        'target_ref': env['TARGET_REF'],
        'path': env['EXAMPLE_PATH'],
        'image': env['IMAGE'],
        'job_status': conclusion_to_status(conclusion),
    }
    if env.get('EXAMPLE_SOURCE'):
        result['source'] = env['EXAMPLE_SOURCE']
    if env.get('MACHINE_TYPE'):
        result['machine_type'] = env['MACHINE_TYPE']
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f"wrote {output}: job_status={result['job_status']}")


if __name__ == '__main__':
    main()
