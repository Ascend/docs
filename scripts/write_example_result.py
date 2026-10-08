#!/usr/bin/env python3
"""Publish result.json for one run-example matrix leg.

The validate-results job (GitHub-hosted runner) queries the GitHub Job
API for the corresponding <name> (matrix.example.name) job's
conclusion,
normalizes it (success / cancelled pass through; anything else,
including a missing conclusion, is a failure), and writes the
machine-readable result artifact consumed by external machines
(schemas/result.schema.json). The self-hosted NPU runner only runs
the example and reports its normal job status; artifact publishing
stays on GitHub-hosted infrastructure.

Reads from the environment:
    GH_TOKEN / GITHUB_REPOSITORY / GITHUB_RUN_ID    auth + API address
    EXPECTED_JOB_NAME    display name: matrix.example.name (basename,
                              extension stripped)
    TRIGGER / TARGET_REPO / TARGET_REF / EXAMPLE_PATH / IMAGE
Writes: --output (default result.json), fields per
schemas/result.schema.json. Exit 1 if the expected job cannot be
found or has no conclusion yet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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


def fetch_conclusion(repo: str, run_id: str, job_name: str,
                     token: str) -> str | None:
    """Find the job by display name; return its conclusion or None."""
    page = 1
    while True:
        url = (f'https://api.github.com/repos/{repo}/actions/runs/{run_id}'
               f'/jobs?per_page=100&page={page}')
        request = urllib.request.Request(url, headers={
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github+json',
        })
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
        jobs = data.get('jobs') or []
        for job in jobs:
            if job_matches(job.get('name') or '', job_name):
                return job.get('conclusion')
        if len(jobs) < 100:
            return None
        page += 1


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
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(f"wrote {output}: job_status={result['job_status']}")


if __name__ == '__main__':
    main()
