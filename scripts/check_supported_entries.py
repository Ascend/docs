#!/usr/bin/env python3
"""Validate the supported entries of an examples manifest.

Checks each entry's path and scheduling fields, including the optional
a3_runner/a3_image pair. Writes the original, A3, and publish matrices
to GITHUB_OUTPUT; exits 1 after listing every problem.

Requires PyYAML.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml

REQUIRED_FIELDS = ('path', 'profile', 'runner', 'image', 'timeout_minutes')


def validate(
    supported: list[dict], target_root: Path, project_root: Path | None = None
) -> tuple[list[dict], list[str]]:
    """Return (valid matrix entries, error messages) for the manifest."""
    entries: list[dict] = []
    errors: list[str] = []
    for item in supported:
        path = item.get('path', '<missing path>')
        if not isinstance(path, str) or not path.strip():
            errors.append('supported example path must be a non-empty string')
            continue
        relative_path = PurePosixPath(path)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            errors.append(f'{path}: path must be relative and stay within its checkout')
            continue
        source = item.get('source', 'upstream')
        if source not in ('upstream', 'project'):
            errors.append(f'{path}: source must be upstream or project')
            continue
        root = project_root if source == 'project' else target_root
        if root is None:
            errors.append(f'{path}: project root is required for project example')
            continue
        resolved_root = root.resolve()
        candidate = (resolved_root / path).resolve()
        if not candidate.is_relative_to(resolved_root):
            errors.append(f'{path}: path must stay within its checkout')
            continue
        if not candidate.exists():
            if source == 'project':
                errors.append(f'supported project example missing: {path}')
            else:
                errors.append(f'supported example missing from target tree: {path}')
            continue
        missing = [field for field in REQUIRED_FIELDS if not item.get(field)]
        if missing:
            errors.append(f'{path}: missing required field(s): {missing}')
            continue
        overlay_args = item.get('overlay_args')
        if overlay_args is None:
            overlay_args = []
        elif not isinstance(overlay_args, list) or not all(
                isinstance(arg, str) and arg.strip() for arg in overlay_args):
            errors.append(
                f'{path}: overlay_args must be a list of non-empty strings')
            continue
        exec_path = item.get('exec')
        if exec_path is not None and (
                not isinstance(exec_path, str) or not exec_path.strip()):
            errors.append(f'{path}: exec must be a non-empty string')
            continue
        # Optional launcher: names a multi-process launch mode the
        # project's run_example.sh understands (e.g. accelerate-deepspeed
        # wraps the bare `python` call in `accelerate launch --config_file`
        # with a materialized DeepSpeed config). Empty/absent = bare run.
        launcher = item.get('launcher')
        if launcher is not None and (
                not isinstance(launcher, str) or not launcher.strip()):
            errors.append(f'{path}: launcher must be a non-empty string')
            continue
        entry = dict(item)
        entry['overlay_args'] = overlay_args
        if exec_path is not None:
            entry['exec'] = exec_path.strip()
        if launcher is not None:
            entry['launcher'] = launcher.strip()
        # Display name for the run-example job label: full relative path
        # with the extension stripped (examples/sft/run_peft.sh ->
        # examples/sft/run_peft; a bare foo.py -> foo). Uniform and
        # unique - same-named scripts in different directories get
        # distinct labels (peft's five */train_dreambooth.py used to
        # collapse to a single "train_dreambooth").
        entry['name'] = str(relative_path.with_suffix(''))
        entries.append(entry)
    return entries, errors


def expand_a3(entries: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """Keep the original matrix and build A3 rows for explicit configurations."""
    original_matrix = list(entries)
    a3_matrix = []
    errors = []
    names = Counter(entry['name'] for entry in entries)
    for index, entry in enumerate(entries):
        if 'a3_runner' not in entry and 'a3_image' not in entry:
            continue
        original = dict(entry)
        runner = original.pop('a3_runner', None)
        image = original.pop('a3_image', None)
        invalid = [
            field for field, value in [('a3_runner', runner), ('a3_image', image)]
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid:
            errors.append(
                f"{entry['path']}: a3_runner and a3_image must be specified "
                f"together as non-empty strings; missing or invalid: {', '.join(invalid)}"
            )
            continue
        name = entry['name']
        if names[name] > 1:
            name = f'{name} ({index + 1})'
        original_matrix[index] = original
        a3_matrix.append(dict(
            original,
            name=f'{name} [a3]',
            machine_type='a3',
            runner=runner.strip(),
            image=image.strip(),
        ))
    return original_matrix, a3_matrix, errors


def build_publish_matrix(
    entries: list[dict], a3_entries: list[dict]
) -> list[dict]:
    """Publish one result per example; baseline rows are A2."""
    return [dict(entry, machine_type='a2') for entry in entries] + a3_entries


def write_github_output(entries: list[dict], a3_entries: list[dict]) -> None:
    output_path = os.environ.get('GITHUB_OUTPUT')
    if not output_path:
        return
    with open(output_path, 'a', encoding='utf-8') as handle:
        groups = [
            ('supported', entries),
            ('a3', a3_entries),
            ('publish', build_publish_matrix(entries, a3_entries)),
        ]
        for group, rows in groups:
            handle.write(f'{group}_matrix<<EOF\n')
            handle.write(json.dumps(rows, ensure_ascii=False))
            handle.write('\nEOF\n')
            handle.write(f'has_{group}={"true" if rows else "false"}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--target-root', required=True,
        help='Checkout of the target project tree')
    parser.add_argument(
        '--manifest', required=True,
        help='Path to examples-guards/<project>/examples_manifest.yaml')
    args = parser.parse_args()
    manifest = yaml.safe_load(
        Path(args.manifest).read_text(encoding='utf-8')) or {}
    supported = manifest.get('supported') or []
    entries, errors = validate(
        supported,
        Path(args.target_root),
        Path(args.manifest).resolve().parent,
    )
    matrix, a3_matrix, a3_errors = expand_a3(entries)
    errors.extend(a3_errors)
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        print('fix the supported section of the manifest before this '
              'pipeline can schedule examples', file=sys.stderr)
        raise SystemExit(1)
    write_github_output(matrix, a3_matrix)
    print(f'manifest ok: {len(entries)} supported entry(ies)')


if __name__ == '__main__':
    main()
