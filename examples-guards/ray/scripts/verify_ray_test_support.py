#!/usr/bin/env python3
"""Verify that installed Ray matches the target and exposes its test package."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path


def target_version(target_root: Path) -> str:
    version_path = target_root / "python" / "ray" / "_version.py"
    if not version_path.is_file():
        raise SystemExit(f"target Ray version file is missing: {version_path}")
    text = version_path.read_text(encoding="utf-8")
    match = re.search(r'''^version = [\'"]([^\'"]+)[\'"]''', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"could not read target Ray version: {version_path}")
    return match.group(1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <target-root>")

    target_root = Path(sys.argv[1]).resolve()
    expected_version = target_version(target_root)

    import ray

    installed_version = ray.__version__
    if installed_version != expected_version:
        raise SystemExit(
            "Ray version mismatch: "
            f"target={expected_version}, installed={installed_version}"
        )

    try:
        test_conftest = importlib.import_module("ray.tests.conftest")
    except ModuleNotFoundError as exc:
        raise SystemExit(f"ray.tests.conftest is unavailable: {exc}") from exc

    expected_tests = (target_root / "python" / "ray" / "tests").resolve()
    actual_conftest = Path(test_conftest.__file__).resolve()
    if not actual_conftest.is_relative_to(expected_tests):
        raise SystemExit(
            "ray.tests was not linked from target checkout: "
            f"expected under {expected_tests}, got {actual_conftest}"
        )

    print(
        "Ray test support verified: "
        f"version={installed_version}, tests={actual_conftest}"
    )


if __name__ == "__main__":
    main()
