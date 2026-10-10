#!/usr/bin/env python3
"""Select named test requirements from the checked-out Ray source tree."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def select(requirements_path: Path, package: str) -> list[str]:
    if not requirements_path.is_file():
        raise SystemExit(f"target requirements file is missing: {requirements_path}")

    expected = canonicalize(package)
    selected: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if match and canonicalize(match.group(1)) == expected:
            selected.append(line)

    if not selected:
        raise SystemExit(
            f"target requirements file does not declare {expected}: {requirements_path}"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirement",
        action="append",
        nargs=2,
        required=True,
        metavar=("FILE", "PACKAGE"),
    )
    args = parser.parse_args()

    seen: set[str] = set()
    for raw_path, package in args.requirement:
        for requirement in select(Path(raw_path), package):
            if requirement not in seen:
                print(requirement)
                seen.add(requirement)


if __name__ == "__main__":
    main()
