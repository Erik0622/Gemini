#!/usr/bin/env python3
"""Utility to scan repository files for unresolved merge conflict markers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "env",
    "venv",
    ".venv",
}


def iter_paths(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        if current.is_dir():
            if current.name in SKIP_DIRS:
                continue
            stack.extend(sorted(current.iterdir(), reverse=True))
        else:
            yield current


def find_conflicts(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    conflicts: list[tuple[int, str]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        for marker in CONFLICT_MARKERS:
            if stripped.startswith(marker):
                conflicts.append((idx, marker))
    return conflicts


def scan(root: Path) -> dict[Path, list[tuple[int, str]]]:
    results: dict[Path, list[tuple[int, str]]] = {}
    for path in iter_paths(root):
        matches = find_conflicts(path)
        if matches:
            results[path] = matches
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root directory to scan (defaults to repository root)",
    )
    args = parser.parse_args(argv)

    root = args.root
    if not root.exists():
        parser.error(f"Root path {root!s} does not exist")

    conflicts = scan(root)
    if not conflicts:
        print(f"No merge conflict markers found under {root!s}.")
        return 0

    print("Merge conflict markers detected:")
    for path, matches in sorted(conflicts.items()):
        for line_no, marker in matches:
            print(f"  {path}: line {line_no} contains '{marker}'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
