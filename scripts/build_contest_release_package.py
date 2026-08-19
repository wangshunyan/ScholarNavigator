#!/usr/bin/env python3
"""Build a source-only contest package from tracked repository files."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


EXCLUDED_PREFIXES = (
    ".git/",
    ".github/",
    "outputs/",
    "datasets/semantic/",
    "legacy/spar_original/",
    "frontend/node_modules/",
)
EXCLUDED_NAMES = {".env"}


def _tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(item for item in result.stdout.decode().split("\0") if item)


def _included(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if name in EXCLUDED_NAMES or normalized.endswith(".env"):
        return False
    if any(normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    if normalized.startswith("datasets/") and not normalized.endswith(".md"):
        return False
    if normalized.startswith("benchmark/") and (
        "evidence_registry_baseline/" in normalized
        or normalized.endswith(".jsonl")
        or "/result" in normalized
    ):
        return False
    return True


def build_package(root: Path, output: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    files = [path for path in _tracked_files(root) if _included(path)]
    violations = [
        path for path in files
        if path.startswith("legacy/spar_original/")
        or path.endswith(".env")
        or path.startswith("outputs/")
        or path.startswith("datasets/semantic/")
    ]
    if violations:
        raise ValueError("release_exclusion_violation:" + ",".join(violations))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            source = root / relative
            if not source.is_file():
                raise ValueError(f"tracked_file_missing:{relative}")
            archive.write(source, relative)
    temporary.replace(output)
    return {
        "schema_version": "contest-release-package-v1",
        "file_count": len(files),
        "output": str(output),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "excluded_names": sorted(EXCLUDED_NAMES),
        "internal_metric_scope": "not_official_competition_scorer",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_package(args.repository_root, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
