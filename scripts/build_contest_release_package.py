#!/usr/bin/env python3
"""Build a source-only contest package from tracked repository files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


MANIFEST_NAME = "release-manifest.json"
SOURCE_DATE_EPOCH = (1980, 1, 1, 0, 0, 0)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _dirty_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True,
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line]


def _write_deterministic(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=SOURCE_DATE_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def _included(path: str) -> bool:
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if name in EXCLUDED_NAMES or normalized.endswith(".env"):
        return False
    if any(normalized.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    # Keep the small/local BM25 title corpus so a fresh release can run an
    # offline retrieval smoke.  Semantic corpora, models and raw datasets
    # remain external assets and are never bundled.
    if normalized.startswith("datasets/") and normalized != "datasets/local_bm25/pasa_papers.jsonl" and not normalized.endswith(".md"):
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
    dirty = _dirty_paths(root)
    if dirty:
        raise ValueError("release_requires_clean_git_tree:" + "|".join(dirty[:20]))
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
    manifest_files = []
    for relative in files:
        source = root / relative
        if not source.is_file():
            raise ValueError(f"tracked_file_missing:{relative}")
        manifest_files.append({
            "path": relative.replace("\\", "/"),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
        })
    manifest = {
        "schema_version": "contest-release-manifest-v1",
        "source_commit": _commit(root),
        "file_count": len(files),
        "files": manifest_files,
        "internal_metric_scope": "not_official_competition_scorer",
        "excluded": {
            "dotenv": True,
            "outputs": True,
            "semantic_models_and_indexes": True,
            "ssh_credentials": True,
        },
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            source = root / relative
            _write_deterministic(archive, relative.replace("\\", "/"), source.read_bytes())
        _write_deterministic(archive, MANIFEST_NAME, manifest_bytes)
    temporary.replace(output)
    return {
        "schema_version": "contest-release-package-v1",
        "file_count": len(files),
        "output": str(output),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "excluded_names": sorted(EXCLUDED_NAMES),
        "internal_metric_scope": "not_official_competition_scorer",
        "manifest_name": MANIFEST_NAME,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
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
