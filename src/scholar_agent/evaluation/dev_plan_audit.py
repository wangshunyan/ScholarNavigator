"""Read-only audit helpers for the repository development feedback loop.

The audit deliberately distinguishes a missing tracked input from a missing
external benchmark artifact.  It never creates placeholder files and never
loads ``.env`` or benchmark gold into an execution path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_PATH_KEYS = {
    "path",
    "run_dir",
    "snapshot_dir",
    "source_run_dir",
    "replay_root",
    "output_dir",
    "results_path",
    "config_path",
    "dataset_path",
}


def audit_benchmark_artifacts(root: str | Path) -> dict[str, Any]:
    """Audit path/hash references in ``benchmark/**/*.json``.

    The return value is deterministic and safe to store in ``outputs``.  A
    missing path under ``outputs/`` is classified as ``missing_external``;
    missing paths elsewhere are ``missing_tracked``.  Existing files with a
    declared digest are checked byte-for-byte.
    """

    repository = Path(root).expanduser().resolve()
    benchmark_root = repository / "benchmark"
    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    if benchmark_root.is_dir():
        manifests = sorted(benchmark_root.rglob("*.json"))
    else:
        manifests = []

    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parse_errors.append(
                {
                    "manifest": _relative(manifest_path, repository),
                    "error": type(exc).__name__,
                }
            )
            continue
        _walk_manifest(
            payload,
            manifest_path=manifest_path,
            repository=repository,
            location="$",
            records=records,
        )

    records.sort(key=lambda item: (item["manifest"], item["location"], item["path"]))
    counts: dict[str, int] = {}
    for item in records:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {
        "schema_version": "dev-plan-artifact-audit-v1",
        "repository": _relative(repository, repository),
        "manifest_count": len(manifests),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "reference_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "records": records,
        "passed": not parse_errors and not any(
            item["status"] in {"missing_tracked", "hash_mismatch", "unsafe_absolute"}
            for item in records
        ),
    }


def _walk_manifest(
    value: Any,
    *,
    manifest_path: Path,
    repository: Path,
    location: str,
    records: list[dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in _PATH_KEYS and isinstance(child, str):
                records.append(
                    _audit_reference(
                        child,
                        key=key,
                        parent=value,
                        manifest_path=manifest_path,
                        repository=repository,
                        location=child_location,
                    )
                )
            _walk_manifest(
                child,
                manifest_path=manifest_path,
                repository=repository,
                location=child_location,
                records=records,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_manifest(
                child,
                manifest_path=manifest_path,
                repository=repository,
                location=f"{location}[{index}]",
                records=records,
            )


def _audit_reference(
    raw_path: str,
    *,
    key: str,
    parent: dict[str, Any],
    manifest_path: Path,
    repository: Path,
    location: str,
) -> dict[str, Any]:
    candidate = Path(raw_path).expanduser()
    expected = _expected_sha(parent, key)
    if candidate.is_absolute():
        status = "unsafe_absolute"
        resolved = candidate
    else:
        resolved = (repository / candidate).resolve()
        try:
            resolved.relative_to(repository)
        except ValueError:
            status = "unsafe_absolute"
        else:
            status = _path_status(resolved, raw_path, expected)

    actual = None
    if status in {"present", "present_unhashed", "hash_mismatch"} and resolved.is_file():
        actual = _sha256(resolved)
    return {
        "manifest": _relative(manifest_path, repository),
        "location": location,
        "key": key,
        "path": raw_path.replace("\\", "/"),
        "resolved": _relative(resolved, repository) if not candidate.is_absolute() else str(resolved),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "status": status,
    }


def _expected_sha(parent: dict[str, Any], key: str) -> str | None:
    direct = parent.get("sha256")
    if isinstance(direct, str):
        return direct
    if key.endswith("_path"):
        sibling = parent.get(f"{key[:-5]}_sha256")
        if isinstance(sibling, str):
            return sibling
    return None


def _path_status(path: Path, raw_path: str, expected: str | None) -> str:
    if not path.exists():
        normalized = raw_path.replace("\\", "/")
        return "missing_external" if normalized.startswith("outputs/") else "missing_tracked"
    if expected is None:
        return "present_unhashed"
    return "present" if path.is_file() and _sha256(path) == expected else "hash_mismatch"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, repository: Path) -> str:
    try:
        return path.resolve().relative_to(repository).as_posix() or "."
    except ValueError:
        return path.as_posix()
