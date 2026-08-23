#!/usr/bin/env python3
"""Safely import a redacted server evidence bundle into ignored local storage.

This command never contacts a server and never reads credentials.  It accepts
only bundles produced by ``package_server_evidence.py`` and verifies every
exported member against the bundle manifest before writing it under
``outputs/imported_server_evidence`` (or an explicitly supplied staging path).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "server_evidence_bundle_v1"
REQUIRED_FILES = {"config.json", "metrics.json", "results.jsonl", "resource_ledger.json"}
FORBIDDEN_TEXT = (
    re.compile(r"(?i)(?:^|[^0-9])172\.16\.36\.16(?:[^0-9]|$)"),
    re.compile(r"(?i)(?:^|[^a-z])(?:ssh[_ -]?private|private[_ -]?key|api[_ -]?key|access[_ -]?token|secret)(?:[^a-z]|$)"),
    re.compile(r"(?i)(?:^|[\\/])(?:users|home|mnt|root)[\\/][^\s\"']+"),
)


class ImportError(ValueError):
    """The archive is not an eligible server evidence bundle."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or name != path.as_posix():
        raise ImportError(f"unsafe_archive_member:{name}")


def _sensitive(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORBIDDEN_TEXT)


def _load_manifest(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImportError("manifest_invalid_json") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ImportError("manifest_schema_mismatch")
    if value.get("archive_contains_redacted_bytes") is not True:
        raise ImportError("manifest_not_redacted")
    run = value.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("run_id"), str):
        raise ImportError("manifest_run_invalid")
    return value


def inspect_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.expanduser().resolve()
    if not bundle.is_file():
        raise ImportError("bundle_missing")
    archive_sha256 = _sha256(bundle.read_bytes())
    try:
        archive = zipfile.ZipFile(bundle)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ImportError("archive_invalid") from exc
    with archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ImportError("duplicate_archive_member")
        for name in names:
            _safe_member(name)
        if "manifest.json" not in names:
            raise ImportError("manifest_missing")
        manifest = _load_manifest(archive.read("manifest.json"))
        rows = manifest["run"].get("files")
        if not isinstance(rows, list) or not rows:
            raise ImportError("manifest_files_invalid")
        expected: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ImportError("manifest_file_row_invalid")
            path = str(row["path"])
            if path in expected or path.startswith("/") or ".." in PurePosixPath(path).parts:
                raise ImportError("manifest_file_path_invalid")
            expected[path] = row
        required = REQUIRED_FILES - set(expected)
        if required:
            raise ImportError("required_artifact_missing:" + ",".join(sorted(required)))
        actual_run_names = {name.removeprefix("run/") for name in names if name.startswith("run/")}
        if actual_run_names != set(expected):
            raise ImportError("archive_manifest_member_mismatch")
        for path, row in expected.items():
            member = "run/" + path
            payload = archive.read(member)
            if row.get("exported_bytes") != len(payload):
                raise ImportError(f"exported_size_mismatch:{path}")
            if row.get("exported_sha256") != _sha256(payload):
                raise ImportError(f"exported_hash_mismatch:{path}")
            if path.endswith((".json", ".jsonl", ".md")) or path.startswith(".run_"):
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ImportError(f"non_utf8_member:{path}") from exc
                if _sensitive(text):
                    raise ImportError(f"sensitive_text_in_member:{path}")
        return {
            "schema_version": "server_evidence_import_v1",
            "status": "ready",
            "archive": bundle.name,
            "archive_sha256": archive_sha256,
            "run_id": manifest["run"]["run_id"],
            "member_count": len(expected),
            "required_files": sorted(REQUIRED_FILES),
            "official_metric_scope": manifest.get("official_metric_scope", "internal_engineering_only"),
            "network_requests": 0,
            "ssh_or_dotenv_read": False,
        }


def import_bundle(bundle: Path, destination: Path) -> dict[str, Any]:
    report = inspect_bundle(bundle)
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            if not name.startswith("run/"):
                continue
            relative = PurePosixPath(name.removeprefix("run/"))
            target = destination.joinpath(*relative.parts).resolve()
            if destination not in target.parents:
                raise ImportError("destination_escape")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        manifest = archive.read("manifest.json")
    (destination / "manifest.json").write_bytes(manifest)
    report["destination"] = str(destination)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("outputs/imported_server_evidence"),
    )
    args = parser.parse_args()
    try:
        report = import_bundle(args.bundle, args.destination)
    except (ImportError, OSError, zipfile.BadZipFile) as exc:
        print(json.dumps({"schema_version": "server_evidence_import_v1", "status": "blocked", "reason": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 3
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
