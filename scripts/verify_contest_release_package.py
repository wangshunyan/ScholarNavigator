#!/usr/bin/env python3
"""Verify a contest source ZIP and its deterministic release manifest offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from scripts.build_contest_release_package import MANIFEST_NAME, MAX_RELEASE_BYTES


FORBIDDEN_PREFIXES = ("outputs/", "datasets/semantic/", "legacy/spar_original/")


class ReleaseVerificationError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_package(package: Path, *, expected_commit: str | None = None) -> dict[str, object]:
    package = package.expanduser().resolve()
    if not package.is_file():
        raise ReleaseVerificationError("package_missing")
    actual_size = package.stat().st_size
    if actual_size > MAX_RELEASE_BYTES:
        raise ReleaseVerificationError(f"archive_exceeds_limit:{actual_size}")
    try:
        with zipfile.ZipFile(package) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ReleaseVerificationError("duplicate_zip_member")
            if MANIFEST_NAME not in names:
                raise ReleaseVerificationError("manifest_missing")
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise ReleaseVerificationError(f"unsafe_zip_member:{name}")
                if name == ".env" or name.endswith(".env") or name.lower().endswith((".pem", ".key")):
                    raise ReleaseVerificationError(f"forbidden_zip_member:{name}")
                if any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
                    raise ReleaseVerificationError(f"forbidden_zip_member:{name}")
            try:
                manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReleaseVerificationError("manifest_invalid_json") from exc
            if manifest.get("schema_version") != "contest-release-manifest-v1":
                raise ReleaseVerificationError("manifest_schema_invalid")
            if expected_commit is not None and manifest.get("source_commit") != expected_commit:
                raise ReleaseVerificationError("source_commit_mismatch")
            rows = manifest.get("files")
            if not isinstance(rows, list) or manifest.get("file_count") != len(rows):
                raise ReleaseVerificationError("manifest_file_count_invalid")
            expected_names = {str(row.get("path")) for row in rows}
            if any(not isinstance(row, dict) for row in rows):
                raise ReleaseVerificationError("manifest_file_entry_invalid")
            if expected_names != set(names) - {MANIFEST_NAME}:
                raise ReleaseVerificationError("manifest_members_mismatch")
            checked = 0
            for row in rows:
                name = row.get("path")
                if not isinstance(name, str) or not isinstance(row.get("sha256"), str):
                    raise ReleaseVerificationError("manifest_file_entry_invalid")
                content = archive.read(name)
                if row.get("bytes") != len(content) or row["sha256"] != _sha256(content):
                    raise ReleaseVerificationError(f"member_hash_mismatch:{name}")
                checked += 1
    except zipfile.BadZipFile as exc:
        raise ReleaseVerificationError("zip_invalid") from exc
    return {
        "schema_version": "contest-release-verification-v1",
        "status": "ready",
        "package": str(package),
        "archive_bytes": actual_size,
        "max_archive_bytes": MAX_RELEASE_BYTES,
        "source_commit": manifest.get("source_commit"),
        "file_count": checked,
        "manifest_name": MANIFEST_NAME,
        "network_requests": 0,
        "gold_or_qrels_loaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    try:
        report = verify_package(args.package, expected_commit=args.expected_commit)
    except (OSError, ReleaseVerificationError) as exc:
        report = {"schema_version": "contest-release-verification-v1", "status": "blocked", "reason": str(exc)}
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 3
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
