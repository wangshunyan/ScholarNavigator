#!/usr/bin/env python3
"""Standard-library verifier for an official scorer intake kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def fail(reason):
    sys.stdout.buffer.write(
        canonical(
            {
                "exit_code": 2,
                "protocol": "official_scorer_package_intake_v1",
                "reason_code": reason,
                "status": "package_schema_or_sandbox_violation",
            }
        )
    )
    return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", required=True)
    args = parser.parse_args()
    try:
        with zipfile.ZipFile(Path(args.kit)) as archive:
            infos = archive.infolist()
            if len(infos) != 5:
                raise ValueError("kit_inventory_invalid")
            files = {}
            for info in infos:
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or info.filename in files:
                    raise ValueError("kit_member_unsafe")
                files[info.filename] = archive.read(info)
        manifest = json.loads(
            files["manifest.json"].decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        claimed = manifest["manifest_sha256"]
        manifest["manifest_sha256"] = "0" * 64
        if digest(canonical(manifest)) != claimed:
            raise ValueError("kit_manifest_invalid")
        seen = set()
        for row in manifest["files"]:
            name = row["path"]
            if (
                name in seen
                or name not in files
                or row["size"] != len(files[name])
                or row["sha256"] != digest(files[name])
            ):
                raise ValueError("kit_inventory_invalid")
            seen.add(name)
        if seen != set(files) - {"manifest.json"}:
            raise ValueError("kit_inventory_invalid")
    except Exception:
        return fail("kit_integrity_invalid")
    sys.stdout.buffer.write(
        canonical(
            {
                "exit_code": 0,
                "protocol": "official_scorer_package_intake_v1",
                "status": "official_scorer_intake_kit_verified",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
