#!/usr/bin/env python3
"""Standard-library verifier for a human-annotator qualification kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath


PROTOCOL = "human_annotator_qualification_intake_v1"
EXPECTED_FILES = {
    "README.txt",
    "calibration_items.json",
    "contract.json",
    "manifest.json",
    "submission_template.json",
    "verify.py",
}


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def canonical(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fail(reason):
    sys.stdout.buffer.write(
        canonical(
            {
                "exit_code": 2,
                "protocol": PROTOCOL,
                "reason_code": reason,
                "status": "qualification_or_role_violation",
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
            if len(infos) != len(EXPECTED_FILES):
                raise ValueError("kit_inventory_invalid")
            files = {}
            for info in infos:
                path = PurePosixPath(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in info.filename
                    or info.filename in files
                    or info.is_dir()
                    or mode not in (0, 0o100000)
                    or info.file_size > 2 * 1024 * 1024
                ):
                    raise ValueError("kit_member_unsafe")
                files[info.filename] = archive.read(info)
        if set(files) != EXPECTED_FILES:
            raise ValueError("kit_inventory_invalid")
        manifest = json.loads(
            files["manifest.json"].decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        if not isinstance(manifest, dict):
            raise ValueError("kit_manifest_invalid")
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
                or name == "manifest.json"
                or row["size"] != len(files[name])
                or row["sha256"] != digest(files[name])
            ):
                raise ValueError("kit_inventory_invalid")
            seen.add(name)
        if seen != set(files) - {"manifest.json"}:
            raise ValueError("kit_inventory_invalid")
        contract = json.loads(
            files["contract.json"].decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        if (
            contract["contract_sha256"] != manifest["contract_sha256"]
            or contract["challenge"] != manifest["challenge"]
            or contract["role"] != manifest["role"]
        ):
            raise ValueError("kit_contract_binding_invalid")
    except Exception:
        return fail("kit_integrity_invalid")
    sys.stdout.buffer.write(
        canonical(
            {
                "exit_code": 0,
                "protocol": PROTOCOL,
                "role": manifest["role"],
                "status": "qualification_kit_verified",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
