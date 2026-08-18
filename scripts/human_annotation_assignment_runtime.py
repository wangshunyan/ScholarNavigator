#!/usr/bin/env python3
"""Standard-library verifier for an offline annotation assignment bundle."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import zipfile


PROTOCOL = "human_annotation_assignment_bundle_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "80cd4bf6f5263231a34a3ad535759f6c6910e835"
ROLES = {"annotator_a", "annotator_b", "adjudicator"}
ROLE_TO_SIDE = {"annotator_a": "A", "annotator_b": "B"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PRINCIPAL_RE = re.compile(r"^prn_[0-9a-f]{16}$")
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_MEMBER_COUNT = 16


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
    ).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    value = decode_any(raw)
    if not isinstance(value, dict):
        raise ValueError("object_required")
    return value


def decode_any(raw):
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("invalid_number")
        ),
    )


def safe_name(value):
    path = pathlib.PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ".." in path.parts
        or path.as_posix() != value
        or path.name == ".env"
        or path.parts[0] in {"operator", "third_party"}
    ):
        raise ValueError("unsafe_path")
    return value


def manifest_hash(value):
    payload = dict(value)
    payload["manifest_sha256"] = "0" * 64
    return digest(canonical(payload))


def fail(reason):
    print(
        json.dumps(
            {
                "exit_code": 2,
                "protocol": PROTOCOL,
                "reason": reason,
                "schema_version": SCHEMA_VERSION,
                "status": "assignment_or_blinding_violation",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 2


def verify(path):
    if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("bundle_size_or_presence_invalid")
    files = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBER_COUNT:
            raise ValueError("bundle_member_limit")
        for info in infos:
            name = safe_name(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if (
                name in files
                or info.is_dir()
                or mode not in (0, 0o100000)
                or info.file_size > MAX_MEMBER_BYTES
            ):
                raise ValueError("bundle_member_unsafe")
            files[name] = archive.read(info)
    if "manifest.json" not in files:
        raise ValueError("bundle_manifest_missing")
    manifest = decode(files["manifest.json"])
    required = {
        "assignment_challenge",
        "assignment_protocol_sha256",
        "bundle_protocol",
        "files",
        "formal_validation_complete",
        "item_count",
        "manifest_sha256",
        "package_identity",
        "principal_commitment",
        "principal_id",
        "qualification_sha256",
        "role",
        "schema_version",
        "source_commit",
        "state",
        "synthetic_only",
    }
    if set(manifest) != required:
        raise ValueError("bundle_manifest_schema_invalid")
    if (
        manifest["bundle_protocol"] != PROTOCOL
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["source_commit"] != SOURCE_COMMIT
        or manifest["role"] not in ROLES
        or manifest["state"] != "issued"
        or manifest["formal_validation_complete"] is not False
        or not PRINCIPAL_RE.fullmatch(str(manifest["principal_id"]))
        or not SHA256_RE.fullmatch(str(manifest["principal_commitment"]))
        or not SHA256_RE.fullmatch(str(manifest["qualification_sha256"]))
        or not SHA256_RE.fullmatch(str(manifest["assignment_challenge"]))
        or not SHA256_RE.fullmatch(str(manifest["assignment_protocol_sha256"]))
        or manifest_hash(manifest) != manifest["manifest_sha256"]
    ):
        raise ValueError("bundle_manifest_binding_invalid")
    inventory = manifest["files"]
    if not isinstance(inventory, list):
        raise ValueError("bundle_inventory_invalid")
    seen = set()
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size"}
            or not isinstance(row["path"], str)
            or row["path"] in seen
            or row["path"] == "manifest.json"
            or row["path"] not in files
            or row["size"] != len(files[row["path"]])
            or row["sha256"] != digest(files[row["path"]])
        ):
            raise ValueError("bundle_inventory_invalid")
        seen.add(row["path"])
    if seen != set(files) - {"manifest.json"}:
        raise ValueError("bundle_inventory_invalid")
    role = manifest["role"]
    if role == "adjudicator":
        required_payload = {
            "README.txt",
            "disagreement_view_contract.json",
            "receipt_template.json",
            "rubric.json",
            "verify.py",
        }
        if set(files) - {"manifest.json"} != required_payload:
            raise ValueError("adjudicator_blinding_violation")
        if manifest["item_count"] != 0:
            raise ValueError("adjudicator_blinding_violation")
    else:
        required_payload = {
            "README.txt",
            "payload/app.js",
            "payload/index.html",
            "payload/items-data.js",
            "payload/items.json",
            "payload/package-data.js",
            "payload/package.json",
            "payload/rubric.json",
            "receipt_template.json",
            "verify.py",
        }
        if set(files) - {"manifest.json"} != required_payload:
            raise ValueError("annotator_payload_invalid")
        if manifest["item_count"] != 471:
            raise ValueError("annotator_payload_invalid")
        package = decode(files["payload/package.json"])
        expected_side = ROLE_TO_SIDE[role]
        if (
            package.get("side") != expected_side
            or manifest["package_identity"].get("side") != expected_side
        ):
            raise ValueError("annotator_package_role_mismatch")
        items = decode_any(files["payload/items.json"])
        if not isinstance(items, list) or len(items) != 471:
            raise ValueError("annotator_payload_invalid")
        aliases = set()
        for item in items:
            if (
                not isinstance(item, dict)
                or set(item) != {"abstract", "alias", "query", "title", "year"}
                or not isinstance(item["alias"], str)
                or item["alias"] in aliases
            ):
                raise ValueError("annotator_payload_invalid")
            aliases.add(item["alias"])
    return {
        "bundle_sha256": digest(path.read_bytes()),
        "exit_code": 0,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "role": role,
        "schema_version": SCHEMA_VERSION,
        "status": "assignment_bundle_verified",
    }


def main():
    if len(sys.argv) != 2:
        return 4
    try:
        result = verify(pathlib.Path(sys.argv[1]))
    except (OSError, UnicodeError, ValueError, TypeError, zipfile.BadZipFile):
        return fail("bundle_invalid")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
