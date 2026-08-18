#!/usr/bin/env python3
"""Build and independently verify standalone_auditor_bundle_v1.

The ``verify`` and ``compare`` paths use only the Python standard library and
never import project modules, execute archive members, use the network, launch
subprocesses, or read files other than the explicitly supplied archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import stat
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath


PROTOCOL = "standalone_auditor_bundle_v1"
SCHEMA = "1"
EXIT_PASSED = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
MANIFEST = "manifest.json"
VERIFIER = "verify.py"
REQUIRED_BLOCKERS = {
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
}
REQUIRED_BLOCKED_CLAIMS = {
    "contest_full1000_completion",
    "contest_human_precision",
    "contest_official_scorer_alignment",
}
REQUIRED_SOURCE_ROLES = {
    "claims",
    "clearance",
    "evidence_index",
    "freshness",
    "missing_inputs",
    "readiness",
    "readiness_contract",
    "release_bundle",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CANONICAL_MEMBERS = {
    "blockers.json",
    "claims.json",
    "evidence_index.json",
    "freshness.json",
    "policy.json",
    "preregistration.json",
    "protocol_dependencies.json",
    "readiness.json",
    "revocation.json",
    VERIFIER,
}
REVOCATION_PROTOCOL = "evidence_revocation_response_v1"
REVOCATION_PROTOCOL_SHA256 = "55558267bdf7193e5b69905b25b4920f34affbb3c259637f7b0647edbe220c84"
PREREGISTRATION_PROTOCOL = "formal_validation_preregistration_v1"


class AuditError(RuntimeError):
    pass


class NotReady(AuditError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json(value: bytes) -> object:
    try:
        text = value.decode("utf-8")
        def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, child in rows:
                if key in result:
                    raise AuditError("duplicate_json_key")
                result[key] = child
            return result
        result = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(AuditError("nonfinite_json_number")),
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise AuditError("invalid_utf8_or_json") from exc
    try:
        canonical = canonical_bytes(result)
    except (RecursionError, MemoryError, TypeError, ValueError) as exc:
        raise AuditError("json_resource_or_type_limit") from exc
    if canonical != value:
        raise AuditError("noncanonical_json")
    return result


def _object(value: object, *, keys: set[str], reason: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuditError(reason)
    return value


def _list(value: object, *, reason: str, maximum: int = 4096) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise AuditError(reason)
    return value


def _string(value: object, *, reason: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise AuditError(reason)
    return value


def _integer(value: object, *, reason: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuditError(reason)
    return value


def _string_list(value: object, *, reason: str) -> list[str]:
    rows = _list(value, reason=reason)
    if any(not isinstance(item, str) or not item for item in rows):
        raise AuditError(reason)
    strings = [str(item) for item in rows]
    if strings != sorted(set(strings)):
        raise AuditError(reason)
    return strings


def _validate_manifest(value: object) -> dict[str, object]:
    manifest = _object(
        value,
        keys={
            "files", "formal_validation_complete", "generated_from_commit",
            "manifest_self_sha256", "protocol", "schema_version", "source_commit",
            "status", "verifier_sha256",
        },
        reason="manifest_schema_invalid",
    )
    if (
        manifest["protocol"] != PROTOCOL
        or manifest["schema_version"] != SCHEMA
        or manifest["status"] != "verified_with_declared_blockers"
        or manifest["formal_validation_complete"] is not False
        or not isinstance(manifest["source_commit"], str)
        or not COMMIT_RE.fullmatch(manifest["source_commit"])
        or not isinstance(manifest["generated_from_commit"], str)
        or not COMMIT_RE.fullmatch(manifest["generated_from_commit"])
        or not isinstance(manifest["manifest_self_sha256"], str)
        or not SHA256_RE.fullmatch(manifest["manifest_self_sha256"])
        or not isinstance(manifest["verifier_sha256"], str)
        or not SHA256_RE.fullmatch(manifest["verifier_sha256"])
    ):
        raise AuditError("manifest_schema_invalid")
    inventory = _list(manifest["files"], reason="manifest_inventory_invalid", maximum=32)
    paths: set[str] = set()
    for raw in inventory:
        item = _object(raw, keys={"path", "role", "sha256", "size"}, reason="manifest_file_entry_invalid")
        name = _safe_name(_string(item["path"], reason="manifest_file_entry_invalid"))
        if name in paths:
            raise AuditError("duplicate_manifest_inventory_entry")
        paths.add(name)
        if item["role"] not in {"auditable_data", "standalone_verifier"}:
            raise AuditError("manifest_file_entry_invalid")
        if not isinstance(item["sha256"], str) or not SHA256_RE.fullmatch(item["sha256"]):
            raise AuditError("manifest_file_entry_invalid")
        _integer(item["size"], reason="manifest_file_entry_invalid")
    return manifest


def _validate_claims(value: object) -> dict[str, object]:
    document = _object(value, keys={"claims", "protocol", "schema_version"}, reason="claims_schema_invalid")
    if document["protocol"] != PROTOCOL or document["schema_version"] != SCHEMA:
        raise AuditError("claims_schema_invalid")
    seen: set[str] = set()
    blocked: set[str] = set()
    for raw in _list(document["claims"], reason="claims_schema_invalid"):
        row = _object(raw, keys={"boundary", "claim_id", "evidence_ids", "scope", "status"}, reason="claim_entry_invalid")
        identity = _string(row["claim_id"], reason="claim_entry_invalid")
        if identity in seen:
            raise AuditError("duplicate_claim_identity")
        seen.add(identity)
        _string(row["boundary"], reason="claim_entry_invalid")
        _string(row["scope"], reason="claim_entry_invalid")
        if row["status"] not in {"verified", "internal_only", "blocked", "not_applicable"}:
            raise AuditError("claim_entry_invalid")
        if row["status"] == "blocked":
            blocked.add(identity)
        _string_list(row["evidence_ids"], reason="claim_entry_invalid")
    if not seen or blocked != REQUIRED_BLOCKED_CLAIMS:
        raise AuditError("blocked_claim_boundary_drift")
    return document


def _validate_evidence(value: object) -> dict[str, object]:
    document = _object(value, keys={"evidence", "protocol", "schema_version"}, reason="evidence_schema_invalid")
    if document["protocol"] != PROTOCOL or document["schema_version"] != SCHEMA:
        raise AuditError("evidence_schema_invalid")
    seen: set[str] = set()
    for raw in _list(document["evidence"], reason="evidence_schema_invalid"):
        row = _object(raw, keys={"evidence_id", "protocol", "role", "sha256", "size", "verification_scope"}, reason="evidence_entry_invalid")
        identity = _string(row["evidence_id"], reason="evidence_entry_invalid")
        if identity in seen:
            raise AuditError("duplicate_evidence_identity")
        seen.add(identity)
        for key in ("protocol", "role"):
            _string(row[key], reason="evidence_entry_invalid")
        if not isinstance(row["sha256"], str) or not SHA256_RE.fullmatch(row["sha256"]):
            raise AuditError("evidence_entry_invalid")
        _integer(row["size"], reason="evidence_entry_invalid")
        if row["verification_scope"] != "externally_unverifiable_reference":
            raise AuditError("internal_evidence_overclaimed")
    if not seen:
        raise AuditError("evidence_inventory_empty")
    return document


def _validate_blockers(value: object) -> dict[str, object]:
    document = _object(value, keys={"blocker_count", "blockers", "protocol", "schema_version"}, reason="blockers_schema_invalid")
    if document["protocol"] != PROTOCOL or document["schema_version"] != SCHEMA or document["blocker_count"] != 3:
        raise AuditError("blockers_schema_invalid")
    seen: set[str] = set()
    for raw in _list(document["blockers"], reason="blockers_schema_invalid", maximum=3):
        row = _object(raw, keys={"blocker_id", "evidence_ids", "future_integration_point", "missing_external_input", "non_substitutes", "status"}, reason="blocker_entry_invalid")
        identity = _string(row["blocker_id"], reason="blocker_entry_invalid")
        if identity in seen or row["status"] != "blocked":
            raise AuditError("blocker_entry_invalid")
        seen.add(identity)
        _string_list(row["evidence_ids"], reason="blocker_entry_invalid")
        _string_list(row["non_substitutes"], reason="blocker_entry_invalid")
        _string(row["future_integration_point"], reason="blocker_entry_invalid")
        _string(row["missing_external_input"], reason="blocker_entry_invalid")
    if seen != REQUIRED_BLOCKERS:
        raise AuditError("formal_blocker_set_drift")
    return document


def _validate_freshness(value: object) -> dict[str, object]:
    document = _object(value, keys={"baseline_head", "protocol", "stale_count", "status"}, reason="freshness_schema_invalid")
    if (
        not isinstance(document["baseline_head"], str)
        or not COMMIT_RE.fullmatch(document["baseline_head"])
        or document["protocol"] != "validation_evidence_freshness_v1"
        or document["stale_count"] != 0
        or document["status"] != "fresh_with_declared_blockers"
    ):
        raise AuditError("stale_evidence_present")
    return document


def _validate_preregistration(value: object) -> dict[str, object]:
    document = _object(
        value,
        keys={
            "blockers",
            "formal_validation_complete",
            "protocol",
            "protocol_sha256",
            "schema_version",
            "seal_sha256",
            "state",
            "status",
        },
        reason="preregistration_schema_invalid",
    )
    if (
        document["protocol"] != PREREGISTRATION_PROTOCOL
        or document["schema_version"] != SCHEMA
        or document["state"] != "sealed"
        or document["status"] != "sealed_with_external_evidence_blockers"
        or document["formal_validation_complete"] is not False
        or not isinstance(document["protocol_sha256"], str)
        or not SHA256_RE.fullmatch(document["protocol_sha256"])
        or not isinstance(document["seal_sha256"], str)
        or not SHA256_RE.fullmatch(document["seal_sha256"])
    ):
        raise AuditError("preregistration_schema_invalid")
    if _string_list(document["blockers"], reason="preregistration_blockers_invalid") != sorted(REQUIRED_BLOCKERS):
        raise AuditError("preregistration_blockers_invalid")
    return document


def _validate_policy(value: object) -> dict[str, object]:
    document = _object(value, keys={"default_strategy", "deterministic_tiebreak_v2_default_enabled"}, reason="policy_schema_invalid")
    if document != {"default_strategy": "current_rules", "deterministic_tiebreak_v2_default_enabled": False}:
        raise AuditError("default_policy_drift")
    return document


def _validate_readiness(value: object) -> dict[str, object]:
    document = _object(value, keys={"formal_validation_complete", "published_readiness_status", "status"}, reason="readiness_schema_invalid")
    if document != {"formal_validation_complete": False, "published_readiness_status": "ready_with_declared_blockers", "status": "verified_with_declared_blockers"}:
        raise AuditError("formal_status_overclaimed")
    return document


def _validate_revocation(value: object) -> dict[str, object]:
    document = _object(
        value,
        keys={
            "active_incident_count",
            "event_count",
            "formal_validation_complete",
            "ledger_sha256",
            "protocol",
            "protocol_sha256",
            "schema_version",
            "status",
        },
        reason="revocation_schema_invalid",
    )
    if (
        document["protocol"] != REVOCATION_PROTOCOL
        or document["protocol_sha256"] != REVOCATION_PROTOCOL_SHA256
        or document["schema_version"] != SCHEMA
        or document["status"] != "revocation_controls_ready"
        or document["active_incident_count"] != 0
        or not isinstance(document["event_count"], int)
        or isinstance(document["event_count"], bool)
        or document["event_count"] < 0
        or document["formal_validation_complete"] is not False
        or not isinstance(document["ledger_sha256"], str)
        or not SHA256_RE.fullmatch(document["ledger_sha256"])
    ):
        raise AuditError("active_or_invalid_revocation_state")
    return document


def _validate_dependencies(value: object) -> dict[str, object]:
    document = _object(value, keys={"implementation_commit_ancestry", "source_files"}, reason="dependencies_schema_invalid")
    ancestry = _object(document["implementation_commit_ancestry"], keys={"head_commit", "source_commit", "source_is_ancestor"}, reason="implementation_commit_ancestry_invalid")
    if (
        not isinstance(ancestry["head_commit"], str)
        or not COMMIT_RE.fullmatch(ancestry["head_commit"])
        or not isinstance(ancestry["source_commit"], str)
        or not COMMIT_RE.fullmatch(ancestry["source_commit"])
        or ancestry["source_is_ancestor"] is not True
    ):
        raise AuditError("implementation_commit_ancestry_invalid")
    roles: set[str] = set()
    for raw in _list(document["source_files"], reason="source_files_schema_invalid", maximum=16):
        row = _object(raw, keys={"role", "sha256"}, reason="source_file_entry_invalid")
        role = _string(row["role"], reason="source_file_entry_invalid")
        if role in roles or not isinstance(row["sha256"], str) or not SHA256_RE.fullmatch(row["sha256"]):
            raise AuditError("source_file_entry_invalid")
        roles.add(role)
    if roles != REQUIRED_SOURCE_ROLES:
        raise AuditError("source_role_inventory_drift")
    return document


def _safe_name(name: str) -> str:
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or posixpath.isabs(name)
        or ":" in PurePosixPath(name).parts[0]
    ):
        raise AuditError("unsafe_archive_path")
    normalized = unicodedata.normalize("NFC", name)
    parts = PurePosixPath(normalized).parts
    if normalized != name or not parts or any(part in {"", ".", ".."} for part in parts):
        raise AuditError("unsafe_or_normalization_conflict_path")
    return normalized


def _read_archive(path: Path, *, max_files: int = 32, max_file: int = 1048576, max_total: int = 5242880) -> dict[str, bytes]:
    try:
        if not path.is_file():
            raise AuditError("archive_missing")
        rows: dict[str, bytes] = {}
        normalized: set[str] = set()
        total = 0
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > max_files:
                raise AuditError("archive_file_limit_exceeded")
            for info in infos:
                name = _safe_name(info.filename)
                if name in rows or name.casefold() in normalized:
                    raise AuditError("duplicate_or_normalization_conflict_member")
                normalized.add(name.casefold())
                mode = info.external_attr >> 16
                if info.is_dir() or stat.S_ISLNK(mode) or (mode and not stat.S_ISREG(mode)):
                    raise AuditError("links_or_nonfiles_forbidden")
                if info.file_size > max_file or info.compress_size == 0 < info.file_size:
                    raise AuditError("archive_resource_limit_exceeded")
                if info.compress_size and info.file_size / info.compress_size > 20:
                    raise AuditError("compression_ratio_exceeded")
                total += info.file_size
                if total > max_total:
                    raise AuditError("archive_total_limit_exceeded")
                rows[name] = archive.read(info)
        return rows
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError("archive_unreadable") from exc


def _self_hash(manifest: dict[str, object]) -> str:
    copy = dict(manifest)
    copy["manifest_self_sha256"] = "0" * 64
    return digest(canonical_bytes(copy))


def verify_archive(path: Path) -> dict[str, object]:
    rows = _read_archive(path)
    if MANIFEST not in rows:
        raise AuditError("manifest_missing")
    manifest = _validate_manifest(strict_json(rows[MANIFEST]))
    inventory = manifest["files"]
    expected = {MANIFEST}
    for item in inventory:
        name = str(item["path"])
        expected.add(name)
        if name not in rows or len(rows[name]) != item["size"] or digest(rows[name]) != item["sha256"]:
            raise AuditError("member_hash_or_size_mismatch")
    if set(rows) != expected or expected - {MANIFEST} != CANONICAL_MEMBERS:
        raise AuditError("archive_inventory_not_closed")
    if manifest.get("manifest_self_sha256") != _self_hash(manifest):
        raise AuditError("manifest_self_hash_mismatch")
    values: dict[str, object] = {}
    for name in sorted(CANONICAL_MEMBERS - {VERIFIER}):
        values[name] = strict_json(rows[name])
    claims = _validate_claims(values["claims.json"])
    evidence = _validate_evidence(values["evidence_index.json"])
    blockers = _validate_blockers(values["blockers.json"])
    readiness = _validate_readiness(values["readiness.json"])
    freshness = _validate_freshness(values["freshness.json"])
    policy = _validate_policy(values["policy.json"])
    _validate_preregistration(values["preregistration.json"])
    _validate_revocation(values["revocation.json"])
    dependencies = _validate_dependencies(values["protocol_dependencies.json"])
    evidence_ids = {str(row["evidence_id"]) for row in evidence["evidence"]}
    if any(ref not in evidence_ids for row in claims["claims"] for ref in row["evidence_ids"]):
        raise AuditError("claim_evidence_reference_missing")
    ancestry = dependencies["implementation_commit_ancestry"]
    if ancestry["source_commit"] != manifest["source_commit"] or ancestry["head_commit"] != manifest["generated_from_commit"]:
        raise AuditError("implementation_commit_identity_mismatch")
    if digest(rows[VERIFIER]) != manifest.get("verifier_sha256"):
        raise AuditError("verifier_hash_mismatch")
    return {
        "archive_sha256": digest(path.read_bytes()),
        "blocker_count": 3,
        "claim_count": len(claims["claims"]),
        "evidence_reference_count": len(evidence["evidence"]),
        "execution": {"archive_code_execution_count": 0, "external_file_read_count": 0, "network_request_count": 0, "subprocess_count": 0},
        "exit_code": EXIT_PASSED,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA,
        "status": "verified_with_declared_blockers",
    }


def _load(path: Path) -> dict[str, object]:
    try:
        value = strict_json(path.read_bytes())
    except FileNotFoundError as exc:
        raise NotReady("required_shareable_source_missing") from exc
    if not isinstance(value, dict): raise AuditError("source_not_object")
    return value


def _repo_path(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", "..", ".env", "third_party"} for part in pure.parts):
        raise AuditError("unsafe_source_path")
    path = (root / Path(*pure.parts)).resolve()
    try: path.relative_to(root.resolve())
    except ValueError as exc: raise AuditError("source_path_escape") from exc
    return path


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False, env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"})
    if completed.returncode: raise AuditError("git_identity_unavailable")
    return completed.stdout.strip()


def _validate_contract(value: object) -> dict[str, object]:
    contract = _object(
        value,
        keys={
            "archive", "execution", "formal_validation_complete", "policy",
            "protocol", "required_blockers", "schema_version", "source_commit",
            "sources", "status",
        },
        reason="contract_schema_invalid",
    )
    if (
        contract["protocol"] != PROTOCOL
        or contract["schema_version"] != SCHEMA
        or contract["formal_validation_complete"] is not False
        or contract["status"] != "verified_with_declared_blockers"
        or not isinstance(contract["source_commit"], str)
        or not COMMIT_RE.fullmatch(contract["source_commit"])
    ):
        raise AuditError("contract_schema_invalid")
    if set(_string_list(contract["required_blockers"], reason="contract_schema_invalid")) != REQUIRED_BLOCKERS:
        raise AuditError("contract_blocker_set_drift")
    _validate_policy(contract["policy"])
    if contract["execution"] != {
        "archive_code_execution_count": 0,
        "external_file_read_count": 0,
        "network_request_count": 0,
        "subprocess_count": 0,
    }:
        raise AuditError("contract_execution_boundary_drift")
    archive = _object(contract["archive"], keys={"compression", "fixed_mode", "fixed_timestamp", "max_file_bytes", "max_files", "max_total_bytes"}, reason="contract_archive_schema_invalid")
    if (
        archive["compression"] != "stored"
        or archive["fixed_mode"] != 420
        or archive["fixed_timestamp"] != [1980, 1, 1, 0, 0, 0]
        or archive["max_file_bytes"] != 1048576
        or archive["max_files"] != 32
        or archive["max_total_bytes"] != 5242880
    ):
        raise AuditError("contract_archive_boundary_drift")
    sources = _object(contract["sources"], keys=REQUIRED_SOURCE_ROLES, reason="contract_source_inventory_drift")
    for path in sources.values():
        _string(path, reason="contract_source_path_invalid")
    return contract


def _audit_sources(contract_path: Path, root: Path) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, str]]:
    contract = _validate_contract(_load(contract_path))
    sources: dict[str, dict[str, object]] = {}
    source_paths: dict[str, Path] = {}
    for role, raw_path in contract["sources"].items():
        path = _repo_path(root, str(raw_path))
        sources[role] = _load(path)
        source_paths[role] = path
    source_hashes = {role: digest(path.read_bytes()) for role, path in source_paths.items()}

    release_bundle = _object(
        sources["release_bundle"],
        keys={"code_identity", "dependency_graph", "files", "generation_command", "protocol", "schema_version", "status", "verification_command"},
        reason="release_bundle_schema_invalid",
    )
    if release_bundle["protocol"] != "validation_readiness_bundle_v1" or release_bundle["status"] != "ready_with_declared_blockers":
        raise AuditError("release_bundle_status_drift")
    release_files = release_bundle["files"]
    if not isinstance(release_files, dict):
        raise AuditError("release_bundle_inventory_invalid")
    for role, member in {
        "claims": "claims.json",
        "evidence_index": "evidence_index.json",
        "missing_inputs": "missing_inputs.json",
        "readiness": "readiness.json",
    }.items():
        entry = release_files.get(member)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size"}:
            raise AuditError("release_bundle_inventory_invalid")
        path = source_paths[role]
        if entry["sha256"] != source_hashes[role] or entry["size"] != path.stat().st_size:
            raise AuditError("shareable_source_hash_drift")

    claims_source = sources["claims"]
    evidence_source = sources["evidence_index"]
    if not isinstance(claims_source.get("claims"), list) or not isinstance(evidence_source.get("evidence"), list):
        raise AuditError("shareable_source_schema_invalid")
    sanitized_claims = {
        "claims": [
            {key: row[key] for key in ("boundary", "claim_id", "evidence_ids", "scope", "status")}
            for row in claims_source["claims"]
            if isinstance(row, dict)
        ],
        "protocol": PROTOCOL,
        "schema_version": SCHEMA,
    }
    if len(sanitized_claims["claims"]) != len(claims_source["claims"]):
        raise AuditError("shareable_source_schema_invalid")
    sanitized_evidence = {
        "evidence": [
            {
                "evidence_id": row["evidence_id"], "protocol": row["protocol"],
                "role": row["role"], "sha256": row["sha256"], "size": row["size"],
                "verification_scope": "externally_unverifiable_reference",
            }
            for row in evidence_source["evidence"]
            if isinstance(row, dict)
        ],
        "protocol": PROTOCOL,
        "schema_version": SCHEMA,
    }
    if len(sanitized_evidence["evidence"]) != len(evidence_source["evidence"]):
        raise AuditError("shareable_source_schema_invalid")
    claims = _validate_claims(sanitized_claims)
    evidence = _validate_evidence(sanitized_evidence)
    evidence_ids = {str(row["evidence_id"]) for row in evidence["evidence"]}
    if any(ref not in evidence_ids for row in claims["claims"] for ref in row["evidence_ids"]):
        raise AuditError("claim_evidence_reference_missing")

    missing = sources["missing_inputs"]
    _validate_blockers({"blocker_count": missing.get("blocker_count"), "blockers": missing.get("blockers"), "protocol": PROTOCOL, "schema_version": SCHEMA})
    freshness = sources["freshness"]
    state_counts = freshness.get("state_counts")
    if not isinstance(state_counts, dict) or state_counts.get("stale") != 0 or freshness.get("status") != "fresh_with_declared_blockers" or freshness.get("formal_validation_complete") is not False:
        raise AuditError("freshness_source_not_closed")
    readiness = sources["readiness"]
    if readiness.get("status") != "ready_with_declared_blockers" or readiness.get("formal_validation_complete") is not False or readiness.get("blocker_count") != 3:
        raise AuditError("readiness_source_not_closed")
    clearance = sources["clearance"]
    prerequisites = clearance.get("global_prerequisites")
    passed = prerequisites.get("passed") if isinstance(prerequisites, dict) else None
    if not isinstance(passed, list) or set(passed) < {"current_rules_default", "default_tiebreak_unchanged", "fresh"} or clearance.get("formal_validation_complete") is not False:
        raise AuditError("policy_prerequisite_missing")

    readiness_contract = sources["readiness_contract"]
    registered = {
        str(row.get("path")): str(row.get("sha256"))
        for row in readiness_contract.get("evidence", [])
        if isinstance(row, dict)
    }
    for role in ("freshness",):
        relative = str(contract["sources"][role])
        if registered.get(relative) != source_hashes[role]:
            raise AuditError("readiness_registered_source_hash_drift")
    standalone_contract_relative = str(Path(contract_path).resolve().relative_to(root.resolve())).replace(os.sep, "/")
    if registered.get(standalone_contract_relative) != digest(contract_path.read_bytes()):
        raise AuditError("readiness_registered_contract_hash_drift")
    return contract, sources, source_hashes


def _audit_revocation(root: Path) -> dict[str, object]:
    protocol_path = root / "benchmark/evidence_revocation_response_v1_protocol.json"
    ledger_path = root / "benchmark/evidence_revocation_response_v1_ledger.json"
    protocol = _load(protocol_path)
    ledger = _load(ledger_path)
    if (
        protocol.get("protocol") != REVOCATION_PROTOCOL
        or protocol.get("schema_version") != SCHEMA
        or protocol.get("protocol_sha256") != REVOCATION_PROTOCOL_SHA256
    ):
        raise AuditError("revocation_protocol_drift")
    protocol_copy = dict(protocol)
    protocol_copy["protocol_sha256"] = "0" * 64
    if digest(json.dumps(protocol_copy, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")) != REVOCATION_PROTOCOL_SHA256:
        raise AuditError("revocation_protocol_hash_mismatch")
    events = ledger.get("events")
    ledger_copy = dict(ledger)
    ledger_copy["ledger_sha256"] = "0" * 64
    expected_ledger = digest(json.dumps(ledger_copy, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if (
        ledger.get("protocol") != REVOCATION_PROTOCOL
        or ledger.get("protocol_sha256") != REVOCATION_PROTOCOL_SHA256
        or ledger.get("schema_version") != SCHEMA
        or not isinstance(events, list)
        or ledger.get("ledger_sha256") != expected_ledger
        or ledger.get("formal_validation_complete") is not False
    ):
        raise NotReady("active_incident_blocks_standalone_bundle")
    transitions = {
        "active": {"under_investigation", "revoked"},
        "under_investigation": {"revoked"},
        "revoked": {"superseded"},
        "superseded": {"restored"},
        "restored": set(),
    }
    states: dict[str, str] = {}
    previous = "0" * 64
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise AuditError("revocation_event_invalid")
        evidence_id = event.get("evidence_id")
        before = states.get(str(evidence_id), "active")
        event_copy = dict(event)
        event_copy["event_sha256"] = "0" * 64
        if (
            event.get("event_index") != index
            or not isinstance(evidence_id, str)
            or not evidence_id
            or event.get("before_state") != before
            or event.get("after_state") not in transitions.get(before, set())
            or event.get("previous_event_sha256") != previous
            or not isinstance(event.get("event_sha256"), str)
            or not SHA256_RE.fullmatch(event["event_sha256"])
            or digest(json.dumps(event_copy, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")) != event["event_sha256"]
        ):
            raise AuditError("revocation_event_invalid")
        states[evidence_id] = str(event["after_state"])
        previous = str(event["event_sha256"])
    active = sorted(
        evidence_id
        for evidence_id, state_name in states.items()
        if state_name in {"under_investigation", "revoked", "superseded"}
    )
    if active:
        raise NotReady("active_incident_blocks_standalone_bundle")
    return {
        "active_incident_count": 0,
        "event_count": len(events),
        "formal_validation_complete": False,
        "ledger_sha256": ledger["ledger_sha256"],
        "protocol": REVOCATION_PROTOCOL,
        "protocol_sha256": REVOCATION_PROTOCOL_SHA256,
        "schema_version": SCHEMA,
        "status": "revocation_controls_ready",
    }


def _compact_hash(value: object) -> str:
    return digest(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _audit_preregistration(root: Path) -> dict[str, object]:
    protocol = _load(root / "benchmark/formal_validation_preregistration_v1_protocol.json")
    seal = _load(root / "benchmark/formal_validation_preregistration_v1_seal.json")
    protocol_copy = dict(protocol)
    protocol_copy["protocol_sha256"] = "0" * 64
    if (
        protocol.get("protocol") != PREREGISTRATION_PROTOCOL
        or protocol.get("schema_version") != SCHEMA
        or protocol.get("formal_validation_complete") is not False
        or not isinstance(protocol.get("protocol_sha256"), str)
        or not SHA256_RE.fullmatch(str(protocol["protocol_sha256"]))
        or _compact_hash(protocol_copy) != protocol["protocol_sha256"]
    ):
        raise AuditError("preregistration_protocol_drift")
    seal_copy = dict(seal)
    seal_copy["seal_sha256"] = "0" * 64
    blockers = seal.get("blockers")
    if (
        seal.get("protocol") != PREREGISTRATION_PROTOCOL
        or seal.get("contract") != "formal_validation_preregistration_seal_v1"
        or seal.get("schema_version") != SCHEMA
        or seal.get("state") != "sealed"
        or seal.get("formal_validation_complete") is not False
        or seal.get("protocol_sha256") != protocol["protocol_sha256"]
        or not isinstance(seal.get("seal_sha256"), str)
        or not SHA256_RE.fullmatch(str(seal["seal_sha256"]))
        or _compact_hash(seal_copy) != seal["seal_sha256"]
        or not isinstance(blockers, list)
        or sorted(blockers) != sorted(REQUIRED_BLOCKERS)
    ):
        raise AuditError("preregistration_seal_drift")
    registered = seal.get("registered_files")
    if not isinstance(registered, list) or not registered:
        raise AuditError("preregistration_registered_files_invalid")
    for row in registered:
        if not isinstance(row, dict) or set(row) != {"path", "role", "sha256", "size"}:
            raise AuditError("preregistration_registered_file_invalid")
        path = _repo_path(root, str(row["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != row["size"]
            or digest(path.read_bytes()) != row["sha256"]
        ):
            raise AuditError("preregistration_registered_file_drift")
    return {
        "blockers": sorted(REQUIRED_BLOCKERS),
        "formal_validation_complete": False,
        "protocol": PREREGISTRATION_PROTOCOL,
        "protocol_sha256": protocol["protocol_sha256"],
        "schema_version": SCHEMA,
        "seal_sha256": seal["seal_sha256"],
        "state": "sealed",
        "status": "sealed_with_external_evidence_blockers",
    }


def build_archive(contract_path: Path, output: Path, root: Path) -> dict[str, object]:
    contract, sources, source_hashes = _audit_sources(contract_path.resolve(), root)
    revocation = _audit_revocation(root)
    preregistration = _audit_preregistration(root)
    source_commit = str(contract.get("source_commit")); head = _git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, head], cwd=root, check=False).returncode == 0
    if not ancestor: raise AuditError("source_commit_not_ancestor")
    missing = sources["missing_inputs"]
    if {row.get("blocker_id") for row in missing.get("blockers", [])} != REQUIRED_BLOCKERS:
        raise AuditError("required_blocker_set_drift")
    clearance = sources["clearance"]
    if set(clearance.get("global_prerequisites", {}).get("passed", [])) < {"current_rules_default", "default_tiebreak_unchanged", "fresh"}:
        raise AuditError("policy_prerequisite_missing")
    claims_source = sources["claims"]
    evidence_source = sources["evidence_index"]
    payloads: dict[str, bytes] = {}
    payloads["claims.json"] = canonical_bytes({"claims": [{key: row[key] for key in ("boundary", "claim_id", "evidence_ids", "scope", "status")} for row in claims_source["claims"]], "protocol": PROTOCOL, "schema_version": SCHEMA})
    payloads["evidence_index.json"] = canonical_bytes({"evidence": [{"evidence_id": row["evidence_id"], "protocol": row["protocol"], "role": row["role"], "sha256": row["sha256"], "size": row["size"], "verification_scope": "externally_unverifiable_reference"} for row in evidence_source["evidence"]], "protocol": PROTOCOL, "schema_version": SCHEMA})
    payloads["blockers.json"] = canonical_bytes({"blocker_count": 3, "blockers": missing["blockers"], "protocol": PROTOCOL, "schema_version": SCHEMA})
    freshness = sources["freshness"]
    payloads["freshness.json"] = canonical_bytes({"baseline_head": freshness["baseline_head"], "protocol": "validation_evidence_freshness_v1", "stale_count": freshness["state_counts"].get("stale", 0), "status": freshness["status"]})
    payloads["policy.json"] = canonical_bytes(contract["policy"])
    payloads["preregistration.json"] = canonical_bytes(preregistration)
    payloads["readiness.json"] = canonical_bytes({"formal_validation_complete": False, "published_readiness_status": sources["readiness"]["status"], "status": "verified_with_declared_blockers"})
    payloads["revocation.json"] = canonical_bytes(revocation)
    payloads["protocol_dependencies.json"] = canonical_bytes({"implementation_commit_ancestry": {"head_commit": head, "source_commit": source_commit, "source_is_ancestor": ancestor}, "source_files": [{"role": key, "sha256": source_hashes[key]} for key in sorted(source_hashes)]})
    verifier_bytes = Path(__file__).read_bytes()
    payloads[VERIFIER] = verifier_bytes
    inventory = [{"path": name, "role": "standalone_verifier" if name == VERIFIER else "auditable_data", "sha256": digest(content), "size": len(content)} for name, content in sorted(payloads.items())]
    manifest: dict[str, object] = {"files": inventory, "formal_validation_complete": False, "generated_from_commit": head, "manifest_self_sha256": "0" * 64, "protocol": PROTOCOL, "schema_version": SCHEMA, "source_commit": source_commit, "status": "verified_with_declared_blockers", "verifier_sha256": digest(verifier_bytes)}
    manifest["manifest_self_sha256"] = _self_hash(manifest)
    payloads[MANIFEST] = canonical_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, tuple(contract["archive"]["fixed_timestamp"])); info.compress_type = zipfile.ZIP_STORED; info.external_attr = 0o100644 << 16; info.create_system = 3
            archive.writestr(info, content)
    os.replace(temporary, output)
    return verify_archive(output)


def _emit(value: dict[str, object]) -> None:
    sys.stdout.buffer.write(canonical_bytes(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build"); build.add_argument("--contract", default="benchmark/standalone_auditor_bundle_v1_contract.json"); build.add_argument("--output", required=True); build.add_argument("--repository-root", default=".")
    verify = commands.add_parser("verify"); verify.add_argument("archive")
    compare = commands.add_parser("compare"); compare.add_argument("first"); compare.add_argument("second")
    audit = commands.add_parser("audit-readiness"); audit.add_argument("--contract", default="benchmark/standalone_auditor_bundle_v1_contract.json"); audit.add_argument("--repository-root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "build": report = build_archive(Path(args.contract), Path(args.output), Path(args.repository_root).resolve())
        elif args.command == "verify": report = verify_archive(Path(args.archive))
        elif args.command == "compare":
            first, second = Path(args.first), Path(args.second); one, two = verify_archive(first), verify_archive(second)
            if first.read_bytes() != second.read_bytes() or canonical_bytes(one) != canonical_bytes(two): raise AuditError("archive_or_report_byte_mismatch")
            report = {**one, "comparison": "byte_identical"}
        else:
            root = Path(args.repository_root).resolve()
            contract_path = Path(args.contract)
            if not contract_path.is_absolute():
                contract_path = root / contract_path
            _contract, sources, _hashes = _audit_sources(contract_path.resolve(), root)
            _audit_revocation(root)
            _audit_preregistration(root)
            report = {
                "blocker_count": 3,
                "claim_count": len(sources["claims"]["claims"]),
                "evidence_reference_count": len(sources["evidence_index"]["evidence"]),
                "exit_code": 0,
                "formal_validation_complete": False,
                "protocol": PROTOCOL,
                "schema_version": SCHEMA,
                "shareable_source_count": len(REQUIRED_SOURCE_ROLES),
                "status": "verified_with_declared_blockers",
            }
        _emit(report); return EXIT_PASSED
    except NotReady as exc:
        _emit({"error_code": str(exc), "exit_code": EXIT_NOT_READY, "protocol": PROTOCOL, "schema_version": SCHEMA, "status": "not_ready_missing_shareable_evidence"}); return EXIT_NOT_READY
    except Exception as exc:
        _emit({"error_code": str(exc) if isinstance(exc, AuditError) else "controlled_input_failure", "exit_code": EXIT_VIOLATION, "protocol": PROTOCOL, "schema_version": SCHEMA, "status": "integrity_or_claim_violation"}); return EXIT_VIOLATION
    except SystemExit as exc:
        if exc.code == 0: raise
        _emit({"exit_code": EXIT_USAGE, "protocol": PROTOCOL, "schema_version": SCHEMA, "status": "usage_error"}); return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
