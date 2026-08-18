"""Deterministic append-only transparency log for public validation evidence.

The log binds release-facing evidence summaries without authenticating the
publisher and without copying private evidence payloads.  Current tracked
state is a candidate-only checkpoint built from immutable Git blobs at the
protocol source commit; it is not a formal release or quality result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL = "evidence_transparency_log_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "f764eb3c0849c53512f9326d7a83429e1c430a7b"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NO_PUBLIC_CHECKPOINT = 3
EXIT_USAGE = 4
ZERO_SHA256 = "0" * 64
HEX = frozenset("0123456789abcdef")
REAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
RELEASE_STATUSES = (
    "candidate_only",
    "public_with_declared_blockers",
    "formal_release",
)
IDENTITY_AUTHENTICATION_BOUNDARY = (
    "not_provided_hash_chain_proves_content_consistency_not_publisher_identity"
)
EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
SOURCE_PATHS = {
    "blockers": "benchmark/validation_readiness_bundle_v1_release/missing_inputs.json",
    "claims": "benchmark/validation_readiness_bundle_v1_release/claims.json",
    "clearance": "benchmark/formal_validation_clearance_v1_evidence/current.json",
    "freshness": "benchmark/validation_evidence_freshness_v1_evidence/current.json",
    "readiness": "benchmark/validation_readiness_bundle_v1_release/readiness.json",
    "release_candidate": (
        "benchmark/release_candidate_reproducibility_v1_evidence/current.json"
    ),
    "revocation": "benchmark/evidence_revocation_response_v1_ledger.json",
    "standalone": "benchmark/standalone_auditor_bundle_v1_evidence/readiness.json",
}
ARTIFACT_ROLES = (
    "clearance_receipt",
    "readiness",
    "release_candidate",
    "standalone",
)
RECORD_KEYS = {
    "artifacts",
    "blockers",
    "claims",
    "code_commit",
    "content_sha256",
    "evidence_epoch",
    "formal_validation_complete",
    "freshness",
    "previous_record_sha256",
    "publisher_identity_authentication",
    "release_identity",
    "release_status",
    "revocation",
    "sequence",
    "supersession",
}
LOG_KEYS = {
    "execution",
    "formal_validation_complete",
    "log_sha256",
    "merkle_root",
    "protocol",
    "record_count",
    "records",
    "schema_version",
    "source_commit",
}
CHECKPOINT_KEYS = {
    "blockers",
    "checkpoint_sha256",
    "formal_validation_complete",
    "identity_authentication",
    "latest_release_identity",
    "latest_release_status",
    "log_length",
    "log_sha256",
    "merkle_root",
    "protocol",
    "public_release_count",
    "schema_version",
    "source_commit",
    "status",
    "verification_command",
}


class TransparencyError(RuntimeError):
    """A log, proof, release, or source binding is invalid."""


class TransparencyNotReady(TransparencyError):
    """Controls are valid while no public release checkpoint exists."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        if key in value:
            raise TransparencyError("duplicate_json_key")
        value[key] = child
    return value


def _depth(value: Any, level: int = 0) -> None:
    if level > 64:
        raise TransparencyError("json_nesting_limit")
    if isinstance(value, Mapping):
        if len(value) > 4096:
            raise TransparencyError("json_member_limit")
        for child in value.values():
            _depth(child, level + 1)
    elif isinstance(value, list):
        if len(value) > 10000:
            raise TransparencyError("json_member_limit")
        for child in value:
            _depth(child, level + 1)


def parse_json_bytes(value: bytes) -> dict[str, Any]:
    if len(value) > 8 * 1024 * 1024:
        raise TransparencyError("json_size_limit")
    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                TransparencyError("nonfinite_json_number")
            ),
        )
    except TransparencyError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, MemoryError) as exc:
        raise TransparencyError("json_input_invalid") from exc
    if not isinstance(parsed, dict):
        raise TransparencyError("json_root_not_object")
    _depth(parsed)
    return parsed


def read_json(path: Path) -> dict[str, Any]:
    try:
        return parse_json_bytes(path.read_bytes())
    except OSError as exc:
        raise TransparencyError("json_input_unavailable") from exc


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(value))
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        raise TransparencyError("json_output_unavailable") from exc


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX


def _require_keys(value: Any, keys: set[str], reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise TransparencyError(reason)
    return value


def _safe_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", "..", ".env"} for part in value.parts)
        or value.parts[0] == "third_party"
    ):
        raise TransparencyError("unsafe_source_path")
    result = (root / Path(*value.parts)).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise TransparencyError("source_path_escape") from exc
    return result


def _protocol_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["protocol_sha256"] = ZERO_SHA256
    return payload


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    _require_keys(
        value,
        {
            "checkpoint",
            "execution",
            "formal_validation_complete",
            "identity_authentication",
            "merkle",
            "protocol",
            "protocol_sha256",
            "real_blockers",
            "record_contract",
            "release_statuses",
            "schema_version",
            "source_commit",
            "sources",
        },
        "protocol_schema_invalid",
    )
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION
        or value["real_blockers"] != list(REAL_BLOCKERS)
        or value["release_statuses"] != list(RELEASE_STATUSES)
        or value["identity_authentication"] != IDENTITY_AUTHENTICATION_BOUNDARY
        or value["sources"] != SOURCE_PATHS
        or value["record_contract"]
        != {
            "content_hash": "sha256_canonical_json_with_zeroed_content_sha256",
            "previous_hash": "previous_record_content_sha256",
            "sequence": "zero_based_contiguous",
            "same_release_identity_conflict": "forbidden",
            "rollback": "previously_retired_artifact_digest_forbidden",
            "supersession": "prior_release_and_revocation_event_required",
        }
        or value["merkle"]
        != {
            "empty_root": "sha256_empty_bytes",
            "leaf": "sha256_0x00_plus_canonical_record",
            "node": "sha256_0x01_plus_left_plus_right",
            "split": "largest_power_of_two_less_than_tree_size",
            "consistency_proof": "full_prefix_leaf_hashes_v1",
        }
        or value["checkpoint"]
        != {
            "candidate_status": "candidate_checkpoint_no_public_release",
            "formal_completion": False,
            "verification_command": (
                "PYTHONPATH=src python "
                "scripts/check_evidence_transparency.py verify-log"
            ),
        }
    ):
        raise TransparencyError("protocol_policy_drift")
    if (
        not _is_digest(value["protocol_sha256"])
        or stable_hash(_protocol_payload(value)) != value["protocol_sha256"]
    ):
        raise TransparencyError("protocol_hash_mismatch")
    return value


def _git_blob(root: Path, commit: str, relative: str) -> bytes:
    _safe_path(root, relative)
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=30,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TransparencyError("source_blob_unavailable") from exc
    if completed.returncode != 0:
        raise TransparencyError("source_blob_unavailable")
    return completed.stdout


def _source_snapshot(
    repository_root: Path, protocol: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    values: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for role, relative in sorted(protocol["sources"].items()):
        raw = _git_blob(repository_root, protocol["source_commit"], str(relative))
        values[role] = parse_json_bytes(raw)
        hashes[role] = sha256_bytes(raw)
    return values, hashes


def _record_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["content_sha256"] = ZERO_SHA256
    return payload


def finalize_record(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["content_sha256"] = ZERO_SHA256
    result["content_sha256"] = stable_hash(result)
    return result


def _artifact(
    *, sha256: str | None, status: str, formal_validation_complete: bool
) -> dict[str, Any]:
    return {
        "formal_validation_complete": formal_validation_complete,
        "sha256": sha256,
        "status": status,
    }


def build_candidate_record(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    sources, hashes = _source_snapshot(repository_root, protocol)
    readiness = sources["readiness"]
    standalone = sources["standalone"]
    release = sources["release_candidate"]
    clearance = sources["clearance"]
    freshness = sources["freshness"]
    revocation = sources["revocation"]
    claims = sources["claims"]
    blockers = sources["blockers"]
    if (
        readiness.get("status") != "ready_with_declared_blockers"
        or readiness.get("blocker_count") != 3
        or readiness.get("formal_validation_complete") is not False
        or standalone.get("status") != "verified_with_declared_blockers"
        or standalone.get("formal_validation_complete") is not False
        or clearance.get("status") not in {"blocked", "partially_satisfied"}
        or clearance.get("formal_validation_complete") is not False
        or freshness.get("status") != "fresh_with_declared_blockers"
        or (freshness.get("state_counts") or {}).get("stale") != 0
        or revocation.get("events") != []
        or blockers.get("blocker_count") != 3
    ):
        raise TransparencyError("candidate_source_state_invalid")
    blocker_ids = sorted(
        str(row.get("blocker_id")) for row in blockers.get("blockers") or []
    )
    if blocker_ids != sorted(REAL_BLOCKERS):
        raise TransparencyError("candidate_blocker_set_drift")
    claim_rows = claims.get("claims")
    if not isinstance(claim_rows, list):
        raise TransparencyError("candidate_claim_inventory_invalid")
    status_counts: dict[str, int] = {}
    claim_ids: list[str] = []
    for row in claim_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("claim_id"), str):
            raise TransparencyError("candidate_claim_inventory_invalid")
        claim_ids.append(str(row["claim_id"]))
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    if len(claim_ids) != len(set(claim_ids)):
        raise TransparencyError("candidate_claim_inventory_invalid")
    artifact_seed = {
        "readiness": hashes["readiness"],
        "standalone": hashes["standalone"],
        "release_candidate": hashes["release_candidate"],
        "clearance": hashes["clearance"],
    }
    release_identity = (
        f"candidate:{protocol['source_commit']}:"
        f"{stable_hash(artifact_seed)[:16]}"
    )
    record = {
        "artifacts": {
            "clearance_receipt": {
                "evidence_sha256": None,
                "formal_validation_complete": False,
                "sha256": None,
                "source_commit": None,
                "status": "not_available",
            },
            "readiness": _artifact(
                sha256=hashes["readiness"],
                status=str(readiness["status"]),
                formal_validation_complete=False,
            ),
            "release_candidate": {
                **_artifact(
                    sha256=hashes["release_candidate"],
                    status=str(release.get("status")),
                    formal_validation_complete=False,
                ),
                "qualification": str(
                    release.get("qualification") or "not_qualified"
                ),
            },
            "standalone": _artifact(
                sha256=hashes["standalone"],
                status=str(standalone["status"]),
                formal_validation_complete=False,
            ),
        },
        "blockers": list(REAL_BLOCKERS),
        "claims": {
            "claim_count": len(claim_rows),
            "claims_sha256": hashes["claims"],
            "identity_sha256": stable_hash(sorted(claim_ids)),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "code_commit": protocol["source_commit"],
        "content_sha256": ZERO_SHA256,
        "evidence_epoch": 0,
        "formal_validation_complete": False,
        "freshness": {
            "sha256": hashes["freshness"],
            "stale_count": 0,
            "status": "fresh_with_declared_blockers",
        },
        "previous_record_sha256": ZERO_SHA256,
        "publisher_identity_authentication": IDENTITY_AUTHENTICATION_BOUNDARY,
        "release_identity": release_identity,
        "release_status": "candidate_only",
        "revocation": {
            "active_incident_count": 0,
            "event_count": 0,
            "ledger_sha256": str(revocation["ledger_sha256"]),
            "status": "revocation_controls_ready",
        },
        "sequence": 0,
        "supersession": {
            "revocation_event_sha256": None,
            "supersedes_release_identity": None,
        },
    }
    return finalize_record(record)


def _validate_artifact(
    value: Any, role: str, *, formal_release: bool
) -> None:
    if role == "clearance_receipt":
        row = _require_keys(
            value,
            {
                "evidence_sha256",
                "formal_validation_complete",
                "sha256",
                "source_commit",
                "status",
            },
            "clearance_receipt_schema_invalid",
        )
        if formal_release:
            if (
                row["status"] != "cleared"
                or row["formal_validation_complete"] is not True
                or not _is_digest(row["sha256"])
                or not _is_digest(row["evidence_sha256"])
                or not _is_commit(row["source_commit"])
            ):
                raise TransparencyError("clearance_receipt_binding_invalid")
        elif row != {
            "evidence_sha256": None,
            "formal_validation_complete": False,
            "sha256": None,
            "source_commit": None,
            "status": "not_available",
        }:
            raise TransparencyError("unexpected_clearance_receipt")
        return
    expected = {
        "formal_validation_complete",
        "sha256",
        "status",
    }
    if role == "release_candidate":
        expected.add("qualification")
    row = _require_keys(value, expected, "artifact_summary_schema_invalid")
    if (
        not _is_digest(row["sha256"])
        or not isinstance(row["status"], str)
        or not row["status"]
        or row["formal_validation_complete"] is not formal_release
    ):
        raise TransparencyError("artifact_summary_invalid")
    if role == "release_candidate" and row["qualification"] not in {
        "qualified",
        "not_qualified",
    }:
        raise TransparencyError("release_qualification_invalid")
    if (
        formal_release
        and role == "release_candidate"
        and row["qualification"] != "qualified"
    ):
        raise TransparencyError("formal_release_candidate_not_qualified")


def validate_record(
    value: Mapping[str, Any],
    *,
    expected_sequence: int,
    expected_previous: str,
    prior_release_ids: set[str],
) -> None:
    _require_keys(value, RECORD_KEYS, "record_schema_invalid")
    if (
        value["sequence"] != expected_sequence
        or value["evidence_epoch"] != expected_sequence
        or value["previous_record_sha256"] != expected_previous
        or not isinstance(value["release_identity"], str)
        or not value["release_identity"]
        or value["release_identity"] in prior_release_ids
        or value["release_status"] not in RELEASE_STATUSES
        or not _is_commit(value["code_commit"])
        or value["publisher_identity_authentication"]
        != IDENTITY_AUTHENTICATION_BOUNDARY
        or not _is_digest(value["content_sha256"])
        or stable_hash(_record_payload(value)) != value["content_sha256"]
    ):
        raise TransparencyError("record_identity_or_hash_invalid")
    formal_release = value["release_status"] == "formal_release"
    if value["formal_validation_complete"] is not formal_release:
        raise TransparencyError("record_formal_status_invalid")
    artifacts = _require_keys(
        value["artifacts"], set(ARTIFACT_ROLES), "artifact_inventory_invalid"
    )
    for role in ARTIFACT_ROLES:
        _validate_artifact(artifacts[role], role, formal_release=formal_release)
    blockers = value["blockers"]
    expected_blockers = [] if formal_release else list(REAL_BLOCKERS)
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or blockers != expected_blockers
    ):
        raise TransparencyError("formal_blocker_set_hidden_or_invalid")
    claims = _require_keys(
        value["claims"],
        {"claim_count", "claims_sha256", "identity_sha256", "status_counts"},
        "claims_summary_invalid",
    )
    if (
        isinstance(claims["claim_count"], bool)
        or not isinstance(claims["claim_count"], int)
        or claims["claim_count"] < 1
        or not _is_digest(claims["claims_sha256"])
        or not _is_digest(claims["identity_sha256"])
        or not isinstance(claims["status_counts"], Mapping)
        or any(
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for key, count in claims["status_counts"].items()
        )
        or sum(claims["status_counts"].values()) != claims["claim_count"]
    ):
        raise TransparencyError("claims_summary_invalid")
    freshness = _require_keys(
        value["freshness"],
        {"sha256", "stale_count", "status"},
        "freshness_summary_invalid",
    )
    if (
        freshness["status"] != "fresh_with_declared_blockers"
        or freshness["stale_count"] != 0
        or not _is_digest(freshness["sha256"])
    ):
        raise TransparencyError("stale_evidence_hidden")
    revocation = _require_keys(
        value["revocation"],
        {"active_incident_count", "event_count", "ledger_sha256", "status"},
        "revocation_summary_invalid",
    )
    if (
        revocation["status"] != "revocation_controls_ready"
        or revocation["active_incident_count"] != 0
        or isinstance(revocation["event_count"], bool)
        or not isinstance(revocation["event_count"], int)
        or revocation["event_count"] < 0
        or not _is_digest(revocation["ledger_sha256"])
    ):
        raise TransparencyError("active_or_revoked_evidence_published")
    supersession = _require_keys(
        value["supersession"],
        {"revocation_event_sha256", "supersedes_release_identity"},
        "supersession_schema_invalid",
    )
    target = supersession["supersedes_release_identity"]
    event_hash = supersession["revocation_event_sha256"]
    if target is None:
        if event_hash is not None:
            raise TransparencyError("unexpected_supersession_event")
    elif (
        not isinstance(target, str)
        or target not in prior_release_ids
        or not _is_digest(event_hash)
    ):
        raise TransparencyError("supersession_history_or_event_invalid")
    receipt = artifacts["clearance_receipt"]
    if formal_release and receipt["source_commit"] != value["code_commit"]:
        raise TransparencyError("clearance_receipt_commit_mismatch")


def leaf_hash(record: Mapping[str, Any]) -> str:
    return sha256_bytes(b"\x00" + canonical_json(record))


def _largest_power_of_two_less_than(value: int) -> int:
    if value < 2:
        raise TransparencyError("merkle_split_invalid")
    return 1 << ((value - 1).bit_length() - 1)


def _merkle_root_bytes(leaves: Sequence[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return leaves[0]
    split = _largest_power_of_two_less_than(len(leaves))
    return hashlib.sha256(
        b"\x01"
        + _merkle_root_bytes(leaves[:split])
        + _merkle_root_bytes(leaves[split:])
    ).digest()


def merkle_root_from_leaf_hashes(values: Sequence[str]) -> str:
    if any(not _is_digest(value) for value in values):
        raise TransparencyError("merkle_leaf_hash_invalid")
    return _merkle_root_bytes([bytes.fromhex(value) for value in values]).hex()


def merkle_root(records: Sequence[Mapping[str, Any]]) -> str:
    return merkle_root_from_leaf_hashes([leaf_hash(record) for record in records])


def _log_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload["log_sha256"] = ZERO_SHA256
    return payload


def build_log(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = {
        "execution": EXECUTION,
        "formal_validation_complete": False,
        "log_sha256": ZERO_SHA256,
        "merkle_root": merkle_root(records),
        "protocol": PROTOCOL,
        "record_count": len(records),
        "records": [copy.deepcopy(dict(record)) for record in records],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
    }
    value["formal_validation_complete"] = bool(records) and bool(
        records[-1]["formal_validation_complete"]
    )
    value["log_sha256"] = stable_hash(value)
    verify_log(value)
    return value


def _artifact_digest(record: Mapping[str, Any], role: str) -> str | None:
    artifact = record["artifacts"][role]
    return artifact.get("sha256")


def verify_log(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(value, LOG_KEYS, "log_schema_invalid")
    records = value["records"]
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["execution"] != EXECUTION
        or not isinstance(records, list)
        or value["record_count"] != len(records)
        or not records
        or not _is_digest(value["merkle_root"])
        or not _is_digest(value["log_sha256"])
        or stable_hash(_log_payload(value)) != value["log_sha256"]
    ):
        raise TransparencyError("log_identity_or_hash_invalid")
    prior_ids: set[str] = set()
    previous = ZERO_SHA256
    retired_digests = {role: set() for role in ARTIFACT_ROLES if role != "clearance_receipt"}
    latest_digest = {role: None for role in retired_digests}
    retired_commits: set[str] = set()
    latest_commit: str | None = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TransparencyError("record_not_object")
        validate_record(
            record,
            expected_sequence=index,
            expected_previous=previous,
            prior_release_ids=prior_ids,
        )
        for role in retired_digests:
            digest = _artifact_digest(record, role)
            if digest in retired_digests[role]:
                raise TransparencyError("artifact_digest_rollback")
            if latest_digest[role] is not None and digest != latest_digest[role]:
                retired_digests[role].add(str(latest_digest[role]))
            latest_digest[role] = digest
        commit = str(record["code_commit"])
        if commit in retired_commits:
            raise TransparencyError("code_commit_rollback")
        if latest_commit is not None and commit != latest_commit:
            retired_commits.add(latest_commit)
        latest_commit = commit
        prior_ids.add(str(record["release_identity"]))
        previous = str(record["content_sha256"])
    expected_root = merkle_root(records)
    if value["merkle_root"] != expected_root:
        raise TransparencyError("merkle_root_mismatch")
    if value["formal_validation_complete"] is not bool(
        records[-1]["formal_validation_complete"]
    ):
        raise TransparencyError("log_formal_status_mismatch")
    public_count = sum(
        record["release_status"] != "candidate_only" for record in records
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "transparency_controls_ready",
        "exit_code": EXIT_READY,
        "record_count": len(records),
        "public_release_count": public_count,
        "merkle_root": value["merkle_root"],
        "log_sha256": value["log_sha256"],
        "latest_release_identity": records[-1]["release_identity"],
        "latest_release_status": records[-1]["release_status"],
        "formal_validation_complete": value["formal_validation_complete"],
        "identity_authentication": IDENTITY_AUTHENTICATION_BOUNDARY,
        "execution": EXECUTION,
    }


def append_record(
    log: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    verify_log(log)
    records = [copy.deepcopy(dict(item)) for item in log["records"]]
    candidate = copy.deepcopy(dict(record))
    if candidate.get("sequence") != len(records):
        raise TransparencyError("append_sequence_invalid")
    if candidate.get("previous_record_sha256") != records[-1]["content_sha256"]:
        raise TransparencyError("append_previous_hash_invalid")
    records.append(candidate)
    return build_log(records)


def _inclusion_path(
    leaf_hashes: Sequence[str], index: int
) -> list[dict[str, str]]:
    if len(leaf_hashes) == 1:
        return []
    split = _largest_power_of_two_less_than(len(leaf_hashes))
    if index < split:
        return [
            *_inclusion_path(leaf_hashes[:split], index),
            {
                "side": "right",
                "sha256": merkle_root_from_leaf_hashes(leaf_hashes[split:]),
            },
        ]
    return [
        *_inclusion_path(leaf_hashes[split:], index - split),
        {
            "side": "left",
            "sha256": merkle_root_from_leaf_hashes(leaf_hashes[:split]),
        },
    ]


def inclusion_proof(log: Mapping[str, Any], sequence: int) -> dict[str, Any]:
    verify_log(log)
    if sequence < 0 or sequence >= len(log["records"]):
        raise TransparencyError("inclusion_sequence_out_of_range")
    leaves = [leaf_hash(record) for record in log["records"]]
    proof = {
        "leaf_hash": leaves[sequence],
        "leaf_index": sequence,
        "path": _inclusion_path(leaves, sequence),
        "proof_sha256": ZERO_SHA256,
        "proof_type": "merkle_inclusion_v1",
        "protocol": PROTOCOL,
        "root_hash": log["merkle_root"],
        "schema_version": SCHEMA_VERSION,
        "tree_size": len(leaves),
    }
    proof["proof_sha256"] = stable_hash(proof)
    verify_inclusion_proof(proof)
    return proof


def verify_inclusion_proof(proof: Mapping[str, Any]) -> None:
    _require_keys(
        proof,
        {
            "leaf_hash",
            "leaf_index",
            "path",
            "proof_sha256",
            "proof_type",
            "protocol",
            "root_hash",
            "schema_version",
            "tree_size",
        },
        "inclusion_proof_schema_invalid",
    )
    payload = dict(proof)
    claimed = payload["proof_sha256"]
    payload["proof_sha256"] = ZERO_SHA256
    if (
        proof["protocol"] != PROTOCOL
        or proof["schema_version"] != SCHEMA_VERSION
        or proof["proof_type"] != "merkle_inclusion_v1"
        or not _is_digest(claimed)
        or stable_hash(payload) != claimed
        or not _is_digest(proof["leaf_hash"])
        or not _is_digest(proof["root_hash"])
        or isinstance(proof["leaf_index"], bool)
        or not isinstance(proof["leaf_index"], int)
        or isinstance(proof["tree_size"], bool)
        or not isinstance(proof["tree_size"], int)
        or not 0 <= proof["leaf_index"] < proof["tree_size"]
        or not isinstance(proof["path"], list)
    ):
        raise TransparencyError("inclusion_proof_invalid")
    current = bytes.fromhex(proof["leaf_hash"])
    for raw in proof["path"]:
        row = _require_keys(
            raw, {"sha256", "side"}, "inclusion_proof_path_invalid"
        )
        if row["side"] not in {"left", "right"} or not _is_digest(row["sha256"]):
            raise TransparencyError("inclusion_proof_path_invalid")
        sibling = bytes.fromhex(row["sha256"])
        current = hashlib.sha256(
            b"\x01"
            + (sibling + current if row["side"] == "left" else current + sibling)
        ).digest()
    if current.hex() != proof["root_hash"]:
        raise TransparencyError("inclusion_proof_root_mismatch")


def consistency_proof(
    old_log: Mapping[str, Any], new_log: Mapping[str, Any]
) -> dict[str, Any]:
    verify_log(old_log)
    verify_log(new_log)
    old_records = old_log["records"]
    new_records = new_log["records"]
    if len(old_records) > len(new_records) or any(
        canonical_json(old) != canonical_json(new)
        for old, new in zip(old_records, new_records)
    ):
        raise TransparencyError("logs_do_not_share_immutable_prefix")
    old_leaves = [leaf_hash(record) for record in old_records]
    new_leaves = [leaf_hash(record) for record in new_records]
    proof = {
        "extension_leaf_hashes": new_leaves[len(old_leaves) :],
        "new_root": new_log["merkle_root"],
        "new_size": len(new_leaves),
        "old_root": old_log["merkle_root"],
        "old_size": len(old_leaves),
        "prefix_leaf_hashes": old_leaves,
        "proof_sha256": ZERO_SHA256,
        "proof_type": "merkle_consistency_full_prefix_v1",
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
    }
    proof["proof_sha256"] = stable_hash(proof)
    verify_consistency_proof(proof)
    return proof


def verify_consistency_proof(proof: Mapping[str, Any]) -> None:
    _require_keys(
        proof,
        {
            "extension_leaf_hashes",
            "new_root",
            "new_size",
            "old_root",
            "old_size",
            "prefix_leaf_hashes",
            "proof_sha256",
            "proof_type",
            "protocol",
            "schema_version",
        },
        "consistency_proof_schema_invalid",
    )
    payload = dict(proof)
    claimed = payload["proof_sha256"]
    payload["proof_sha256"] = ZERO_SHA256
    prefix = proof["prefix_leaf_hashes"]
    extension = proof["extension_leaf_hashes"]
    if (
        proof["protocol"] != PROTOCOL
        or proof["schema_version"] != SCHEMA_VERSION
        or proof["proof_type"] != "merkle_consistency_full_prefix_v1"
        or not _is_digest(claimed)
        or stable_hash(payload) != claimed
        or not isinstance(prefix, list)
        or not isinstance(extension, list)
        or proof["old_size"] != len(prefix)
        or proof["new_size"] != len(prefix) + len(extension)
        or proof["old_size"] < 1
        or proof["new_size"] < proof["old_size"]
        or merkle_root_from_leaf_hashes(prefix) != proof["old_root"]
        or merkle_root_from_leaf_hashes([*prefix, *extension])
        != proof["new_root"]
    ):
        raise TransparencyError("consistency_proof_invalid")


def _checkpoint_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["checkpoint_sha256"] = ZERO_SHA256
    return payload


def build_checkpoint(log: Mapping[str, Any]) -> dict[str, Any]:
    report = verify_log(log)
    status = (
        "public_release_checkpoint"
        if report["public_release_count"]
        else "candidate_checkpoint_no_public_release"
    )
    latest = log["records"][-1]
    checkpoint = {
        "blockers": list(latest["blockers"]),
        "checkpoint_sha256": ZERO_SHA256,
        "formal_validation_complete": log["formal_validation_complete"],
        "identity_authentication": IDENTITY_AUTHENTICATION_BOUNDARY,
        "latest_release_identity": latest["release_identity"],
        "latest_release_status": latest["release_status"],
        "log_length": len(log["records"]),
        "log_sha256": log["log_sha256"],
        "merkle_root": log["merkle_root"],
        "protocol": PROTOCOL,
        "public_release_count": report["public_release_count"],
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "status": status,
        "verification_command": (
            "PYTHONPATH=src python scripts/check_evidence_transparency.py verify-log"
        ),
    }
    checkpoint["checkpoint_sha256"] = stable_hash(checkpoint)
    verify_checkpoint(checkpoint, log)
    return checkpoint


def verify_checkpoint(
    checkpoint: Mapping[str, Any], log: Mapping[str, Any]
) -> dict[str, Any]:
    report = verify_log(log)
    _require_keys(checkpoint, CHECKPOINT_KEYS, "checkpoint_schema_invalid")
    payload = dict(checkpoint)
    claimed = payload["checkpoint_sha256"]
    payload["checkpoint_sha256"] = ZERO_SHA256
    expected_status = (
        "public_release_checkpoint"
        if report["public_release_count"]
        else "candidate_checkpoint_no_public_release"
    )
    latest = log["records"][-1]
    if (
        checkpoint["protocol"] != PROTOCOL
        or checkpoint["schema_version"] != SCHEMA_VERSION
        or checkpoint["source_commit"] != SOURCE_COMMIT
        or checkpoint["status"] != expected_status
        or checkpoint["log_length"] != len(log["records"])
        or checkpoint["log_sha256"] != log["log_sha256"]
        or checkpoint["merkle_root"] != log["merkle_root"]
        or checkpoint["latest_release_identity"] != latest["release_identity"]
        or checkpoint["latest_release_status"] != latest["release_status"]
        or checkpoint["public_release_count"] != report["public_release_count"]
        or checkpoint["blockers"] != latest["blockers"]
        or checkpoint["formal_validation_complete"]
        is not log["formal_validation_complete"]
        or checkpoint["identity_authentication"]
        != IDENTITY_AUTHENTICATION_BOUNDARY
        or checkpoint["verification_command"]
        != "PYTHONPATH=src python scripts/check_evidence_transparency.py verify-log"
        or not _is_digest(claimed)
        or stable_hash(payload) != claimed
    ):
        raise TransparencyError("checkpoint_log_or_policy_mismatch")
    return {
        **report,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_status": checkpoint["status"],
    }


def build_current(
    repository_root: Path, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = build_candidate_record(repository_root, protocol)
    log = build_log([record])
    return log, build_checkpoint(log)


def audit_current(
    repository_root: Path,
    protocol: Mapping[str, Any],
    log: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    expected_log, expected_checkpoint = build_current(repository_root, protocol)
    if log != expected_log or checkpoint != expected_checkpoint:
        raise TransparencyError("current_candidate_checkpoint_drift")
    verification = verify_checkpoint(checkpoint, log)
    if verification["public_release_count"] != 0:
        raise TransparencyError("unexpected_public_release_checkpoint")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "no_public_release_checkpoint",
        "exit_code": EXIT_NO_PUBLIC_CHECKPOINT,
        "controls_ready": True,
        "candidate_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "log_length": checkpoint["log_length"],
        "merkle_root": checkpoint["merkle_root"],
        "real_blockers": list(REAL_BLOCKERS),
        "real_blocker_count": 3,
        "formal_validation_complete": False,
        "identity_authentication": IDENTITY_AUTHENTICATION_BOUNDARY,
        "execution": EXECUTION,
    }


def synthetic_record(
    *,
    sequence: int,
    previous: str,
    release_identity: str,
    digest_seed: str,
    code_commit: str,
    supersedes: str | None = None,
) -> dict[str, Any]:
    digest = lambda role: stable_hash({"seed": digest_seed, "role": role})
    record = {
        "artifacts": {
            "clearance_receipt": {
                "evidence_sha256": None,
                "formal_validation_complete": False,
                "sha256": None,
                "source_commit": None,
                "status": "not_available",
            },
            "readiness": _artifact(
                sha256=digest("readiness"),
                status="ready_with_declared_blockers",
                formal_validation_complete=False,
            ),
            "release_candidate": {
                **_artifact(
                    sha256=digest("release_candidate"),
                    status="reproducible_release_ready",
                    formal_validation_complete=False,
                ),
                "qualification": "qualified",
            },
            "standalone": _artifact(
                sha256=digest("standalone"),
                status="verified_with_declared_blockers",
                formal_validation_complete=False,
            ),
        },
        "blockers": list(REAL_BLOCKERS),
        "claims": {
            "claim_count": 3,
            "claims_sha256": digest("claims"),
            "identity_sha256": digest("claim_ids"),
            "status_counts": {"blocked": 3},
        },
        "code_commit": code_commit,
        "content_sha256": ZERO_SHA256,
        "evidence_epoch": sequence,
        "formal_validation_complete": False,
        "freshness": {
            "sha256": digest("freshness"),
            "stale_count": 0,
            "status": "fresh_with_declared_blockers",
        },
        "previous_record_sha256": previous,
        "publisher_identity_authentication": IDENTITY_AUTHENTICATION_BOUNDARY,
        "release_identity": release_identity,
        "release_status": "public_with_declared_blockers",
        "revocation": {
            "active_incident_count": 0,
            "event_count": 1 if supersedes else 0,
            "ledger_sha256": digest("revocation"),
            "status": "revocation_controls_ready",
        },
        "sequence": sequence,
        "supersession": {
            "revocation_event_sha256": digest("revocation_event")
            if supersedes
            else None,
            "supersedes_release_identity": supersedes,
        },
    }
    return finalize_record(record)


def simulate_matrix() -> dict[str, Any]:
    first = synthetic_record(
        sequence=0,
        previous=ZERO_SHA256,
        release_identity="synthetic_transparency:release-a",
        digest_seed="a",
        code_commit="1" * 40,
    )
    base = build_log([first])
    second = synthetic_record(
        sequence=1,
        previous=first["content_sha256"],
        release_identity="synthetic_transparency:release-b",
        digest_seed="b",
        code_commit="2" * 40,
    )
    extended = append_record(base, second)
    third = synthetic_record(
        sequence=2,
        previous=second["content_sha256"],
        release_identity="synthetic_transparency:release-c",
        digest_seed="c",
        code_commit="3" * 40,
        supersedes="synthetic_transparency:release-b",
    )
    superseded = append_record(extended, third)
    inclusion = inclusion_proof(superseded, 1)
    consistency = consistency_proof(base, superseded)
    scenarios: list[dict[str, Any]] = [
        {"scenario": "normal_append", "accepted": True},
        {"scenario": "cross_version_release", "accepted": True},
        {"scenario": "legal_revocation_supersession", "accepted": True},
        {
            "scenario": "inclusion_and_consistency_proofs",
            "accepted": True,
        },
    ]

    def rejected(name: str, operation: Any) -> None:
        try:
            operation()
        except (TransparencyError, KeyError, TypeError, ValueError):
            scenarios.append({"scenario": name, "accepted": False})
        else:
            raise TransparencyError(f"synthetic_attack_not_rejected:{name}")

    deleted = copy.deepcopy(superseded)
    deleted["records"].pop(0)
    deleted["record_count"] -= 1
    deleted["merkle_root"] = merkle_root(deleted["records"])
    deleted["log_sha256"] = ZERO_SHA256
    deleted["log_sha256"] = stable_hash(deleted)
    rejected("history_deletion", lambda: verify_checkpoint(build_checkpoint(base), deleted))

    conflict = copy.deepcopy(second)
    conflict["release_identity"] = first["release_identity"]
    conflict = finalize_record(conflict)
    rejected("same_release_identity_conflict", lambda: append_record(base, conflict))

    branch = synthetic_record(
        sequence=1,
        previous=first["content_sha256"],
        release_identity="synthetic_transparency:release-fork",
        digest_seed="fork",
        code_commit="4" * 40,
    )
    forked = append_record(base, branch)
    rejected("log_fork", lambda: consistency_proof(extended, forked))

    hidden = copy.deepcopy(second)
    hidden["blockers"] = []
    hidden = finalize_record(hidden)
    rejected("blocker_hidden", lambda: append_record(base, hidden))

    stale = copy.deepcopy(second)
    stale["freshness"]["stale_count"] = 1
    stale = finalize_record(stale)
    rejected("stale_hidden", lambda: append_record(base, stale))

    incident = copy.deepcopy(second)
    incident["revocation"]["active_incident_count"] = 1
    incident["revocation"]["status"] = "active_incident_blocks_release"
    incident = finalize_record(incident)
    rejected("active_incident_hidden", lambda: append_record(base, incident))

    rollback = synthetic_record(
        sequence=2,
        previous=second["content_sha256"],
        release_identity="synthetic_transparency:release-rollback",
        digest_seed="a",
        code_commit="1" * 40,
    )
    rejected("old_root_or_evidence_rollback", lambda: append_record(extended, rollback))

    forged = copy.deepcopy(second)
    forged["release_status"] = "formal_release"
    forged["formal_validation_complete"] = True
    forged["blockers"] = []
    forged["artifacts"]["clearance_receipt"] = {
        "evidence_sha256": "e" * 64,
        "formal_validation_complete": True,
        "sha256": "f" * 64,
        "source_commit": "9" * 40,
        "status": "cleared",
    }
    for role in ("readiness", "release_candidate", "standalone"):
        forged["artifacts"][role]["formal_validation_complete"] = True
    forged = finalize_record(forged)
    rejected("forged_clearance_receipt", lambda: append_record(base, forged))

    verify_inclusion_proof(inclusion)
    verify_consistency_proof(consistency)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "transparency_controls_ready",
        "exit_code": EXIT_READY,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "accepted_scenario_count": sum(row["accepted"] for row in scenarios),
        "rejected_scenario_count": sum(not row["accepted"] for row in scenarios),
        "synthetic_only": True,
        "synthetic_artifacts_persisted": False,
        "formal_validation_complete": False,
        "identity_authentication": IDENTITY_AUTHENTICATION_BOUNDARY,
        "execution": EXECUTION,
    }
    report["report_sha256"] = stable_hash(report)
    return report
