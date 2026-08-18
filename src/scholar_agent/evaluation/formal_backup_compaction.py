"""Deterministic content-addressed backup compaction controls for Full1000.

The control plane models the worst-case storage bound without compression,
sparse files, or an assumed future deduplication ratio.  Exact content
addressing is an integrity rule: an already stored immutable blob is referenced
by hash instead of being copied into every root.  Historical roots and their
supersession links remain verifiable.

Only synthetic query identities and payloads are used here.  The module never
starts retrieval, reads credentials, writes Snapshots, or computes quality
metrics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.crash_consistency import (
    durable_atomic_write_bytes,
    stable_json_bytes,
)
from scholar_agent.evaluation.formal_multivolume_storage import (
    AGGREGATE_MAX_BYTES,
    AGGREGATE_MAX_FILES,
    PRIMARY_RESERVE_BYTES,
    PRIMARY_RESERVE_INODES,
    QUERY_COUNT,
    SHARD_COUNT,
    SHARD_MAX_BYTES,
    SHARD_MAX_FILES,
)
from scholar_agent.evaluation.formal_shard_streaming_retention import (
    REQUIRED_ARCHIVE_ROLES,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash


PROTOCOL = "formal_backup_compaction_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "af1e2d28ec41870a8f69e9a217bfeb8baec5becd"
FROZEN_PROTOCOL_SHA256 = (
    "4202229308c385e7d32d3d7b027586687f2b86c37d7811d55129ca593d90e597"
)
POLICY_CONTRACT = "formal_backup_compaction_policy_v1"
ROOT_CONTRACT = "formal_backup_compaction_root_v1"
PUBLICATION_CONTRACT = "formal_backup_compaction_publication_v1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
NOT_AVAILABLE = "not_available"
OLD_BACKUP_REQUIRED_BYTES = 2_119_029_489_664
OLD_BACKUP_REQUIRED_INODES = 210_940
ACTIVE_SHARD_WINDOW = 4
SELECTED_GENERATION_UPPER = 1_040
GENERATION_MAX_FILES = 64
INDEX_ENTRY_MAX_BYTES = 512
PARENT_CHAIN_INDEX_ENTRIES = SELECTED_GENERATION_UPPER * GENERATION_MAX_FILES
PARENT_CHAIN_INDEX_BYTES = PARENT_CHAIN_INDEX_ENTRIES * INDEX_ENTRY_MAX_BYTES
PARENT_CHAIN_INDEX_FILES = SELECTED_GENERATION_UPPER + SHARD_COUNT + 1
COMPACTION_STAGING_SHARDS = 1
RECOVERY_WORKSPACE_SHARDS = ACTIVE_SHARD_WINDOW
FINAL_ARCHIVE_SHARDS = SHARD_COUNT
NEW_BACKUP_REQUIRED_BYTES = (
    FINAL_ARCHIVE_SHARDS * SHARD_MAX_BYTES
    + ACTIVE_SHARD_WINDOW * SHARD_MAX_BYTES
    + COMPACTION_STAGING_SHARDS * SHARD_MAX_BYTES
    + PARENT_CHAIN_INDEX_BYTES
    + AGGREGATE_MAX_BYTES
    + RECOVERY_WORKSPACE_SHARDS * SHARD_MAX_BYTES
    + PRIMARY_RESERVE_BYTES
)
NEW_BACKUP_REQUIRED_INODES = (
    FINAL_ARCHIVE_SHARDS * SHARD_MAX_FILES
    + ACTIVE_SHARD_WINDOW * SHARD_MAX_FILES
    + COMPACTION_STAGING_SHARDS * SHARD_MAX_FILES
    + PARENT_CHAIN_INDEX_FILES
    + AGGREGATE_MAX_FILES
    + RECOVERY_WORKSPACE_SHARDS * SHARD_MAX_FILES
    + PRIMARY_RESERVE_INODES
)
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
ROOT_KINDS = {"baseline", "incremental", "compacted_baseline"}
MAX_JSON_BYTES = 16 * 1024 * 1024
ZERO_SHA256 = "0" * 64


class BackupCompactionError(RuntimeError):
    """A compaction, content-address, history, or recovery invariant failed."""


class BackupCompactionNotReady(BackupCompactionError):
    """The control is valid but no qualified real backup target is present."""


def canonical_json(value: Any) -> bytes:
    return stable_json_bytes(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    durable_atomic_write_bytes(path, canonical_json(dict(value)))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def read_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BackupCompactionError("artifact_unavailable") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise BackupCompactionError("artifact_size_exceeded")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                ValueError("nonfinite_json_number")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupCompactionError("artifact_json_invalid") from exc
    if not isinstance(value, dict):
        raise BackupCompactionError("artifact_root_invalid")
    return value


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BackupCompactionError("bound_artifact_unavailable") from exc


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str):
        raise BackupCompactionError("unsafe_binding_path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not value
        or ".." in path.parts
        or "\\" in value
        or ".env" in path.parts
        or path.parts[0] == "third_party"
        or str(path) != value
    ):
        raise BackupCompactionError("unsafe_binding_path")
    return value


def _digest_without(value: Mapping[str, Any], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return stable_hash(payload)


def capacity_model() -> dict[str, Any]:
    components = {
        "active_shard_window": {
            "coefficient": ACTIVE_SHARD_WINDOW,
            "unit_bytes": SHARD_MAX_BYTES,
            "bytes": ACTIVE_SHARD_WINDOW * SHARD_MAX_BYTES,
            "files": ACTIVE_SHARD_WINDOW * SHARD_MAX_FILES,
            "source": "formal_shard_streaming_retention_v1",
        },
        "aggregate": {
            "coefficient": 1,
            "unit_bytes": AGGREGATE_MAX_BYTES,
            "bytes": AGGREGATE_MAX_BYTES,
            "files": AGGREGATE_MAX_FILES,
            "source": "formal_run_storage_governance_v1",
        },
        "compaction_staging": {
            "coefficient": COMPACTION_STAGING_SHARDS,
            "unit_bytes": SHARD_MAX_BYTES,
            "bytes": COMPACTION_STAGING_SHARDS * SHARD_MAX_BYTES,
            "files": COMPACTION_STAGING_SHARDS * SHARD_MAX_FILES,
            "source": "single_shard_atomic_compaction_staging_v1",
        },
        "final_shard_archives": {
            "coefficient": FINAL_ARCHIVE_SHARDS,
            "unit_bytes": SHARD_MAX_BYTES,
            "bytes": FINAL_ARCHIVE_SHARDS * SHARD_MAX_BYTES,
            "files": FINAL_ARCHIVE_SHARDS * SHARD_MAX_FILES,
            "source": "formal_multivolume_storage_v1",
        },
        "parent_chain_index": {
            "coefficient": PARENT_CHAIN_INDEX_ENTRIES,
            "unit_bytes": INDEX_ENTRY_MAX_BYTES,
            "bytes": PARENT_CHAIN_INDEX_BYTES,
            "files": PARENT_CHAIN_INDEX_FILES,
            "source": "1040_generations_x_64_files_x_512_byte_entry",
        },
        "recovery_workspace": {
            "coefficient": RECOVERY_WORKSPACE_SHARDS,
            "unit_bytes": SHARD_MAX_BYTES,
            "bytes": RECOVERY_WORKSPACE_SHARDS * SHARD_MAX_BYTES,
            "files": RECOVERY_WORKSPACE_SHARDS * SHARD_MAX_FILES,
            "source": "active_shard_window_4_restore_workspace",
        },
        "safety_reserve": {
            "coefficient": 1,
            "unit_bytes": PRIMARY_RESERVE_BYTES,
            "bytes": PRIMARY_RESERVE_BYTES,
            "files": PRIMARY_RESERVE_INODES,
            "source": "formal_run_storage_governance_v1",
        },
    }
    bytes_total = sum(int(row["bytes"]) for row in components.values())
    files_total = sum(int(row["files"]) for row in components.values())
    if bytes_total != NEW_BACKUP_REQUIRED_BYTES or files_total != NEW_BACKUP_REQUIRED_INODES:
        raise BackupCompactionError("capacity_component_sum_mismatch")
    reduction_bytes = OLD_BACKUP_REQUIRED_BYTES - bytes_total
    return {
        "old_backup_required_bytes": OLD_BACKUP_REQUIRED_BYTES,
        "old_backup_required_inodes": OLD_BACKUP_REQUIRED_INODES,
        "new_backup_required_bytes": bytes_total,
        "new_backup_required_inodes": files_total,
        "reduction_bytes": reduction_bytes,
        "reduction_basis_points": (
            reduction_bytes * 10_000 // OLD_BACKUP_REQUIRED_BYTES
        ),
        "components": components,
        "compression_credit_bytes": 0,
        "sparse_file_credit_bytes": 0,
        "future_deduplication_credit_bytes": 0,
        "future_cleanup_credit_bytes": 0,
        "unknown_capacity_inputs": [],
    }


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_object(path)
    if set(value) != {
        "bindings",
        "capacity",
        "execution",
        "formal_validation_complete",
        "policy",
        "population",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
    }:
        raise BackupCompactionError("protocol_schema_invalid")
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["formal_validation_complete"] is not False
        or value["execution"] != EXECUTION_ZERO
        or value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256
        or _digest_without(value, "protocol_sha256") != FROZEN_PROTOCOL_SHA256
    ):
        raise BackupCompactionError("protocol_identity_invalid")
    if value["population"] != {
        "active_shard_window": ACTIVE_SHARD_WINDOW,
        "http_attempt_upper": 19_280,
        "query_count": QUERY_COUNT,
        "selected_generation_upper": SELECTED_GENERATION_UPPER,
        "shard_count": SHARD_COUNT,
    }:
        raise BackupCompactionError("protocol_population_invalid")
    if value["policy"] != {
        "ability_default_enabled": False,
        "compaction_staging_shards": COMPACTION_STAGING_SHARDS,
        "content_address_algorithm": "sha256",
        "exact_blob_reuse_is_integrity_not_capacity_credit": True,
        "final_archive_copies_per_sealed_shard": 1,
        "history_root_rewrite_allowed": False,
        "recovery_workspace_shards": RECOVERY_WORKSPACE_SHARDS,
        "unique_blob_deletion_allowed": False,
    }:
        raise BackupCompactionError("protocol_policy_invalid")
    if value["capacity"] != capacity_model():
        raise BackupCompactionError("protocol_capacity_invalid")
    expected_bindings = {
        "backup_target_attestation",
        "crash_consistency",
        "disaster_recovery",
        "execution_plan",
        "host_attestation",
        "launch_control",
        "multivolume_storage",
        "portable_execution_site",
        "provider_ingest",
        "resource_accounting",
        "shard_streaming_retention",
        "storage_governance",
    }
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        raise BackupCompactionError("protocol_binding_inventory_invalid")
    for row in bindings.values():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise BackupCompactionError("protocol_binding_invalid")
        target = repository_root / _safe_relative(row["path"])
        if not target.is_file() or sha256_file(target) != row["sha256"]:
            raise BackupCompactionError("protocol_binding_hash_drift")
    return value


def build_policy(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "contract": POLICY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "default_enabled": False,
        "capacity": capacity_model(),
        "root_rules": {
            "compacted_baseline_keeps_parent_root": True,
            "content_bytes_rewritten_after_publish": False,
            "incremental_contains_new_committed_generations_only": True,
            "old_roots_retained": True,
            "single_complete_archive_per_sealed_shard": True,
            "unique_blob_deletion_allowed": False,
        },
        "query_or_request_semantics_changed": False,
        "formal_validation_complete": False,
    }
    value["policy_sha256"] = stable_hash(value)
    return value


def calculate_capacity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol["capacity"] != capacity_model():
        raise BackupCompactionError("capacity_protocol_drift")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_compaction_ready",
        "exit_code": EXIT_READY,
        "capacity": capacity_model(),
        "capacity_model_uses_compression": False,
        "capacity_model_uses_sparse_files": False,
        "capacity_model_uses_future_deduplication": False,
        "capacity_model_uses_future_cleanup": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def _query_identity(index: int) -> str:
    return stable_hash({"fixture_query_index": index})


def _result_identity(index: int, *, attempt: int) -> str:
    return stable_hash(
        {
            "fixture_query_index": index,
            "selected_attempt": attempt,
            "synthetic_result": "offline",
        }
    )


def build_shard_state(
    shard_index: int,
    *,
    cursor: int,
    attempt: int = 0,
    generation: int = 1,
) -> dict[str, Any]:
    if shard_index not in range(SHARD_COUNT):
        raise BackupCompactionError("shard_index_invalid")
    if cursor not in range(51):
        raise BackupCompactionError("shard_cursor_invalid")
    start = shard_index * 50
    query_results = [
        {
            "query_identity": _query_identity(start + offset),
            "result_sha256": _result_identity(start + offset, attempt=attempt),
        }
        for offset in range(cursor)
    ]
    files = {
        role: canonical_json(
            {
                "attempt": attempt,
                "cursor": cursor,
                "generation": generation,
                "role": role,
                "shard_index": shard_index,
                "synthetic_only": True,
            }
        )
        for role in REQUIRED_ARCHIVE_ROLES
    }
    value = {
        "shard_index": shard_index,
        "attempt_identity": stable_hash(
            {"attempt": attempt, "shard_index": shard_index}
        ),
        "generation": generation,
        "query_cursor": cursor,
        "query_results": query_results,
        "files": files,
    }
    value["state_sha256"] = stable_hash(
        {
            **value,
            "files": {
                role: hashlib.sha256(content).hexdigest()
                for role, content in sorted(files.items())
            },
        }
    )
    return value


class ContentAddressedBackup:
    """In-memory deterministic content-addressed store used by the gate."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.roots: dict[str, dict[str, Any]] = {}
        self.publications: list[dict[str, Any]] = []
        self.active_root_sha256: str | None = None
        self._staged_root: dict[str, Any] | None = None

    def _put_blob(self, content: bytes) -> dict[str, Any]:
        digest = hashlib.sha256(content).hexdigest()
        prior = self.blobs.get(digest)
        if prior is not None and prior != content:
            raise BackupCompactionError("content_address_collision")
        self.blobs[digest] = content
        return {"sha256": digest, "size": len(content)}

    def _state_record(self, state: Mapping[str, Any]) -> dict[str, Any]:
        raw_files = state["files"]
        if isinstance(raw_files, Mapping):
            files = [
                {
                    "role": role,
                    **self._put_blob(raw_files[role]),
                }
                for role in REQUIRED_ARCHIVE_ROLES
            ]
        elif isinstance(raw_files, list):
            files = copy.deepcopy(raw_files)
            if [row.get("role") for row in files if isinstance(row, dict)] != list(
                REQUIRED_ARCHIVE_ROLES
            ):
                raise BackupCompactionError("root_role_inventory_invalid")
            for item in files:
                blob = self.blobs.get(str(item.get("sha256")))
                if (
                    blob is None
                    or item.get("size") != len(blob)
                    or item.get("sha256") != hashlib.sha256(blob).hexdigest()
                ):
                    raise BackupCompactionError("root_blob_integrity_invalid")
        else:
            raise BackupCompactionError("root_file_inventory_invalid")
        return {
            "attempt_identity": state["attempt_identity"],
            "files": files,
            "generation": state["generation"],
            "query_cursor": state["query_cursor"],
            "query_results": copy.deepcopy(state["query_results"]),
            "shard_index": state["shard_index"],
            "state_sha256": state["state_sha256"],
        }

    @staticmethod
    def _index(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "attempt_identity": row["attempt_identity"],
                "generation": row["generation"],
                "shard_index": row["shard_index"],
                "state_sha256": row["state_sha256"],
            }
            for row in states
        ]

    def build_root(
        self,
        *,
        kind: str,
        states: Sequence[Mapping[str, Any]],
        parent_root_sha256: str | None,
        supersedes_root_sha256: str | None = None,
    ) -> dict[str, Any]:
        if kind not in ROOT_KINDS:
            raise BackupCompactionError("root_kind_invalid")
        if kind == "baseline" and parent_root_sha256 is not None:
            raise BackupCompactionError("genesis_parent_forbidden")
        if kind != "baseline" and parent_root_sha256 is None:
            raise BackupCompactionError("non_genesis_parent_required")
        if kind == "compacted_baseline" and (
            supersedes_root_sha256 != parent_root_sha256
        ):
            raise BackupCompactionError("compaction_supersession_invalid")
        if kind != "compacted_baseline" and supersedes_root_sha256 is not None:
            raise BackupCompactionError("unexpected_supersession")
        records = [
            self._state_record(row)
            for row in sorted(states, key=lambda item: int(item["shard_index"]))
        ]
        shards = [int(row["shard_index"]) for row in records]
        if len(shards) != len(set(shards)):
            raise BackupCompactionError("root_duplicate_shard")
        if kind in {"baseline", "compacted_baseline"} and shards != list(
            range(SHARD_COUNT)
        ):
            raise BackupCompactionError("baseline_shard_coverage_invalid")
        if kind == "incremental" and not records:
            raise BackupCompactionError("incremental_empty")
        sequence = len(self.publications)
        value: dict[str, Any] = {
            "contract": ROOT_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "kind": kind,
            "parent_root_sha256": parent_root_sha256,
            "supersedes_root_sha256": supersedes_root_sha256,
            "states": records,
            "index": self._index(records),
            "history_genesis_rebuilt": False,
            "compression_credit_bytes": 0,
            "future_deduplication_credit_bytes": 0,
            "formal_validation_complete": False,
        }
        value["root_sha256"] = stable_hash(value)
        return value

    def _verify_root_payload(self, root: Mapping[str, Any]) -> None:
        if set(root) != {
            "compression_credit_bytes",
            "contract",
            "formal_validation_complete",
            "future_deduplication_credit_bytes",
            "history_genesis_rebuilt",
            "index",
            "kind",
            "parent_root_sha256",
            "root_sha256",
            "schema_version",
            "sequence",
            "states",
            "supersedes_root_sha256",
        }:
            raise BackupCompactionError("root_schema_invalid")
        if (
            root["contract"] != ROOT_CONTRACT
            or root["schema_version"] != SCHEMA_VERSION
            or root["kind"] not in ROOT_KINDS
            or root["history_genesis_rebuilt"] is not False
            or root["compression_credit_bytes"] != 0
            or root["future_deduplication_credit_bytes"] != 0
            or root["formal_validation_complete"] is not False
            or _digest_without(root, "root_sha256") != root["root_sha256"]
            or not isinstance(root["states"], list)
            or root["index"] != self._index(root["states"])
        ):
            raise BackupCompactionError("root_integrity_invalid")
        states = root["states"]
        shards = [row.get("shard_index") for row in states if isinstance(row, dict)]
        if len(shards) != len(states) or shards != sorted(shards):
            raise BackupCompactionError("root_state_order_invalid")
        if len(shards) != len(set(shards)):
            raise BackupCompactionError("root_duplicate_shard")
        if root["kind"] in {"baseline", "compacted_baseline"} and shards != list(
            range(SHARD_COUNT)
        ):
            raise BackupCompactionError("baseline_shard_coverage_invalid")
        if root["kind"] == "incremental" and not states:
            raise BackupCompactionError("incremental_empty")
        if root["kind"] == "baseline":
            if (
                root["parent_root_sha256"] is not None
                or root["supersedes_root_sha256"] is not None
                or root["sequence"] != 0
            ):
                raise BackupCompactionError("genesis_identity_invalid")
        elif root["parent_root_sha256"] is None:
            raise BackupCompactionError("root_parent_missing")
        if root["kind"] == "compacted_baseline":
            if root["supersedes_root_sha256"] != root["parent_root_sha256"]:
                raise BackupCompactionError("compaction_supersession_invalid")
        elif root["supersedes_root_sha256"] is not None:
            raise BackupCompactionError("unexpected_supersession")
        for state in states:
            if set(state) != {
                "attempt_identity",
                "files",
                "generation",
                "query_cursor",
                "query_results",
                "shard_index",
                "state_sha256",
            }:
                raise BackupCompactionError("root_state_schema_invalid")
            files = state["files"]
            roles = [row.get("role") for row in files if isinstance(row, dict)]
            if roles != list(REQUIRED_ARCHIVE_ROLES):
                raise BackupCompactionError("root_role_inventory_invalid")
            if state["query_cursor"] != len(state["query_results"]):
                raise BackupCompactionError("root_query_cursor_invalid")
            state_digest = stable_hash(
                {
                    "attempt_identity": state["attempt_identity"],
                    "files": {
                        row["role"]: row["sha256"] for row in state["files"]
                    },
                    "generation": state["generation"],
                    "query_cursor": state["query_cursor"],
                    "query_results": state["query_results"],
                    "shard_index": state["shard_index"],
                }
            )
            if state_digest != state["state_sha256"]:
                raise BackupCompactionError("root_state_digest_invalid")
            for item in files:
                blob = self.blobs.get(str(item.get("sha256")))
                if blob is None:
                    raise BackupCompactionError("root_blob_missing")
                if (
                    item.get("size") != len(blob)
                    or item.get("sha256") != hashlib.sha256(blob).hexdigest()
                ):
                    raise BackupCompactionError("root_blob_integrity_invalid")

    def verify_root(self, root_sha256: str) -> None:
        root = self.roots.get(root_sha256)
        if root is None:
            raise BackupCompactionError("root_missing")
        self._verify_root_payload(root)
        seen: set[str] = set()
        current: str | None = root_sha256
        while current is not None:
            if current in seen:
                raise BackupCompactionError("root_parent_cycle")
            seen.add(current)
            item = self.roots.get(current)
            if item is None:
                raise BackupCompactionError("root_parent_missing")
            self._verify_root_payload(item)
            current = item["parent_root_sha256"]

    def publish(
        self,
        root: Mapping[str, Any],
        *,
        fault_before_publish: bool = False,
    ) -> str:
        candidate = copy.deepcopy(dict(root))
        self._verify_root_payload(candidate)
        parent = candidate["parent_root_sha256"]
        if parent is not None:
            if parent != self.active_root_sha256:
                raise BackupCompactionError("stale_parent_publication")
            self.verify_root(parent)
        if candidate["sequence"] != len(self.publications):
            raise BackupCompactionError("publication_sequence_invalid")
        self._staged_root = candidate
        if fault_before_publish:
            self._staged_root = None
            raise BackupCompactionError("compaction_interrupted_previous_root_active")
        digest = str(candidate["root_sha256"])
        if digest in self.roots:
            raise BackupCompactionError("duplicate_root_publication")
        self.roots[digest] = candidate
        previous = (
            self.publications[-1]["publication_sha256"]
            if self.publications
            else ZERO_SHA256
        )
        event: dict[str, Any] = {
            "contract": PUBLICATION_CONTRACT,
            "sequence": len(self.publications),
            "root_sha256": digest,
            "parent_root_sha256": parent,
            "previous_publication_sha256": previous,
        }
        event["publication_sha256"] = stable_hash(event)
        self.publications.append(event)
        self.active_root_sha256 = digest
        self._staged_root = None
        return digest

    def verify_publications(self) -> None:
        previous = ZERO_SHA256
        for sequence, event in enumerate(self.publications):
            if (
                set(event)
                != {
                    "contract",
                    "parent_root_sha256",
                    "previous_publication_sha256",
                    "publication_sha256",
                    "root_sha256",
                    "sequence",
                }
                or event["contract"] != PUBLICATION_CONTRACT
                or event["sequence"] != sequence
                or event["previous_publication_sha256"] != previous
                or _digest_without(event, "publication_sha256")
                != event["publication_sha256"]
                or event["root_sha256"] not in self.roots
            ):
                raise BackupCompactionError("publication_chain_invalid")
            if sequence == 0:
                if event["parent_root_sha256"] is not None:
                    raise BackupCompactionError("publication_genesis_invalid")
            elif event["parent_root_sha256"] != self.publications[sequence - 1][
                "root_sha256"
            ]:
                raise BackupCompactionError("publication_parent_mismatch")
            previous = str(event["publication_sha256"])

    def resolve_states(self, root_sha256: str) -> dict[int, dict[str, Any]]:
        self.verify_root(root_sha256)
        chain: list[Mapping[str, Any]] = []
        current: str | None = root_sha256
        while current is not None:
            root = self.roots[current]
            chain.append(root)
            current = root["parent_root_sha256"]
        states: dict[int, dict[str, Any]] = {}
        for root in reversed(chain):
            if root["kind"] in {"baseline", "compacted_baseline"}:
                states = {}
            for state in root["states"]:
                states[int(state["shard_index"])] = copy.deepcopy(state)
        if set(states) != set(range(SHARD_COUNT)):
            raise BackupCompactionError("recovery_shard_coverage_invalid")
        return states

    def fallback_root(self) -> str:
        for event in reversed(self.publications):
            root_sha256 = str(event["root_sha256"])
            try:
                self.verify_root(root_sha256)
            except BackupCompactionError:
                continue
            return root_sha256
        raise BackupCompactionError("no_valid_restore_point")

    def delete_blob(self, blob_sha256: str) -> None:
        for root in self.roots.values():
            for state in root["states"]:
                if any(
                    row["sha256"] == blob_sha256 for row in state["files"]
                ):
                    raise BackupCompactionError("unique_blob_delete_forbidden")
        self.blobs.pop(blob_sha256, None)


def _aggregate(states: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if set(states) != set(range(SHARD_COUNT)):
        raise BackupCompactionError("aggregate_shard_coverage_invalid")
    rows = [
        result
        for shard in range(SHARD_COUNT)
        for result in states[shard]["query_results"]
    ]
    if len(rows) != QUERY_COUNT:
        raise BackupCompactionError("aggregate_query_coverage_invalid")
    identities = [row["query_identity"] for row in rows]
    if len(identities) != len(set(identities)):
        raise BackupCompactionError("aggregate_duplicate_query")
    return {
        "query_count": len(rows),
        "query_results_sha256": stable_hash(rows),
        "top20_delivery_sha256": stable_hash(
            [{"query_identity": row["query_identity"], "rank_limit": 20} for row in rows]
        ),
    }


def _expect_error(call: Any) -> str:
    try:
        call()
    except BackupCompactionError as exc:
        return str(exc)
    return "no_error"


def simulate_compaction(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol["capacity"] != capacity_model():
        raise BackupCompactionError("capacity_protocol_drift")
    store = ContentAddressedBackup()
    states = {
        shard: build_shard_state(shard, cursor=0, generation=0)
        for shard in range(SHARD_COUNT)
    }
    genesis = store.build_root(
        kind="baseline",
        states=list(states.values()),
        parent_root_sha256=None,
    )
    genesis_sha = store.publish(genesis)
    adapter_calls: dict[str, int] = {}
    compacted_roots: list[str] = []
    incremental_roots: list[str] = []
    recovered_after_400 = False
    for batch_start in range(0, SHARD_COUNT, ACTIVE_SHARD_WINDOW):
        changed: list[dict[str, Any]] = []
        for shard in range(batch_start, batch_start + ACTIVE_SHARD_WINDOW):
            for offset in range(50):
                identity = _query_identity(shard * 50 + offset)
                adapter_calls[identity] = adapter_calls.get(identity, 0) + 1
                if adapter_calls[identity] != 1:
                    raise BackupCompactionError("committed_query_repeated")
            states[shard] = build_shard_state(
                shard,
                cursor=50,
                generation=1,
            )
            changed.append(states[shard])
        incremental = store.build_root(
            kind="incremental",
            states=changed,
            parent_root_sha256=store.active_root_sha256,
        )
        incremental_roots.append(store.publish(incremental))
        if batch_start == ACTIVE_SHARD_WINDOW:
            compacted = store.build_root(
                kind="compacted_baseline",
                states=list(states.values()),
                parent_root_sha256=store.active_root_sha256,
                supersedes_root_sha256=store.active_root_sha256,
            )
            compacted_roots.append(store.publish(compacted))
            recovered = store.resolve_states(store.active_root_sha256 or "")
            recovered_after_400 = (
                sum(row["query_cursor"] for row in recovered.values()) == 400
            )
    pre_final_compaction_root = str(store.active_root_sha256)
    final_compacted = store.build_root(
        kind="compacted_baseline",
        states=list(states.values()),
        parent_root_sha256=store.active_root_sha256,
        supersedes_root_sha256=store.active_root_sha256,
    )
    compacted_roots.append(store.publish(final_compacted))
    final_root = str(store.active_root_sha256)
    store.verify_publications()
    for root_sha256 in list(store.roots):
        store.verify_root(root_sha256)

    interrupted_candidate = store.build_root(
        kind="compacted_baseline",
        states=list(states.values()),
        parent_root_sha256=store.active_root_sha256,
        supersedes_root_sha256=store.active_root_sha256,
    )
    active_before_fault = store.active_root_sha256
    interrupted = _expect_error(
        lambda: store.publish(interrupted_candidate, fault_before_publish=True)
    )
    interruption_preserved_previous = (
        store.active_root_sha256 == active_before_fault
    )

    tampered = copy.deepcopy(store.roots[final_root])
    tampered["index"][0]["generation"] += 1
    tampered["root_sha256"] = _digest_without(tampered, "root_sha256")
    tamper_store = copy.deepcopy(store)
    tamper_store.roots[tampered["root_sha256"]] = tampered
    index_tamper = _expect_error(
        lambda: tamper_store.verify_root(str(tampered["root_sha256"]))
    )

    parent_store = copy.deepcopy(store)
    parent = str(parent_store.roots[final_root]["parent_root_sha256"])
    parent_store.roots.pop(parent)
    parent_missing = _expect_error(lambda: parent_store.verify_root(final_root))

    referenced_blob = str(store.roots[final_root]["states"][0]["files"][0]["sha256"])
    unique_delete = _expect_error(lambda: store.delete_blob(referenced_blob))

    damaged_store = copy.deepcopy(store)
    damaged_store.roots[final_root]["index"][0]["generation"] += 1
    fallback = damaged_store.fallback_root()
    fallback_states = damaged_store.resolve_states(fallback)

    replacement = build_shard_state(5, cursor=50, attempt=1, generation=2)
    states[5] = replacement
    replacement_root = store.build_root(
        kind="incremental",
        states=[replacement],
        parent_root_sha256=store.active_root_sha256,
    )
    store.publish(replacement_root)
    replacement_compacted = store.build_root(
        kind="compacted_baseline",
        states=list(states.values()),
        parent_root_sha256=store.active_root_sha256,
        supersedes_root_sha256=store.active_root_sha256,
    )
    store.publish(replacement_compacted)
    final_states = store.resolve_states(str(store.active_root_sha256))
    aggregate = _aggregate(final_states)
    uninterrupted = _aggregate(
        {
            shard: (
                build_shard_state(shard, cursor=50, attempt=1, generation=2)
                if shard == 5
                else build_shard_state(shard, cursor=50, generation=1)
            )
            for shard in range(SHARD_COUNT)
        }
    )

    scenarios = {
        "aggregate_equivalence": (
            aggregate == uninterrupted
            and aggregate["query_count"] == QUERY_COUNT
        ),
        "baseline_damage_fallback": (
            fallback == pre_final_compaction_root
            and sum(row["query_cursor"] for row in fallback_states.values())
            == QUERY_COUNT
        ),
        "compaction_interruption": (
            interrupted == "compaction_interrupted_previous_root_active"
            and interruption_preserved_previous
        ),
        "index_tamper": index_tamper == "root_integrity_invalid",
        "multiple_compactions": len(compacted_roots) == 2,
        "normal_incremental": len(incremental_roots) == 5,
        "old_root_verification": genesis_sha in store.roots,
        "parent_chain_missing": parent_missing == "root_parent_missing",
        "restore_after_400": recovered_after_400,
        "single_shard_replacement": (
            final_states[5]["attempt_identity"] == replacement["attempt_identity"]
            and final_states[5]["generation"] == 2
        ),
        "unique_blob_delete": unique_delete == "unique_blob_delete_forbidden",
        "window_4_continuous_archive": (
            len(adapter_calls) == QUERY_COUNT
            and max(adapter_calls.values(), default=0) == 1
        ),
    }
    failed = sorted(key for key, passed in scenarios.items() if not passed)
    if failed:
        raise BackupCompactionError("synthetic_scenario_failed:" + failed[0])
    model = capacity_model()
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_compaction_ready",
        "exit_code": EXIT_READY,
        "scenario_count": len(scenarios),
        "scenarios": {
            key: {"status": "passed"} for key in sorted(scenarios)
        },
        "query_count": QUERY_COUNT,
        "shard_count": SHARD_COUNT,
        "active_shard_window": ACTIVE_SHARD_WINDOW,
        "incremental_root_count": len(incremental_roots) + 1,
        "compacted_baseline_count": len(compacted_roots) + 1,
        "retained_root_count": len(store.roots),
        "adapter_call_count": sum(adapter_calls.values()),
        "duplicate_request_count": sum(
            count - 1 for count in adapter_calls.values() if count > 1
        ),
        "selected_resource_ledger_query_count": len(adapter_calls),
        "resource_ledger_conserved": (
            len(adapter_calls) == QUERY_COUNT
            and sum(adapter_calls.values()) == QUERY_COUNT
        ),
        "aggregate_matches_uninterrupted": aggregate == uninterrupted,
        "old_backup_required_bytes": model["old_backup_required_bytes"],
        "new_backup_required_bytes": model["new_backup_required_bytes"],
        "history_root_rewritten": False,
        "synthetic_only": True,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def verify_recovery(protocol: Mapping[str, Any]) -> dict[str, Any]:
    simulation = simulate_compaction(protocol)
    if (
        simulation["duplicate_request_count"] != 0
        or simulation["resource_ledger_conserved"] is not True
        or simulation["aggregate_matches_uninterrupted"] is not True
        or simulation["history_root_rewritten"] is not False
    ):
        raise BackupCompactionError("recovery_verification_failed")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "backup_compaction_ready",
        "exit_code": EXIT_READY,
        "query_count": QUERY_COUNT,
        "duplicate_request_count": 0,
        "resource_ledger_conserved": True,
        "aggregate_matches_uninterrupted": True,
        "all_retained_restore_points_recoverable": True,
        "old_roots_verifiable": True,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def audit_readiness(protocol: Mapping[str, Any]) -> dict[str, Any]:
    model = capacity_model()
    if protocol["capacity"] != model:
        raise BackupCompactionError("capacity_protocol_drift")
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "not_ready_missing_qualified_backup_target",
        "exit_code": EXIT_NOT_READY,
        "old_backup_required_bytes": OLD_BACKUP_REQUIRED_BYTES,
        "old_backup_required_inodes": OLD_BACKUP_REQUIRED_INODES,
        "new_backup_required_bytes": NEW_BACKUP_REQUIRED_BYTES,
        "new_backup_required_inodes": NEW_BACKUP_REQUIRED_INODES,
        "backup_available_bytes": NOT_AVAILABLE,
        "backup_available_inodes": NOT_AVAILABLE,
        "backup_quota_bytes": NOT_AVAILABLE,
        "backup_failure_domain_independent": NOT_AVAILABLE,
        "missing_backup_fields": [
            "backup.available_bytes",
            "backup.available_inodes",
            "backup.failure_domain_independent",
            "backup.filesystem_quota_bytes",
        ],
        "controls_default_enabled": False,
        "formal_run_started": False,
        "full1000_blocker_cleared": False,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
