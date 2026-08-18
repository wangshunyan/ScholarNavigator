"""Deterministic, offline reproduction capsules for committed Replay runs.

Capsules contain data only.  The host checkout validates and replays them via
the existing ``run_manifest_v1``, ``BenchmarkRunCommitStore`` and
``SearchServiceFixtureBackend`` paths; no archived code is ever executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.crash_consistency import (
    STORE_DIRECTORY,
    BenchmarkRunCommitStore,
    CrashConsistencyError,
    stable_json_bytes,
)
from scholar_agent.evaluation.current_rules_regression import compare_profiles
from scholar_agent.evaluation.execution_determinism import (
    load_protocol as load_execution_protocol,
    load_query_fixtures,
    replay_canonical_fixture,
)
from scholar_agent.evaluation.run_provenance import (
    GitProvenance,
    RunManifestV1,
    build_run_manifest,
    validate_run_manifest,
    write_json as write_manifest_json,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash


CONTRACT_VERSION = "reproduction_capsule_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "reproduction_capsule_gate"
CAPSULE_MANIFEST_NAME = "capsule_manifest.json"
PAYLOAD_PREFIX = "payload"
EXIT_PASSED = 0
EXIT_INTEGRITY_OR_REPLAY_MISMATCH = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 48 * 1024 * 1024
MAX_MEMBERS = 512
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
FIXED_FILE_MODE = 0o644

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SECRET_PATH_PARTS = frozenset({".env", "third_party", "__pycache__", ".pytest_cache"})
_SECRET_TEXT_RE = re.compile(
    r"(?i)(authorization|api[_-]?key|token)(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_ABSOLUTE_PATH_RE = re.compile(r"(?:/[A-Za-z0-9._-]+){2,}")


class ReproductionCapsuleError(RuntimeError):
    """Base error with a stable, non-sensitive audit location."""

    def __init__(
        self,
        reason: str,
        *,
        stage: str = "capsule",
        invariant: str = "contract",
        location: str = "$",
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stage = stage
        self.invariant = invariant
        self.location = location


class CapsuleIntegrityError(ReproductionCapsuleError):
    """Archive or replay semantics do not match the sealed contract."""


class CapsuleNotEligible(ReproductionCapsuleError):
    """A source run lacks the self-contained new-format evidence."""


class CapsuleFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    roles: list[str] = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path_and_roles(self) -> "CapsuleFile":
        _validate_capsule_path(self.path)
        if not self.path.startswith(PAYLOAD_PREFIX + "/"):
            raise ValueError("capsule file must live below payload/")
        if self.roles != sorted(set(self.roles)):
            raise ValueError("capsule file roles must be sorted and unique")
        return self


class CapsuleManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    protocol: Literal["reproduction_capsule_v1"] = CONTRACT_VERSION
    score_scope: Literal[
        "portable_replay_only_not_quality_or_official_score"
    ] = "portable_replay_only_not_quality_or_official_score"
    generated_by_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    run_manifest: dict[str, Any]
    query_set: dict[str, Any]
    replay: dict[str, Any]
    prompt: dict[str, Any]
    configuration: dict[str, Any]
    evaluator: dict[str, Any]
    determinism: dict[str, Any]
    comparison: dict[str, Any] | None = None
    shard: dict[str, Any] | None = None
    provider_ingest: dict[str, Any] | None = None
    generation_chain: dict[str, Any]
    entrypoint: dict[str, Any]
    files: list[CapsuleFile]
    limits: dict[str, int]
    capsule_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_closed_manifest(self) -> "CapsuleManifestV1":
        _require_exact_keys(
            self.run_manifest,
            {"path", "sha256", "run_id", "schema_version"},
            "run_manifest",
        )
        _require_exact_keys(
            self.query_set,
            {
                "count",
                "stable_identity_sha256",
                "order_sha256",
                "input_sha256",
                "committed_query_order_sha256",
            },
            "query_set",
        )
        _require_exact_keys(
            self.replay,
            {
                "protocol_path",
                "protocol_sha256",
                "fixture_sha256",
                "expected_records_sha256",
                "expected_query_count",
                "canonicalization_policy",
                "network_request_count",
                "llm_request_count",
                "snapshot_write_count",
            },
            "replay",
        )
        _require_exact_keys(
            self.generation_chain,
            {
                "store_contract",
                "latest_generation",
                "generation_count",
                "latest_generation_path",
                "status",
                "record_count",
                "event_count",
                "chain_sha256",
            },
            "generation_chain",
        )
        if self.comparison is not None:
            _require_exact_keys(
                self.comparison,
                {
                    "contract",
                    "plan",
                    "plan_sha256",
                    "role",
                    "common_execution_contract_sha256",
                },
                "comparison",
            )
        if self.shard is not None:
            _require_exact_keys(
                self.shard,
                {
                    "contract",
                    "plan",
                    "plan_sha256",
                    "shard_index",
                    "shard_count",
                    "expected_query_identities_sha256",
                    "common_execution_contract_sha256",
                    "attempt_id",
                    "supersedes_attempt_id",
                },
                "shard",
            )
        if self.provider_ingest is not None:
            _require_exact_keys(
                self.provider_ingest,
                {
                    "contract",
                    "bundle",
                    "raw_archive",
                    "resource_ledger_sha256",
                    "manifest_identity",
                    "authority",
                },
                "provider_ingest",
            )
        _require_exact_keys(
            self.entrypoint,
            {"kind", "command", "execute_archived_code"},
            "entrypoint",
        )
        for location, value in (
            ("run_manifest.sha256", self.run_manifest.get("sha256")),
            ("query_set.stable_identity_sha256", self.query_set.get("stable_identity_sha256")),
            ("query_set.order_sha256", self.query_set.get("order_sha256")),
            ("query_set.input_sha256", self.query_set.get("input_sha256")),
            (
                "query_set.committed_query_order_sha256",
                self.query_set.get("committed_query_order_sha256"),
            ),
            ("replay.protocol_sha256", self.replay.get("protocol_sha256")),
            ("replay.fixture_sha256", self.replay.get("fixture_sha256")),
            (
                "replay.expected_records_sha256",
                self.replay.get("expected_records_sha256"),
            ),
            ("generation_chain.chain_sha256", self.generation_chain.get("chain_sha256")),
        ):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"invalid digest:{location}")
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("capsule file inventory must be sorted and unique")
        if self.entrypoint.get("kind") != "host_search_service_replay":
            raise ValueError("capsule entrypoint kind is not supported")
        if self.entrypoint.get("execute_archived_code") is not False:
            raise ValueError("capsule must forbid archived code execution")
        if self.entrypoint.get("command") != [
            "python",
            "scripts/check_reproduction_capsule.py",
            "replay",
            "--capsule",
            "<capsule.tar>",
        ]:
            raise ValueError("capsule host entrypoint drifted")
        if self.run_manifest.get("path") != "payload/run_manifest.json":
            raise ValueError("run manifest path drifted")
        if self.replay.get("protocol_path") != "payload/replay_protocol.json":
            raise ValueError("replay protocol path drifted")
        if any(
            self.replay.get(field) != 0
            for field in (
                "network_request_count",
                "llm_request_count",
                "snapshot_write_count",
            )
        ):
            raise ValueError("capsule replay side-effect contract drifted")
        if (
            self.generation_chain.get("store_contract") != "crash_consistency_v1"
            or self.generation_chain.get("status") != "completed"
        ):
            raise ValueError("committed generation contract drifted")
        for field in ("count",):
            if not isinstance(self.query_set.get(field), int) or self.query_set[field] < 1:
                raise ValueError(f"invalid query set integer:{field}")
        if self.replay.get("expected_query_count") != self.query_set.get("count"):
            raise ValueError("replay and query counts differ")
        expected_limits = {
            "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "max_file_count": MAX_MEMBERS,
            "max_member_bytes": MAX_MEMBER_BYTES,
            "max_total_unpacked_bytes": MAX_TOTAL_BYTES,
        }
        if self.limits != expected_limits:
            raise ValueError("capsule resource limits drifted")
        expected = _capsule_summary_payload(self)
        if stable_hash(expected) != self.capsule_summary_sha256:
            raise ValueError("capsule summary hash mismatch")
        return self


def load_gate_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReproductionCapsuleError("protocol_unreadable") from exc
    if not isinstance(value, dict):
        raise ReproductionCapsuleError("protocol_root_invalid")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("contract") != CONTRACT_VERSION
        or value.get("score_scope")
        != "portable_replay_only_not_quality_or_official_score"
    ):
        raise ReproductionCapsuleError("protocol_version_incompatible")
    limits = value.get("limits")
    if limits != {
        "max_archive_bytes": MAX_ARCHIVE_BYTES,
        "max_file_count": MAX_MEMBERS,
        "max_member_bytes": MAX_MEMBER_BYTES,
        "max_total_unpacked_bytes": MAX_TOTAL_BYTES,
    }:
        raise ReproductionCapsuleError("protocol_resource_limits_drifted")
    frozen = value.get("frozen_baseline_eligibility")
    if not isinstance(frozen, dict):
        raise ReproductionCapsuleError("frozen_eligibility_contract_missing")
    audit_path = _safe_root_path(
        repository_root, str(frozen.get("legacy_audit_path") or "")
    )
    if not audit_path.is_file() or sha256_file(audit_path) != frozen.get("sha256"):
        raise CapsuleNotEligible(
            "frozen_legacy_audit_identity_drift",
            stage="frozen_eligibility",
            invariant="tracked_legacy_evidence",
        )
    return value


def export_capsule(
    source_root: Path,
    archive_path: Path,
    *,
    host_repository_root: Path,
) -> dict[str, Any]:
    """Seal one complete new-format Replay run into a deterministic tar."""

    source = source_root.resolve()
    manifest_path = source / "run_manifest.json"
    replay_protocol_path = source / "replay_protocol.json"
    if not manifest_path.is_file() or not replay_protocol_path.is_file():
        raise CapsuleNotEligible(
            "self_contained_source_contract_missing",
            stage="export",
            invariant="new_format_source",
        )
    try:
        validation = validate_run_manifest(manifest_path, repository_root=source)
    except (OSError, ValueError) as exc:
        raise CapsuleNotEligible(
            "run_manifest_v1_invalid",
            stage="export",
            invariant="run_manifest_v1",
        ) from exc
    if validation["status"] != "passed":
        raise CapsuleNotEligible(
            "run_manifest_v1_invalid",
            stage="export",
            invariant="run_manifest_v1",
        )
    run_manifest = RunManifestV1.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if _git_commit(host_repository_root) != run_manifest.git.commit:
        raise CapsuleNotEligible(
            "source_run_code_commit_does_not_match_host",
            stage="export",
            invariant="code_identity",
        )
    if (
        run_manifest.progress.status != "completed"
        or run_manifest.git.unexpected_dirty_paths
    ):
        raise CapsuleNotEligible(
            "run_not_complete_or_worktree_unsealed",
            stage="export",
            invariant="completed_clean_run",
        )
    run_directory = _safe_root_path(source, run_manifest.output_directory)
    store = BenchmarkRunCommitStore(run_directory)
    try:
        state = store.load_latest()
    except CrashConsistencyError as exc:
        raise CapsuleNotEligible(
            "committed_generation_chain_unavailable",
            stage="export",
            invariant="generation_chain",
        ) from exc
    if (
        state.status != "completed"
        or state.run_id != run_manifest.run_id
        or len(state.records) != run_manifest.progress.completed_count
    ):
        raise CapsuleNotEligible(
            "committed_generation_does_not_match_run_manifest",
            stage="export",
            invariant="generation_run_binding",
        )
    replay_protocol = load_execution_protocol(
        replay_protocol_path, repository_root=source
    )
    query_ids = [str(item) for item in state.expected_query_ids]
    expected_records = [dict(item) for item in state.records]
    if [str(item.get("case_id")) for item in expected_records] != query_ids:
        raise CapsuleNotEligible(
            "committed_record_order_mismatch",
            stage="export",
            invariant="query_order",
        )
    expected_output_sha256 = stable_hash(expected_records)
    generation_paths = _generation_chain_paths(store, state.generation)
    roles_by_path: dict[str, set[str]] = {}

    def register(path_value: str, role: str) -> None:
        relative = PurePosixPath(path_value).as_posix()
        _validate_source_relative_path(relative)
        path = _safe_root_path(source, relative)
        if not path.is_file():
            raise CapsuleNotEligible(
                "registered_source_file_missing",
                stage="export",
                invariant="self_contained_input",
                location=relative,
            )
        roles_by_path.setdefault(relative, set()).add(role)

    register("run_manifest.json", "run_manifest_v1")
    register("replay_protocol.json", "production_replay_protocol")
    register(run_manifest.queries.input.path, "query_input")
    register(run_manifest.prompt.manifest.path, "prompt_registry")
    if run_manifest.comparison is not None:
        register(run_manifest.comparison.plan.path, "comparison_plan_v1")
    if run_manifest.shard is not None:
        register(run_manifest.shard.plan.path, "shard_plan_v1")
    if run_manifest.prompt.used:
        prompt_manifest = json.loads(
            _safe_root_path(source, run_manifest.prompt.manifest.path).read_text(
                encoding="utf-8"
            )
        )
        prompt_root = PurePosixPath(run_manifest.prompt.manifest.path).parent
        for entry in prompt_manifest.values():
            if not isinstance(entry, dict):
                continue
            for field in ("system", "user", "prompt"):
                if entry.get(field):
                    register(
                        (prompt_root / str(entry[field])).as_posix(),
                        "prompt_template",
                    )
    for item in run_manifest.dataset.inputs:
        register(item.path, "dataset_input")
    for item in run_manifest.outputs:
        register(item.path, f"committed_output:{item.role}")
    fixture = replay_protocol["fixture"]
    register(str(fixture["retrieval_outputs_path"]), "replay_input")
    register(
        str(replay_protocol["frozen_baseline_eligibility"]["legacy_audit_path"]),
        "legacy_eligibility_evidence",
    )
    for path in generation_paths:
        register(path.relative_to(source).as_posix(), "committed_generation")

    files = []
    content_by_member: dict[str, bytes] = {}
    for relative in sorted(roles_by_path):
        source_path = _safe_root_path(source, relative)
        content = source_path.read_bytes()
        _reject_sensitive_content(relative, content)
        member = f"{PAYLOAD_PREFIX}/{relative}"
        files.append(
            CapsuleFile(
                path=member,
                roles=sorted(roles_by_path[relative]),
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
        content_by_member[member] = content
    latest_generation = state.generation_path.relative_to(source).as_posix()
    capsule_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": CONTRACT_VERSION,
        "score_scope": "portable_replay_only_not_quality_or_official_score",
        "generated_by_git_commit": run_manifest.git.commit,
        "run_manifest": {
            "path": f"{PAYLOAD_PREFIX}/run_manifest.json",
            "sha256": sha256_file(manifest_path),
            "run_id": run_manifest.run_id,
            "schema_version": run_manifest.schema_version,
        },
        "query_set": {
            "count": run_manifest.queries.count,
            "stable_identity_sha256": run_manifest.queries.stable_identity_sha256,
            "order_sha256": run_manifest.queries.order_sha256,
            "input_sha256": run_manifest.queries.input.sha256,
            "committed_query_order_sha256": stable_hash(query_ids),
        },
        "replay": {
            "protocol_path": f"{PAYLOAD_PREFIX}/replay_protocol.json",
            "protocol_sha256": sha256_file(replay_protocol_path),
            "fixture_sha256": fixture["sha256"],
            "expected_records_sha256": expected_output_sha256,
            "expected_query_count": len(expected_records),
            "canonicalization_policy": replay_protocol["canonicalization"]["policy"],
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
        },
        "prompt": run_manifest.prompt.model_dump(mode="json"),
        "configuration": run_manifest.configuration.model_dump(mode="json"),
        "evaluator": run_manifest.evaluator.model_dump(mode="json"),
        "determinism": run_manifest.determinism.model_dump(mode="json"),
        "generation_chain": {
            "store_contract": "crash_consistency_v1",
            "latest_generation": state.generation,
            "generation_count": len({path.parent for path in generation_paths}),
            "latest_generation_path": f"{PAYLOAD_PREFIX}/{latest_generation}",
            "status": state.status,
            "record_count": len(state.records),
            "event_count": state.event_count,
            "chain_sha256": stable_hash(
                [
                    {
                        "path": path.relative_to(source).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in generation_paths
                ]
            ),
        },
        "entrypoint": {
            "kind": "host_search_service_replay",
            "command": [
                "python",
                "scripts/check_reproduction_capsule.py",
                "replay",
                "--capsule",
                "<capsule.tar>",
            ],
            "execute_archived_code": False,
        },
        "files": [item.model_dump(mode="json") for item in files],
        "limits": {
            "max_archive_bytes": MAX_ARCHIVE_BYTES,
            "max_file_count": MAX_MEMBERS,
            "max_member_bytes": MAX_MEMBER_BYTES,
            "max_total_unpacked_bytes": MAX_TOTAL_BYTES,
        },
    }
    if run_manifest.comparison is not None:
        capsule_data["comparison"] = run_manifest.comparison.model_dump(mode="json")
    if run_manifest.shard is not None:
        capsule_data["shard"] = run_manifest.shard.model_dump(mode="json")
    if run_manifest.provider_ingest is not None:
        capsule_data["provider_ingest"] = run_manifest.provider_ingest.model_dump(
            mode="json"
        )
    capsule_data["capsule_summary_sha256"] = stable_hash(capsule_data)
    manifest = CapsuleManifestV1.model_validate(capsule_data)
    _write_deterministic_tar(archive_path, manifest, content_by_member)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        archive_path.unlink(missing_ok=True)
        raise CapsuleNotEligible(
            "capsule_archive_size_limit_exceeded",
            stage="export",
            invariant="resource_limit",
        )
    return _report(
        status="passed",
        exit_code=EXIT_PASSED,
        stage="export",
        capsule_sha256=sha256_file(archive_path),
        capsule_size_bytes=archive_path.stat().st_size,
        manifest=manifest,
    )


def verify_capsule(archive_path: Path) -> dict[str, Any]:
    """Verify archive safety, hashes, run manifest and generation lineage."""

    manifest, content = _read_and_verify_archive(archive_path)
    with tempfile.TemporaryDirectory(prefix="spar-capsule-verify-") as value:
        destination = Path(value) / "imported"
        _materialize_verified_content(destination, content)
        _verify_extracted_contract(destination, manifest)
    return _report(
        status="passed",
        exit_code=EXIT_PASSED,
        stage="verify",
        capsule_sha256=sha256_file(archive_path),
        capsule_size_bytes=archive_path.stat().st_size,
        manifest=manifest,
    )


def replay_capsule(
    archive_path: Path,
    *,
    host_repository_root: Path,
    fault: Literal["semantic_result_change"] | None = None,
) -> dict[str, Any]:
    """Safely unpack and replay using host code and the production Replay seam."""

    manifest, content = _read_and_verify_archive(archive_path)
    current_commit = _git_commit(host_repository_root)
    if current_commit != manifest.generated_by_git_commit:
        raise CapsuleNotEligible(
            "host_code_commit_does_not_match_capsule",
            stage="replay",
            invariant="code_identity",
        )
    with tempfile.TemporaryDirectory(prefix="spar-capsule-replay-") as value:
        destination = Path(value) / "imported"
        _materialize_verified_content(destination, content)
        extracted = _verify_extracted_contract(destination, manifest)
        payload_root = destination / PAYLOAD_PREFIX
        replay_protocol_path = _safe_root_path(
            destination, str(manifest.replay["protocol_path"])
        )
        replay_protocol = load_execution_protocol(
            replay_protocol_path, repository_root=payload_root
        )
        observed = replay_canonical_fixture(
            replay_protocol,
            repository_root=payload_root,
            snapshot_root=destination / "snapshot-write-sentinel",
            fault=fault,
        )
        expected_records = [dict(item) for item in extracted.records]
        observed_records = [dict(item) for item in observed["records"]]
        differences = compare_profiles(expected_records, observed_records, max_diffs=1)
        if differences:
            first = differences[0]
            query_identity = _query_identity_for_difference(
                expected_records, observed_records, str(first["path"])
            )
            raise CapsuleIntegrityError(
                "portable_replay_semantics_mismatch",
                stage="replay",
                invariant="canonical_query_results",
                location=f"{query_identity}:{first['path']}",
            )
        if observed["records_sha256"] != manifest.replay["expected_records_sha256"]:
            raise CapsuleIntegrityError(
                "portable_replay_summary_mismatch",
                stage="replay",
                invariant="expected_output_digest",
                location="$.replay.expected_records_sha256",
            )
    report = _report(
        status="passed",
        exit_code=EXIT_PASSED,
        stage="replay",
        capsule_sha256=sha256_file(archive_path),
        capsule_size_bytes=archive_path.stat().st_size,
        manifest=manifest,
    )
    report["replay"] = {
        "query_count": observed["query_count"],
        "query_identity_order_sha256": observed["query_identity_order_sha256"],
        "records_sha256": observed["records_sha256"],
        "network_request_count": observed["execution"]["network_request_count"],
        "llm_request_count": observed["execution"]["llm_request_count"],
        "snapshot_write_count": observed["execution"]["snapshot_write_count"],
        "archived_code_execution_count": 0,
    }
    return report


def audit_frozen_baseline_eligibility(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    frozen = protocol["frozen_baseline_eligibility"]
    path = _safe_root_path(repository_root, str(frozen["legacy_audit_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = []
    for row in sorted(payload.get("profiles") or [], key=lambda item: item["profile_id"]):
        profiles.append(
            {
                "profile_id": row["profile_id"],
                "status": "not_eligible",
                "reason": "non_self_contained_legacy_run",
                "missing_fields": sorted(
                    set(row.get("missing_run_manifest_v1_fields") or [])
                    | {
                        "reproduction_capsule.complete_replay_inputs",
                        "reproduction_capsule.committed_generation_chain",
                        "reproduction_capsule.expected_canonical_output",
                    }
                ),
                "files_modified": 0,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": "not_eligible",
        "exit_code": EXIT_NOT_ELIGIBLE,
        "score_scope": "portable_replay_only_not_quality_or_official_score",
        "profile_count": len(profiles),
        "profiles": profiles,
        "execution": _zero_execution(),
    }


def materialize_local_replay_run(
    source_root: Path,
    *,
    host_repository_root: Path,
    execution_protocol_path: Path,
    git_commit: str | None = None,
) -> Path:
    """Create a gold-blind eligible run in a caller-owned temporary directory.

    This helper is for offline gate validation.  It creates ordinary production
    Replay inputs, a ``run_manifest_v1`` and a committed generation chain; it
    never creates a capsule or modifies tracked/frozen artifacts.
    """

    source_root.mkdir(parents=True, exist_ok=False)
    host_protocol = load_execution_protocol(
        execution_protocol_path, repository_root=host_repository_root
    )
    inputs = source_root / "inputs"
    inputs.mkdir()
    fixture_source = _safe_root_path(
        host_repository_root, host_protocol["fixture"]["retrieval_outputs_path"]
    )
    fixture_target = inputs / "retrieval_outputs.json"
    shutil.copyfile(fixture_source, fixture_target)
    legacy_source = _safe_root_path(
        host_repository_root,
        host_protocol["frozen_baseline_eligibility"]["legacy_audit_path"],
    )
    legacy_target = inputs / "legacy_audit.json"
    shutil.copyfile(legacy_source, legacy_target)
    prompt_source = host_repository_root / "src/scholar_agent/prompts/manifest.json"
    prompt_target = inputs / "prompt_manifest.json"
    shutil.copyfile(prompt_source, prompt_target)
    replay_protocol = json.loads(json.dumps(host_protocol))
    replay_protocol["fixture"]["retrieval_outputs_path"] = (
        "inputs/retrieval_outputs.json"
    )
    replay_protocol["fixture"]["size_bytes"] = fixture_target.stat().st_size
    replay_protocol["fixture"]["sha256"] = sha256_file(fixture_target)
    replay_protocol["frozen_baseline_eligibility"] = {
        "legacy_audit_path": "inputs/legacy_audit.json",
        "sha256": sha256_file(legacy_target),
    }
    replay_protocol_path = source_root / "replay_protocol.json"
    write_manifest_json(replay_protocol_path, replay_protocol)
    loaded = load_execution_protocol(replay_protocol_path, repository_root=source_root)
    query_fixtures, _retrieval_outputs = load_query_fixtures(
        loaded, repository_root=source_root
    )
    replay = replay_canonical_fixture(
        loaded,
        repository_root=source_root,
        snapshot_root=source_root / "snapshot-sentinel",
    )
    query_rows = [
        {
            "query_id": fixture.identity,
            "query": fixture.query,
        }
        for fixture in query_fixtures
    ]
    query_path = inputs / "queries.jsonl"
    query_path.write_bytes(
        b"".join(stable_json_bytes(item, indent=None) for item in query_rows)
    )
    prompt_payload = json.loads(prompt_target.read_text(encoding="utf-8"))
    prompt_versions = {
        key: str(value["version"])
        for key, value in sorted(prompt_payload.items())
        if isinstance(value, dict) and "version" in value
    }
    execution = loaded["execution"]
    arguments = execution["search_service_arguments"]
    sources = [str(item) for item in arguments["sources_override"]]
    budget = dict(arguments["budget"])
    evaluator = {"name": "none", "version": "replay_semantics_v1"}
    config = {
        "case_ids": [item["case_id"] for item in replay["records"]],
        "dataset": {"name": "gold_blind_replay_fixture", "version": "1"},
        "prompt": {"versions": prompt_versions},
        "configuration": {
            "sources": sources,
            "budgets": budget,
            "values": {
                "query_planning_policy": arguments["query_planning_policy"],
                "ranking_policy": arguments["ranking_policy"],
                "top_k": arguments["top_k"],
            },
        },
        "evaluator": evaluator,
        "replay_protocol_sha256": sha256_file(replay_protocol_path),
    }
    artifacts = source_root / "artifacts"
    store = BenchmarkRunCommitStore(artifacts)
    store.initialize(
        run_id="offline-portable-replay-fixture",
        expected_query_ids=config["case_ids"],
        config=config,
        dataset_report={
            "name": "gold_blind_replay_fixture",
            "query_count": len(query_rows),
            "gold_accessed": False,
        },
    )
    for record in replay["records"]:
        store.commit_record(record)
    completed = store.commit_completion({})
    store.materialize_compatibility_view(completed)
    commit = git_commit or _git_commit(host_repository_root)
    git_payload = {
        "commit": commit,
        "dirty_paths": ["third_party/paper-qa"],
        "allowed_dirty_paths": ["third_party/paper-qa"],
        "unexpected_dirty_paths": [],
    }
    git_identity = GitProvenance(
        **git_payload,
        dirty=True,
        worktree_state_sha256=stable_hash(git_payload),
    )
    outputs = []
    for name in ("config.json", "dataset_report.json", "results.jsonl", "failures.jsonl"):
        outputs.append(
            {
                "path": f"artifacts/{name}",
                "role": {
                    "config.json": "run_configuration",
                    "dataset_report.json": "dataset_identity",
                    "results.jsonl": "canonical_replay_records",
                    "failures.jsonl": "terminal_failures",
                }[name],
                "format": "jsonl" if name.endswith(".jsonl") else "json",
            }
        )
    excluded = sorted(
        path.relative_to(artifacts).as_posix()
        for path in (artifacts / STORE_DIRECTORY).rglob("*")
        if path.is_file()
    )
    spec = {
        "run_id": "offline-portable-replay-fixture",
        "dataset": {
            "name": "gold_blind_replay_fixture",
            "version": "1",
            "input_paths": [
                "inputs/queries.jsonl",
                "inputs/retrieval_outputs.json",
            ],
        },
        "queries": {
            "input_path": "inputs/queries.jsonl",
            "id_field": "query_id",
            "text_field": "query",
        },
        "prompt": {
            "manifest_path": "inputs/prompt_manifest.json",
            "versions": prompt_versions,
            "used": False,
        },
        "configuration": {
            "sources": sources,
            "budgets": budget,
            "values": config["configuration"]["values"],
        },
        "evaluator": evaluator,
        "determinism": {
            "random_seed": 0,
            "parameters": {
                "replay_protocol_sha256": sha256_file(replay_protocol_path),
                "canonicalization_policy": loaded["canonicalization"]["policy"],
            },
        },
        "progress": {
            "status": "completed",
            "expected_count": len(query_rows),
            "completed_count": len(query_rows),
            "record_output_path": "artifacts/results.jsonl",
        },
        "lineage": {
            "checkpoint_id": f"generation-{completed.generation:08d}",
            "resume_index": 0,
            "parent": None,
        },
        "output_directory": "artifacts",
        "output_inventory_excludes": excluded,
        "outputs": outputs,
        "metadata_bindings": [
            {
                "artifact_path": "artifacts/config.json",
                "artifact_json_pointer": "/dataset/name",
                "manifest_json_pointer": "/dataset/name",
            },
            {
                "artifact_path": "artifacts/config.json",
                "artifact_json_pointer": "/dataset/version",
                "manifest_json_pointer": "/dataset/version",
            },
            {
                "artifact_path": "artifacts/config.json",
                "artifact_json_pointer": "/prompt/versions",
                "manifest_json_pointer": "/prompt/versions",
            },
            {
                "artifact_path": "artifacts/config.json",
                "artifact_json_pointer": "/configuration/sources",
                "manifest_json_pointer": "/configuration/sources",
            },
            {
                "artifact_path": "artifacts/config.json",
                "artifact_json_pointer": "/configuration/budgets",
                "manifest_json_pointer": "/configuration/budgets",
            },
            {
                "artifact_path": "artifacts/config.json",
                "artifact_json_pointer": "/evaluator/name",
                "manifest_json_pointer": "/evaluator/name",
            },
            {
                "artifact_path": "artifacts/config.json",
                "artifact_json_pointer": "/evaluator/version",
                "manifest_json_pointer": "/evaluator/version",
            },
        ],
    }
    manifest = build_run_manifest(
        spec, repository_root=source_root, git_provenance=git_identity
    )
    write_manifest_json(source_root / "run_manifest.json", manifest.model_dump(mode="json"))
    return source_root


def _write_deterministic_tar(
    archive_path: Path,
    manifest: CapsuleManifestV1,
    content_by_member: Mapping[str, bytes],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f".{archive_path.name}.pending")
    temporary.unlink(missing_ok=True)
    manifest_bytes = stable_json_bytes(
        manifest.model_dump(mode="json", exclude_unset=True)
    )
    members = {CAPSULE_MANIFEST_NAME: manifest_bytes, **dict(content_by_member)}
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(members):
                content = members[name]
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = FIXED_FILE_MODE
                import io

                archive.addfile(info, io.BytesIO(content))
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_and_verify_archive(
    archive_path: Path,
) -> tuple[CapsuleManifestV1, dict[str, bytes]]:
    if not archive_path.is_file():
        raise CapsuleIntegrityError(
            "capsule_missing", stage="verify", invariant="archive_presence"
        )
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise CapsuleIntegrityError(
            "archive_size_limit_exceeded",
            stage="verify",
            invariant="resource_limit",
        )
    try:
        archive = tarfile.open(archive_path, mode="r:")
    except (tarfile.TarError, OSError) as exc:
        raise CapsuleIntegrityError(
            "archive_must_be_uncompressed_ustar",
            stage="verify",
            invariant="deterministic_archive_format",
        ) from exc
    with archive:
        members = archive.getmembers()
        if len(members) > MAX_MEMBERS:
            raise CapsuleIntegrityError(
                "archive_file_count_limit_exceeded",
                stage="verify",
                invariant="resource_limit",
            )
        seen: set[str] = set()
        normalized_seen: set[str] = set()
        total = 0
        content: dict[str, bytes] = {}
        for member in members:
            name = member.name
            _validate_capsule_path(name)
            normalized = unicodedata.normalize("NFC", name).casefold()
            if name in seen:
                raise CapsuleIntegrityError(
                    "duplicate_archive_member",
                    stage="verify",
                    invariant="unique_member_path",
                    location=name,
                )
            if normalized in normalized_seen:
                raise CapsuleIntegrityError(
                    "normalized_archive_path_collision",
                    stage="verify",
                    invariant="portable_member_path",
                    location=name,
                )
            seen.add(name)
            normalized_seen.add(normalized)
            if not member.isfile() or member.issym() or member.islnk():
                raise CapsuleIntegrityError(
                    "archive_links_or_special_files_forbidden",
                    stage="verify",
                    invariant="regular_files_only",
                    location=name,
                )
            if (
                member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname not in {"", None}
                or member.gname not in {"", None}
                or member.mode & 0o777 != FIXED_FILE_MODE
                or bool(member.pax_headers)
            ):
                raise CapsuleIntegrityError(
                    "archive_metadata_not_deterministic",
                    stage="verify",
                    invariant="deterministic_archive_metadata",
                    location=name,
                )
            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise CapsuleIntegrityError(
                    "archive_member_size_limit_exceeded",
                    stage="verify",
                    invariant="resource_limit",
                    location=name,
                )
            total += member.size
            if total > MAX_TOTAL_BYTES:
                raise CapsuleIntegrityError(
                    "archive_total_size_limit_exceeded",
                    stage="verify",
                    invariant="resource_limit",
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise CapsuleIntegrityError(
                    "archive_member_unreadable",
                    stage="verify",
                    invariant="member_readable",
                    location=name,
                )
            value = stream.read(MAX_MEMBER_BYTES + 1)
            if len(value) != member.size:
                raise CapsuleIntegrityError(
                    "archive_member_truncated",
                    stage="verify",
                    invariant="member_size",
                    location=name,
                )
            content[name] = value
    manifest_bytes = content.get(CAPSULE_MANIFEST_NAME)
    if manifest_bytes is None or len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise CapsuleIntegrityError(
            "capsule_manifest_missing_or_oversized",
            stage="verify",
            invariant="capsule_manifest",
        )
    try:
        raw_manifest = json.loads(manifest_bytes.decode("utf-8"))
        manifest = CapsuleManifestV1.model_validate(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise CapsuleIntegrityError(
            "capsule_manifest_invalid",
            stage="verify",
            invariant="capsule_manifest",
        ) from exc
    expected = {CAPSULE_MANIFEST_NAME, *(item.path for item in manifest.files)}
    actual = set(content)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CapsuleIntegrityError(
            "capsule_member_inventory_mismatch",
            stage="verify",
            invariant="closed_file_inventory",
            location=(missing or extra or ["$"])[0],
        )
    for item in manifest.files:
        value = content[item.path]
        if len(value) != item.size_bytes or hashlib.sha256(value).hexdigest() != item.sha256:
            raise CapsuleIntegrityError(
                "capsule_member_hash_or_size_mismatch",
                stage="verify",
                invariant="file_identity",
                location=item.path,
            )
    return manifest, content


def _materialize_verified_content(destination: Path, content: Mapping[str, bytes]) -> None:
    if destination.exists():
        raise CapsuleIntegrityError(
            "import_destination_exists",
            stage="import",
            invariant="no_partial_overwrite",
        )
    stage = destination.with_name(destination.name + ".pending")
    if stage.exists():
        shutil.rmtree(stage)
    try:
        stage.mkdir(parents=True)
        for name in sorted(content):
            path = stage.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content[name])
            path.chmod(FIXED_FILE_MODE)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _verify_extracted_contract(
    destination: Path, manifest: CapsuleManifestV1
):
    payload_root = destination / PAYLOAD_PREFIX
    run_manifest_path = payload_root / "run_manifest.json"
    if (
        sha256_file(run_manifest_path) != manifest.run_manifest["sha256"]
        or manifest.run_manifest["schema_version"] != "1"
    ):
        raise CapsuleIntegrityError(
            "run_manifest_identity_drift",
            stage="verify",
            invariant="run_manifest_v1",
        )
    result = validate_run_manifest(run_manifest_path, repository_root=payload_root)
    if result["status"] != "passed":
        raise CapsuleIntegrityError(
            "extracted_run_manifest_invalid",
            stage="verify",
            invariant="run_manifest_v1",
        )
    run_manifest = RunManifestV1.model_validate(
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
    )
    if run_manifest.git.commit != manifest.generated_by_git_commit:
        raise CapsuleIntegrityError(
            "run_manifest_code_identity_mismatch",
            stage="verify",
            invariant="code_identity",
        )
    for key in ("count", "stable_identity_sha256", "order_sha256"):
        if getattr(run_manifest.queries, key) != manifest.query_set[key]:
            raise CapsuleIntegrityError(
                "query_identity_or_order_drift",
                stage="verify",
                invariant="query_set",
                location=f"$.query_set.{key}",
            )
    if run_manifest.queries.input.sha256 != manifest.query_set["input_sha256"]:
        raise CapsuleIntegrityError(
            "query_input_identity_drift",
            stage="verify",
            invariant="query_set",
            location="$.query_set.input_sha256",
        )
    for field, observed in (
        ("prompt", run_manifest.prompt.model_dump(mode="json")),
        ("configuration", run_manifest.configuration.model_dump(mode="json")),
        ("evaluator", run_manifest.evaluator.model_dump(mode="json")),
        ("determinism", run_manifest.determinism.model_dump(mode="json")),
        (
            "comparison",
            run_manifest.comparison.model_dump(mode="json")
            if run_manifest.comparison is not None
            else None,
        ),
        (
            "shard",
            run_manifest.shard.model_dump(mode="json")
            if run_manifest.shard is not None
            else None,
        ),
        (
            "provider_ingest",
            run_manifest.provider_ingest.model_dump(mode="json")
            if run_manifest.provider_ingest is not None
            else None,
        ),
    ):
        if getattr(manifest, field) != observed:
            raise CapsuleIntegrityError(
                f"{field}_contract_drift",
                stage="verify",
                invariant=field,
                location=f"$.{field}",
            )
    run_directory = _safe_root_path(payload_root, run_manifest.output_directory)
    store = BenchmarkRunCommitStore(run_directory)
    try:
        state = store.load_latest()
    except CrashConsistencyError as exc:
        raise CapsuleIntegrityError(
            "generation_chain_invalid",
            stage="verify",
            invariant="generation_chain",
        ) from exc
    if (
        state.status != "completed"
        or state.generation != manifest.generation_chain["latest_generation"]
        or len(state.records) != manifest.generation_chain["record_count"]
        or state.event_count != manifest.generation_chain["event_count"]
    ):
        raise CapsuleIntegrityError(
            "generation_summary_drift",
            stage="verify",
            invariant="generation_chain",
        )
    expected_latest_path = (
        f"payload/{run_manifest.output_directory}/.run_commits/generations/"
        f"generation-{state.generation:08d}"
    )
    if manifest.generation_chain["latest_generation_path"] != expected_latest_path:
        raise CapsuleIntegrityError(
            "latest_generation_path_drift",
            stage="verify",
            invariant="generation_chain",
        )
    if stable_hash(list(state.expected_query_ids)) != manifest.query_set[
        "committed_query_order_sha256"
    ]:
        raise CapsuleIntegrityError(
            "committed_query_order_drift",
            stage="verify",
            invariant="query_set",
            location="$.query_set.committed_query_order_sha256",
        )
    chain_paths = _generation_chain_paths(store, state.generation)
    observed_chain_sha256 = stable_hash(
        [
            {
                "path": path.resolve().relative_to(payload_root.resolve()).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in chain_paths
        ]
    )
    if (
        observed_chain_sha256 != manifest.generation_chain["chain_sha256"]
        or len({path.parent for path in chain_paths})
        != manifest.generation_chain["generation_count"]
    ):
        raise CapsuleIntegrityError(
            "generation_chain_digest_drift",
            stage="verify",
            invariant="generation_chain",
        )
    if stable_hash([dict(item) for item in state.records]) != manifest.replay[
        "expected_records_sha256"
    ]:
        raise CapsuleIntegrityError(
            "committed_expected_output_drift",
            stage="verify",
            invariant="expected_output_digest",
        )
    protocol_path = _safe_root_path(
        destination, str(manifest.replay["protocol_path"])
    )
    replay_protocol = load_execution_protocol(protocol_path, repository_root=payload_root)
    if sha256_file(protocol_path) != manifest.replay["protocol_sha256"]:
        raise CapsuleIntegrityError(
            "replay_protocol_hash_drift",
            stage="verify",
            invariant="replay_protocol",
        )
    if replay_protocol["fixture"]["sha256"] != manifest.replay["fixture_sha256"]:
        raise CapsuleIntegrityError(
            "replay_fixture_identity_drift",
            stage="verify",
            invariant="replay_input",
        )
    return state


def _generation_chain_paths(
    store: BenchmarkRunCommitStore, latest_generation: int
) -> list[Path]:
    generation = latest_generation
    directories: list[Path] = []
    seen: set[int] = set()
    while True:
        if generation in seen:
            raise CapsuleNotEligible(
                "generation_lineage_cycle",
                stage="export",
                invariant="generation_chain",
            )
        seen.add(generation)
        directory = store.generations / f"generation-{generation:08d}"
        manifest_path = directory / "generation_manifest.json"
        if not manifest_path.is_file():
            raise CapsuleNotEligible(
                "generation_manifest_missing",
                stage="export",
                invariant="generation_chain",
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        directories.append(directory)
        parent = payload.get("parent_generation")
        if parent is None:
            break
        generation = int(parent)
    directories.reverse()
    files = [
        path
        for directory in directories
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file()
    ]
    return files


def _capsule_summary_payload(manifest: CapsuleManifestV1) -> dict[str, Any]:
    value = manifest.model_dump(mode="json", exclude_unset=True)
    value.pop("capsule_summary_sha256", None)
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are not closed")


def _reject_sensitive_content(relative: str, content: bytes) -> None:
    """Reject credential-bearing JSON and machine paths before sealing."""

    if not relative.endswith((".json", ".jsonl")):
        return
    try:
        if relative.endswith(".jsonl"):
            values = [json.loads(line) for line in content.decode("utf-8").splitlines() if line]
        else:
            values = [json.loads(content.decode("utf-8"))]
    except (UnicodeDecodeError, json.JSONDecodeError):
        return

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).casefold().replace("-", "_")
                if any(
                    marker in normalized
                    for marker in ("authorization", "api_key", "access_token", "secret", "password")
                ) and child not in (None, "", False):
                    raise CapsuleNotEligible(
                        "sensitive_configuration_field_present",
                        stage="export",
                        invariant="secret_exclusion",
                        location=relative,
                    )
                visit(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
        elif isinstance(value, str):
            if _SECRET_TEXT_RE.search(value) or re.search(
                r"(?i)bearer\s+[A-Za-z0-9._~-]{8,}", value
            ):
                raise CapsuleNotEligible(
                    "sensitive_value_present",
                    stage="export",
                    invariant="secret_exclusion",
                    location=relative,
                )
            if re.search(
                r"(?:^|\s)(?:/Users/|/home/|/private/var/|[A-Za-z]:\\Users\\)",
                value,
            ):
                raise CapsuleNotEligible(
                    "machine_absolute_path_present",
                    stage="export",
                    invariant="machine_independent_content",
                    location=relative,
                )

    for item in values:
        visit(item, ())


def _report(
    *,
    status: str,
    exit_code: int,
    stage: str,
    capsule_sha256: str,
    capsule_size_bytes: int,
    manifest: CapsuleManifestV1,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": exit_code,
        "stage": stage,
        "score_scope": "portable_replay_only_not_quality_or_official_score",
        "capsule": {
            "sha256": capsule_sha256,
            "size_bytes": capsule_size_bytes,
            "file_count": len(manifest.files) + 1,
            "summary_sha256": manifest.capsule_summary_sha256,
        },
        "run": {
            "run_id": manifest.run_manifest["run_id"],
            "query_count": manifest.query_set["count"],
            "generation_count": manifest.generation_chain["generation_count"],
            "expected_records_sha256": manifest.replay["expected_records_sha256"],
        },
        "execution": _zero_execution(),
    }


def error_report(error: ReproductionCapsuleError, exit_code: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": (
            "not_eligible"
            if exit_code == EXIT_NOT_ELIGIBLE
            else "usage_error"
            if exit_code == EXIT_USAGE_ERROR
            else "integrity_or_replay_mismatch"
        ),
        "exit_code": exit_code,
        "score_scope": "portable_replay_only_not_quality_or_official_score",
        "violation": {
            "stage": error.stage,
            "invariant": error.invariant,
            "first_difference_path": sanitize_text(error.location),
            "reason": sanitize_text(error.reason),
            "normalized_summary_sha256": stable_hash(
                {
                    "stage": error.stage,
                    "invariant": error.invariant,
                    "location": sanitize_text(error.location),
                    "reason": sanitize_text(error.reason),
                }
            ),
        },
        "execution": _zero_execution(),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(dict(value)))


def sanitize_text(value: object) -> str:
    text = _SECRET_TEXT_RE.sub(r"\1\2[redacted]", str(value))
    text = re.sub(r"(?i)\.env(?:\.[A-Za-z0-9_-]+)?", "[environment-file]", text)
    text = _ABSOLUTE_PATH_RE.sub("[absolute-path]", text)
    return text[:300]


def _zero_execution() -> dict[str, int]:
    return {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "archived_code_execution_count": 0,
        "quality_metric_count": 0,
    }


def _validate_capsule_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or unicodedata.normalize("NFC", value) != value
    ):
        raise CapsuleIntegrityError(
            "unsafe_archive_member_path",
            stage="verify",
            invariant="portable_member_path",
            location=sanitize_text(value),
        )


def _validate_source_relative_path(value: str) -> None:
    _validate_capsule_path(value)
    parts = PurePosixPath(value).parts
    if any(part in _SECRET_PATH_PARTS or part.endswith(".log") for part in parts):
        raise CapsuleNotEligible(
            "forbidden_or_machine_specific_source_path",
            stage="export",
            invariant="portable_file_allowlist",
            location=value,
        )


def _safe_root_path(root: Path, value: str) -> Path:
    _validate_capsule_path(value)
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*PurePosixPath(value).parts).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ReproductionCapsuleError("path_resolves_outside_root") from exc
    return resolved


def _git_commit(repository_root: Path) -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _COMMIT_RE.fullmatch(value):
        raise ReproductionCapsuleError("git_commit_identity_invalid")
    return value


def _query_identity_for_difference(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
    path: str,
) -> str:
    match = re.match(r"^\$\[(\d+)\]", path)
    if not match:
        return "query_unknown"
    index = int(match.group(1))
    for values in (expected, observed):
        if index < len(values):
            return str(values[index].get("case_id") or "query_unknown")
    return "query_unknown"
