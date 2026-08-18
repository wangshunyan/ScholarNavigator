"""Versioned, deterministic provenance manifests for offline Benchmark runs.

The contract composes existing Snapshot/content hashes, query-only manifests,
Prompt versions, evaluator versions, and checkpoint lineage.  It never loads a
dataset adapter, evaluator, connector, LLM runtime, or project environment.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash


RUN_MANIFEST_SCHEMA_VERSION = "1"
RUN_MANIFEST_KIND = "run_manifest_v1"
RUN_PROVENANCE_GATE = "run_provenance_gate_v1"

REQUIRED_METADATA_BINDINGS = frozenset(
    {
        "/dataset/name",
        "/dataset/version",
        "/prompt/versions",
        "/configuration/sources",
        "/configuration/budgets",
        "/evaluator/name",
        "/evaluator/version",
    }
)

EXIT_OK = 0
EXIT_INTEGRITY_FAILURE = 2
EXIT_LEGACY_METADATA_INCOMPLETE = 3
EXIT_USAGE_ERROR = 4

RunStatus = Literal["planned", "running", "partial", "completed", "failed"]
ReportStatus = Literal["passed", "invalid", "legacy_metadata_incomplete"]


class RunProvenanceError(RuntimeError):
    """The provenance input cannot be represented without guessing."""


class FileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_portable_path(self) -> "FileIdentity":
        _validate_relative_path(self.path)
        return self


class OutputIdentity(FileIdentity):
    role: str
    format: Literal["json", "jsonl", "text", "binary"]
    record_count: int | None = Field(default=None, ge=0)


class DatasetIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    identity_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inputs: list[FileIdentity]


class QuerySetIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: FileIdentity
    id_field: str
    text_field: str
    count: int = Field(ge=0)
    stable_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PromptIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: FileIdentity
    versions: dict[str, str]
    used: bool


class RunConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[str]
    budgets: dict[str, int | float]
    values: dict[str, Any]
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_summary(self) -> "RunConfiguration":
        expected = stable_hash(
            {"sources": self.sources, "budgets": self.budgets, "values": self.values}
        )
        if expected != self.summary_sha256:
            raise ValueError("run configuration summary mismatch")
        return self


class EvaluatorIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str


class DeterminismIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    random_seed: int | None = None
    parameters: dict[str, Any]
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_summary(self) -> "DeterminismIdentity":
        if stable_hash(
            {"random_seed": self.random_seed, "parameters": self.parameters}
        ) != self.summary_sha256:
            raise ValueError("determinism summary mismatch")
        return self


class RunProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    expected_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    record_output_path: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "RunProgress":
        if self.completed_count > self.expected_count:
            raise ValueError("completed count exceeds expected count")
        if self.record_output_path is not None:
            _validate_relative_path(self.record_output_path)
        return self


class ParentRunLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str
    checkpoint_id: str

    @model_validator(mode="after")
    def validate_path(self) -> "ParentRunLink":
        _validate_relative_path(self.manifest_path)
        return self


class RunLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    resume_index: int = Field(ge=0)
    parent: ParentRunLink | None = None


class GitProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str = Field(min_length=40, max_length=40)
    dirty: bool
    dirty_paths: list[str]
    allowed_dirty_paths: list[str]
    unexpected_dirty_paths: list[str]
    worktree_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_state(self) -> "GitProvenance":
        for value in self.dirty_paths + self.allowed_dirty_paths + self.unexpected_dirty_paths:
            _validate_relative_path(value)
        expected = stable_hash(
            {
                "commit": self.commit,
                "dirty_paths": self.dirty_paths,
                "allowed_dirty_paths": self.allowed_dirty_paths,
                "unexpected_dirty_paths": self.unexpected_dirty_paths,
            }
        )
        if expected != self.worktree_state_sha256:
            raise ValueError("worktree state summary mismatch")
        if self.dirty != bool(self.dirty_paths):
            raise ValueError("dirty flag mismatch")
        return self


class MetadataBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_path: str
    artifact_json_pointer: str
    manifest_json_pointer: str

    @model_validator(mode="after")
    def validate_path(self) -> "MetadataBinding":
        _validate_relative_path(self.artifact_path)
        if not self.artifact_json_pointer.startswith("/"):
            raise ValueError("artifact JSON pointer must start with /")
        if not self.manifest_json_pointer.startswith("/"):
            raise ValueError("manifest JSON pointer must start with /")
        return self


class ComparisonRunBinding(BaseModel):
    """Pre-execution binding for one side of an offline paired experiment."""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["comparison_plan_v1"] = "comparison_plan_v1"
    plan: FileIdentity
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["baseline", "candidate"]
    common_execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_plan_identity(self) -> "ComparisonRunBinding":
        if self.plan.sha256 != self.plan_sha256:
            raise ValueError("comparison plan digest mismatch")
        return self


class ShardRunBinding(BaseModel):
    """Pre-execution binding for one attempt of a deterministic shard."""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["shard_plan_v1"] = "shard_plan_v1"
    plan: FileIdentity
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_index: int = Field(ge=0)
    shard_count: int = Field(ge=1)
    expected_query_identities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    common_execution_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    supersedes_attempt_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
    )

    @model_validator(mode="after")
    def validate_plan_identity(self) -> "ShardRunBinding":
        if self.plan.sha256 != self.plan_sha256:
            raise ValueError("shard plan digest mismatch")
        if self.shard_index >= self.shard_count:
            raise ValueError("shard index exceeds shard count")
        if self.supersedes_attempt_id == self.attempt_id:
            raise ValueError("shard attempt cannot supersede itself")
        return self


class ResourceLedgerBinding(BaseModel):
    """Optional authoritative accounting artifact for a new-format run."""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["resource_ledger_v1"] = "resource_ledger_v1"
    output: FileIdentity
    manifest_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority: Literal["committed_generation_only"] = "committed_generation_only"


class ProviderIngestBinding(BaseModel):
    """Optional parser-pre raw response provenance for a new-format run."""

    model_config = ConfigDict(extra="forbid")

    contract: Literal["provider_ingest_provenance_v1"] = (
        "provider_ingest_provenance_v1"
    )
    bundle: FileIdentity
    raw_archive: FileIdentity
    resource_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority: Literal["committed_generation_only"] = "committed_generation_only"


class RunManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_kind: Literal["run_manifest_v1"] = RUN_MANIFEST_KIND
    schema_version: Literal["1"] = RUN_MANIFEST_SCHEMA_VERSION
    run_id: str
    dataset: DatasetIdentity
    queries: QuerySetIdentity
    prompt: PromptIdentity
    configuration: RunConfiguration
    evaluator: EvaluatorIdentity
    determinism: DeterminismIdentity
    progress: RunProgress
    lineage: RunLineage
    git: GitProvenance
    output_directory: str
    output_inventory_excludes: list[str] = Field(default_factory=list)
    outputs: list[OutputIdentity]
    metadata_bindings: list[MetadataBinding] = Field(default_factory=list)
    comparison: ComparisonRunBinding | None = None
    shard: ShardRunBinding | None = None
    resource_ledger: ResourceLedgerBinding | None = None
    provider_ingest: ProviderIngestBinding | None = None
    score_scope: Literal["internal_not_official"] = "internal_not_official"

    @model_validator(mode="after")
    def validate_closed_contract(self) -> "RunManifestV1":
        _validate_relative_path(self.output_directory)
        for value in self.output_inventory_excludes:
            _validate_relative_path(value)
        if self.progress.expected_count != self.queries.count:
            raise ValueError("query and progress expected counts differ")
        output_paths = [item.path for item in self.outputs]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("duplicate output path")
        if self.progress.record_output_path not in {None, *output_paths}:
            raise ValueError("record output is not registered")
        if self.lineage.parent is None and self.lineage.resume_index != 0:
            raise ValueError("root run resume index must be zero")
        registered = set(output_paths)
        binding_targets = {
            item.manifest_json_pointer for item in self.metadata_bindings
        }
        missing_bindings = REQUIRED_METADATA_BINDINGS - binding_targets
        if missing_bindings:
            raise ValueError(
                "required metadata bindings missing:"
                + ",".join(sorted(missing_bindings))
            )
        if any(
            item.artifact_path not in registered for item in self.metadata_bindings
        ):
            raise ValueError("metadata binding artifact is not a registered output")
        if self.resource_ledger is not None:
            ledger_outputs = {
                item.path
                for item in self.outputs
                if item.role == "resource_ledger_v1"
            }
            if self.resource_ledger.output.path not in ledger_outputs:
                raise ValueError("resource ledger is not a registered ledger output")
        if self.provider_ingest is not None:
            required = {
                self.provider_ingest.bundle.path: "provider_ingest_provenance_v1",
                self.provider_ingest.raw_archive.path: "provider_ingest_raw_response_v1",
            }
            roles = {item.path: item.role for item in self.outputs}
            if any(roles.get(path) != role for path, role in required.items()):
                raise ValueError("provider ingest outputs are not registered with exact roles")
            if self.resource_ledger is None:
                raise ValueError("provider ingest provenance requires resource ledger binding")
            if (
                self.provider_ingest.resource_ledger_sha256
                != self.resource_ledger.output.sha256
            ):
                raise ValueError("provider ingest resource ledger digest mismatch")
        return self


def build_run_manifest(
    spec: Mapping[str, Any],
    *,
    repository_root: Path,
    git_provenance: GitProvenance | None = None,
) -> RunManifestV1:
    """Materialize a run manifest from a gold-blind, pre-registered spec."""

    dataset_spec = _mapping(spec, "dataset")
    dataset_inputs = [
        file_identity(str(path), repository_root)
        for path in sorted(str(path) for path in dataset_spec.get("input_paths", []))
    ]
    dataset = DatasetIdentity(
        name=str(dataset_spec["name"]),
        version=str(dataset_spec["version"]),
        inputs=dataset_inputs,
        identity_summary_sha256=stable_hash(
            {
                "name": str(dataset_spec["name"]),
                "version": str(dataset_spec["version"]),
                "inputs": [item.model_dump(mode="json") for item in dataset_inputs],
            }
        ),
    )
    query_spec = _mapping(spec, "queries")
    queries = build_query_identity(
        str(query_spec["input_path"]),
        repository_root=repository_root,
        id_field=str(query_spec["id_field"]),
        text_field=str(query_spec["text_field"]),
    )
    prompt_spec = _mapping(spec, "prompt")
    prompt = PromptIdentity(
        manifest=file_identity(str(prompt_spec["manifest_path"]), repository_root),
        versions={str(key): str(value) for key, value in prompt_spec["versions"].items()},
        used=bool(prompt_spec["used"]),
    )
    config_spec = _mapping(spec, "configuration")
    config_payload = {
        "sources": [str(value) for value in config_spec["sources"]],
        "budgets": dict(config_spec["budgets"]),
        "values": dict(config_spec["values"]),
    }
    configuration = RunConfiguration(
        **config_payload, summary_sha256=stable_hash(config_payload)
    )
    determinism_spec = _mapping(spec, "determinism")
    determinism_payload = {
        "random_seed": determinism_spec.get("random_seed"),
        "parameters": dict(determinism_spec["parameters"]),
    }
    determinism = DeterminismIdentity(
        **determinism_payload, summary_sha256=stable_hash(determinism_payload)
    )
    outputs = [
        output_identity(item, repository_root)
        for item in sorted(
            list(spec.get("outputs") or []), key=lambda value: str(value["path"])
        )
    ]
    comparison_spec = spec.get("comparison")
    comparison = None
    if comparison_spec is not None:
        if not isinstance(comparison_spec, Mapping):
            raise RunProvenanceError("spec section invalid:comparison")
        plan = file_identity(str(comparison_spec["plan_path"]), repository_root)
        comparison = ComparisonRunBinding(
            plan=plan,
            plan_sha256=plan.sha256,
            role=str(comparison_spec["role"]),
            common_execution_contract_sha256=str(
                comparison_spec["common_execution_contract_sha256"]
            ),
        )
    shard_spec = spec.get("shard")
    shard = None
    if shard_spec is not None:
        if not isinstance(shard_spec, Mapping):
            raise RunProvenanceError("spec section invalid:shard")
        plan = file_identity(str(shard_spec["plan_path"]), repository_root)
        shard = ShardRunBinding(
            plan=plan,
            plan_sha256=plan.sha256,
            shard_index=int(shard_spec["shard_index"]),
            shard_count=int(shard_spec["shard_count"]),
            expected_query_identities_sha256=str(
                shard_spec["expected_query_identities_sha256"]
            ),
            common_execution_contract_sha256=str(
                shard_spec["common_execution_contract_sha256"]
            ),
            attempt_id=str(shard_spec["attempt_id"]),
            supersedes_attempt_id=(
                str(shard_spec["supersedes_attempt_id"])
                if shard_spec.get("supersedes_attempt_id") is not None
                else None
            ),
        )
    resource_spec = spec.get("resource_ledger")
    resource_ledger = None
    if resource_spec is not None:
        if not isinstance(resource_spec, Mapping):
            raise RunProvenanceError("spec section invalid:resource_ledger")
        output = file_identity(str(resource_spec["output_path"]), repository_root)
        resource_ledger = ResourceLedgerBinding(
            output=output,
            manifest_identity=str(resource_spec["manifest_identity"]),
        )
    ingest_spec = spec.get("provider_ingest")
    provider_ingest = None
    if ingest_spec is not None:
        if not isinstance(ingest_spec, Mapping):
            raise RunProvenanceError("spec section invalid:provider_ingest")
        provider_ingest = ProviderIngestBinding(
            bundle=file_identity(str(ingest_spec["bundle_path"]), repository_root),
            raw_archive=file_identity(
                str(ingest_spec["raw_archive_path"]), repository_root
            ),
            resource_ledger_sha256=str(ingest_spec["resource_ledger_sha256"]),
            manifest_identity=str(ingest_spec["manifest_identity"]),
        )
    manifest = RunManifestV1(
        run_id=str(spec["run_id"]),
        dataset=dataset,
        queries=queries,
        prompt=prompt,
        configuration=configuration,
        evaluator=EvaluatorIdentity.model_validate(spec["evaluator"]),
        determinism=determinism,
        progress=RunProgress.model_validate(spec["progress"]),
        lineage=RunLineage.model_validate(spec["lineage"]),
        git=git_provenance or collect_git_provenance(repository_root),
        output_directory=str(spec["output_directory"]),
        output_inventory_excludes=sorted(
            str(value) for value in spec.get("output_inventory_excludes", [])
        ),
        outputs=outputs,
        comparison=comparison,
        shard=shard,
        resource_ledger=resource_ledger,
        provider_ingest=provider_ingest,
        metadata_bindings=[
            MetadataBinding.model_validate(item)
            for item in sorted(
                spec.get("metadata_bindings", []),
                key=lambda value: (
                    str(value["manifest_json_pointer"]),
                    str(value["artifact_path"]),
                    str(value["artifact_json_pointer"]),
                ),
            )
        ],
    )
    report = validate_run_manifest_document(manifest, repository_root=repository_root)
    if report["status"] != "passed":
        raise RunProvenanceError(json.dumps(report["violations"], sort_keys=True))
    return manifest


def validate_run_manifest(
    manifest_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    attempts = {"network": 0}
    with _forbid_network(attempts):
        violations: list[dict[str, Any]] = []
        manifest = _load_manifest(manifest_path, violations)
        if manifest is not None:
            violations.extend(
                _validate_manifest_recursive(
                    manifest,
                    manifest_path=manifest_path,
                    repository_root=repository_root,
                    ancestors=(),
                )
            )
        report = _validation_report(
            status="passed" if not violations else "invalid",
            violations=violations,
            network_count=attempts["network"],
        )
    return report


def validate_run_manifest_document(
    manifest: RunManifestV1, *, repository_root: Path
) -> dict[str, Any]:
    violations = _validate_manifest_files(manifest, repository_root)
    return _validation_report(
        status="passed" if not violations else "invalid",
        violations=violations,
        network_count=0,
    )


def build_query_identity(
    path_value: str,
    *,
    repository_root: Path,
    id_field: str,
    text_field: str,
) -> QuerySetIdentity:
    path = resolve_repo_path(repository_root, path_value)
    hashes: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RunProvenanceError(
                    f"invalid query JSONL:{path_value}:{line_number}"
                ) from exc
            if not isinstance(row, dict) or id_field not in row or text_field not in row:
                raise RunProvenanceError(
                    f"query fields missing:{path_value}:{line_number}"
                )
            hashes.append(
                stable_hash(
                    {"id": str(row[id_field]), "query": str(row[text_field])}
                )
            )
    return QuerySetIdentity(
        input=file_identity(path_value, repository_root),
        id_field=id_field,
        text_field=text_field,
        count=len(hashes),
        stable_identity_sha256=stable_hash(sorted(hashes)),
        order_sha256=stable_hash(hashes),
    )


def file_identity(path_value: str, repository_root: Path) -> FileIdentity:
    path = resolve_repo_path(repository_root, path_value)
    if not path.is_file():
        raise RunProvenanceError(f"file missing:{path_value}")
    return FileIdentity(
        path=PurePosixPath(path_value).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def output_identity(spec: Mapping[str, Any], repository_root: Path) -> OutputIdentity:
    base = file_identity(str(spec["path"]), repository_root)
    output_format = str(spec["format"])
    record_count = (
        count_jsonl_records(resolve_repo_path(repository_root, base.path))
        if output_format == "jsonl"
        else None
    )
    return OutputIdentity(
        **base.model_dump(mode="json"),
        role=str(spec["role"]),
        format=output_format,
        record_count=record_count,
    )


def collect_git_provenance(
    repository_root: Path,
    *,
    allowed_dirty_paths: Sequence[str] = ("third_party/paper-qa",),
) -> GitProvenance:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty_paths, allowed, unexpected = classify_worktree_paths(
        status_lines, allowed_dirty_paths
    )
    payload = {
        "commit": commit,
        "dirty_paths": dirty_paths,
        "allowed_dirty_paths": allowed,
        "unexpected_dirty_paths": unexpected,
    }
    return GitProvenance(
        **payload,
        dirty=bool(dirty_paths),
        worktree_state_sha256=stable_hash(payload),
    )


def classify_worktree_paths(
    status_lines: Sequence[str], allowed_dirty_paths: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    allowed_roots = sorted(
        set(PurePosixPath(value).as_posix() for value in allowed_dirty_paths)
    )
    paths: set[str] = set()
    for line in status_lines:
        if len(line) < 4:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        value = value.strip('"')
        _validate_relative_path(value)
        paths.add(PurePosixPath(value).as_posix())
    dirty = sorted(paths)
    accepted = sorted(
        path
        for path in dirty
        if any(path == root or path.startswith(root + "/") for root in allowed_roots)
    )
    unexpected = sorted(set(dirty) - set(accepted))
    return dirty, accepted, unexpected


def audit_legacy_profiles(
    profile_path: Path, *, repository_root: Path
) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != "1":
        return _legacy_report(
            "invalid",
            [
                _violation(
                    "schema_version_incompatible",
                    "$",
                    "1",
                    profile.get("schema_version"),
                )
            ],
        )
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for item in sorted(
        profile.get("profiles", []), key=lambda value: value["profile_id"]
    ):
        profile_id = str(item["profile_id"])
        file_states = []
        for frozen in sorted(
            item.get("frozen_files", []), key=lambda value: value["path"]
        ):
            path_value = str(frozen["path"])
            path = resolve_repo_path(repository_root, path_value)
            state = "verified"
            observed_hash = None
            if not path.is_file():
                state = "missing"
                violations.append(
                    _violation(
                        "file_missing",
                        f"{profile_id}:{path_value}",
                        frozen["sha256"],
                        None,
                    )
                )
            else:
                observed_hash = sha256_file(path)
                expected_size = frozen.get("size_bytes")
                if expected_size is not None and path.stat().st_size != int(
                    expected_size
                ):
                    state = "tampered"
                    violations.append(
                        _violation(
                            "file_size_mismatch",
                            f"{profile_id}:{path_value}",
                            int(expected_size),
                            path.stat().st_size,
                        )
                    )
                if observed_hash != frozen["sha256"]:
                    state = "tampered"
                    violations.append(
                        _violation(
                            "file_tampered",
                            f"{profile_id}:{path_value}",
                            frozen["sha256"],
                            observed_hash,
                        )
                    )
            file_states.append(
                {"path": path_value, "state": state, "sha256": observed_hash}
            )
        results_path = resolve_repo_path(repository_root, str(item["results_path"]))
        observed_count = count_jsonl_records(results_path) if results_path.is_file() else 0
        expected_count = int(item["expected_query_count"])
        missing_fields = sorted(
            set(str(value) for value in item["missing_run_manifest_v1_fields"])
        )
        status = "legacy_metadata_incomplete"
        rows.append(
            {
                "profile_id": profile_id,
                "status": status,
                "expected_query_count": expected_count,
                "observed_record_count": observed_count,
                "completed_count_claim_available": False,
                "record_is_complete": observed_count >= expected_count,
                "downstream_main_analysis_count": item.get(
                    "downstream_main_analysis_count"
                ),
                "downstream_excluded_count": item.get("downstream_excluded_count"),
                "missing_run_manifest_v1_fields": missing_fields,
                "frozen_files": file_states,
            }
        )
    if violations:
        return _legacy_report("invalid", violations, rows)
    return _legacy_report("legacy_metadata_incomplete", [], rows)


def count_jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def resolve_repo_path(repository_root: Path, path_value: str) -> Path:
    _validate_relative_path(path_value)
    root = repository_root.resolve()
    path = root.joinpath(*PurePosixPath(path_value).parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path resolves outside repository:{path_value}") from exc
    return path


def _validate_manifest_recursive(
    manifest: RunManifestV1,
    *,
    manifest_path: Path,
    repository_root: Path,
    ancestors: tuple[Path, ...],
) -> list[dict[str, Any]]:
    resolved = manifest_path.resolve()
    if resolved in ancestors:
        return [
            _violation(
                "lineage_cycle",
                manifest.run_id,
                "acyclic",
                _display_path(resolved, repository_root),
            )
        ]
    violations = _validate_manifest_files(manifest, repository_root)
    parent = manifest.lineage.parent
    if parent is None:
        return violations
    parent_path = resolve_repo_path(repository_root, parent.manifest_path)
    if not parent_path.is_file():
        violations.append(
            _violation(
                "lineage_parent_missing", parent.manifest_path, "existing", None
            )
        )
        return violations
    actual_hash = sha256_file(parent_path)
    if actual_hash != parent.manifest_sha256:
        violations.append(
            _violation(
                "lineage_parent_hash_mismatch",
                parent.manifest_path,
                parent.manifest_sha256,
                actual_hash,
            )
        )
    parent_violations: list[dict[str, Any]] = []
    parent_manifest = _load_manifest(parent_path, parent_violations)
    violations.extend(parent_violations)
    if parent_manifest is None:
        return violations
    if parent_manifest.run_id != parent.run_id:
        violations.append(
            _violation(
                "lineage_parent_run_mismatch",
                manifest.run_id,
                parent.run_id,
                parent_manifest.run_id,
            )
        )
    if parent_manifest.lineage.checkpoint_id != parent.checkpoint_id:
        violations.append(
            _violation(
                "lineage_checkpoint_mismatch",
                manifest.run_id,
                parent.checkpoint_id,
                parent_manifest.lineage.checkpoint_id,
            )
        )
    if manifest.lineage.resume_index != parent_manifest.lineage.resume_index + 1:
        violations.append(
            _violation(
                "lineage_resume_index_mismatch",
                manifest.run_id,
                parent_manifest.lineage.resume_index + 1,
                manifest.lineage.resume_index,
            )
        )
    if (
        manifest.queries.stable_identity_sha256
        != parent_manifest.queries.stable_identity_sha256
        or manifest.queries.order_sha256 != parent_manifest.queries.order_sha256
    ):
        violations.append(
            _violation(
                "lineage_query_identity_drift",
                manifest.run_id,
                "same query identity/order",
                "different",
            )
        )
    if manifest.configuration.summary_sha256 != parent_manifest.configuration.summary_sha256:
        violations.append(
            _violation(
                "lineage_configuration_drift",
                manifest.run_id,
                parent_manifest.configuration.summary_sha256,
                manifest.configuration.summary_sha256,
            )
        )
    if manifest.comparison != parent_manifest.comparison:
        violations.append(
            _violation(
                "lineage_comparison_binding_drift",
                manifest.run_id,
                parent_manifest.comparison.model_dump(mode="json")
                if parent_manifest.comparison
                else None,
                manifest.comparison.model_dump(mode="json")
                if manifest.comparison
                else None,
            )
        )
    if manifest.shard != parent_manifest.shard:
        violations.append(
            _violation(
                "lineage_shard_binding_drift",
                manifest.run_id,
                parent_manifest.shard.model_dump(mode="json")
                if parent_manifest.shard
                else None,
                manifest.shard.model_dump(mode="json")
                if manifest.shard
                else None,
            )
        )
    if manifest.progress.completed_count < parent_manifest.progress.completed_count:
        violations.append(
            _violation(
                "lineage_progress_regression",
                manifest.run_id,
                f">={parent_manifest.progress.completed_count}",
                manifest.progress.completed_count,
            )
        )
    violations.extend(
        _validate_manifest_recursive(
            parent_manifest,
            manifest_path=parent_path,
            repository_root=repository_root,
            ancestors=(*ancestors, resolved),
        )
    )
    return violations


def _validate_manifest_files(
    manifest: RunManifestV1, repository_root: Path
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for item in [
        *manifest.dataset.inputs,
        manifest.queries.input,
        manifest.prompt.manifest,
        *([manifest.comparison.plan] if manifest.comparison is not None else []),
        *([manifest.shard.plan] if manifest.shard is not None else []),
        *manifest.outputs,
    ]:
        _check_file_identity(item, repository_root, violations)
    try:
        observed_queries = build_query_identity(
            manifest.queries.input.path,
            repository_root=repository_root,
            id_field=manifest.queries.id_field,
            text_field=manifest.queries.text_field,
        )
    except (OSError, RunProvenanceError) as exc:
        violations.append(
            _violation(
                "query_input_invalid",
                manifest.queries.input.path,
                "readable query-only JSONL",
                type(exc).__name__,
            )
        )
    else:
        for field in ("count", "stable_identity_sha256", "order_sha256"):
            expected = getattr(manifest.queries, field)
            observed = getattr(observed_queries, field)
            if expected != observed:
                violations.append(
                    _violation(
                        f"query_{field}_mismatch",
                        f"queries.{field}",
                        expected,
                        observed,
                    )
                )
    expected_dataset_hash = stable_hash(
        {
            "name": manifest.dataset.name,
            "version": manifest.dataset.version,
            "inputs": [item.model_dump(mode="json") for item in manifest.dataset.inputs],
        }
    )
    if expected_dataset_hash != manifest.dataset.identity_summary_sha256:
        violations.append(
            _violation(
                "dataset_identity_mismatch",
                "dataset.identity_summary_sha256",
                manifest.dataset.identity_summary_sha256,
                expected_dataset_hash,
            )
        )
    _validate_output_inventory(manifest, repository_root, violations)
    if (
        manifest.progress.status == "completed"
        and manifest.progress.completed_count != manifest.progress.expected_count
    ):
        violations.append(
            _violation(
                "completed_run_record_count_insufficient",
                "progress.completed_count",
                manifest.progress.expected_count,
                manifest.progress.completed_count,
            )
        )
    if manifest.progress.record_output_path:
        record = next(
            (
                item
                for item in manifest.outputs
                if item.path == manifest.progress.record_output_path
            ),
            None,
        )
        if record is not None and record.record_count != manifest.progress.completed_count:
            violations.append(
                _violation(
                    "progress_record_count_mismatch",
                    record.path,
                    manifest.progress.completed_count,
                    record.record_count,
                )
            )
    if manifest.resource_ledger is not None:
        ledger_path = resolve_repo_path(
            repository_root, manifest.resource_ledger.output.path
        )
        try:
            from scholar_agent.evaluation.resource_accounting import (  # noqa: PLC0415
                ResourceLedgerV1,
                opaque_resource_identity,
                validate_resource_ledger,
            )

            ledger = ResourceLedgerV1.model_validate_json(
                ledger_path.read_text(encoding="utf-8")
            )
            ledger_report = validate_resource_ledger(ledger)
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            violations.append(
                _violation(
                    "resource_ledger_unreadable",
                    manifest.resource_ledger.output.path,
                    "valid resource_ledger_v1",
                    type(exc).__name__,
                )
            )
        else:
            expected_run_identity = opaque_resource_identity("run", manifest.run_id)
            if ledger.run_identity != expected_run_identity:
                violations.append(
                    _violation(
                        "resource_ledger_run_identity_mismatch",
                        manifest.resource_ledger.output.path,
                        expected_run_identity,
                        ledger.run_identity,
                    )
                )
            if ledger.manifest_identity != manifest.resource_ledger.manifest_identity:
                violations.append(
                    _violation(
                        "resource_ledger_manifest_identity_mismatch",
                        manifest.resource_ledger.output.path,
                        manifest.resource_ledger.manifest_identity,
                        ledger.manifest_identity,
                    )
                )
            if len(ledger.expected_query_identities) != manifest.queries.count:
                violations.append(
                    _violation(
                        "resource_ledger_query_count_mismatch",
                        manifest.resource_ledger.output.path,
                        manifest.queries.count,
                        len(ledger.expected_query_identities),
                    )
                )
            if ledger_report["status"] != "passed":
                violations.append(
                    _violation(
                        "resource_ledger_accounting_invalid",
                        manifest.resource_ledger.output.path,
                        "passed",
                        ledger_report["status"],
                    )
                )
    if manifest.provider_ingest is not None:
        try:
            from scholar_agent.evaluation.provider_ingest_provenance import (  # noqa: PLC0415
                verify_capture_bundle,
            )

            ingest_report = verify_capture_bundle(
                resolve_repo_path(repository_root, manifest.provider_ingest.bundle.path),
                resolve_repo_path(
                    repository_root, manifest.provider_ingest.raw_archive.path
                ),
                resource_ledger_path=(
                    resolve_repo_path(
                        repository_root, manifest.resource_ledger.output.path
                    )
                    if manifest.resource_ledger is not None
                    else None
                ),
            )
        except (OSError, ValueError, ValidationError) as exc:
            violations.append(
                _violation(
                    "provider_ingest_unreadable",
                    manifest.provider_ingest.bundle.path,
                    "valid provider_ingest_provenance_v1",
                    type(exc).__name__,
                )
            )
        else:
            if ingest_report["status"] != "passed":
                violations.append(
                    _violation(
                        "provider_ingest_invalid",
                        manifest.provider_ingest.bundle.path,
                        "passed",
                        ingest_report["status"],
                    )
                )
    payload = manifest.model_dump(mode="json")
    for binding in manifest.metadata_bindings:
        artifact_path = resolve_repo_path(repository_root, binding.artifact_path)
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            actual = _json_pointer(artifact, binding.artifact_json_pointer)
            expected = _json_pointer(payload, binding.manifest_json_pointer)
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            violations.append(
                _violation(
                    "metadata_binding_unreadable",
                    binding.artifact_path + binding.artifact_json_pointer,
                    "readable binding",
                    type(exc).__name__,
                )
            )
            continue
        if actual != expected:
            violations.append(
                _violation(
                    "metadata_binding_mismatch",
                    binding.artifact_path + binding.artifact_json_pointer,
                    expected,
                    actual,
                )
            )
    return violations


def _validate_output_inventory(
    manifest: RunManifestV1,
    repository_root: Path,
    violations: list[dict[str, Any]],
) -> None:
    output_root = resolve_repo_path(repository_root, manifest.output_directory)
    if not output_root.is_dir():
        violations.append(
            _violation(
                "output_directory_missing",
                manifest.output_directory,
                "directory",
                None,
            )
        )
        return
    expected: set[str] = set()
    for output in manifest.outputs:
        path = resolve_repo_path(repository_root, output.path)
        try:
            expected.add(path.relative_to(output_root).as_posix())
        except ValueError:
            violations.append(
                _violation(
                    "output_outside_directory",
                    output.path,
                    manifest.output_directory,
                    output.path,
                )
            )
    excluded = set(manifest.output_inventory_excludes)
    actual = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.relative_to(output_root).as_posix() not in excluded
    }
    for value in sorted(actual - expected):
        violations.append(
            _violation(
                "unregistered_output_file",
                f"{manifest.output_directory}/{value}",
                "registered or excluded",
                "unregistered",
            )
        )
    for value in sorted(expected - actual):
        violations.append(
            _violation(
                "registered_output_missing",
                f"{manifest.output_directory}/{value}",
                "existing",
                None,
            )
        )


def _check_file_identity(
    item: FileIdentity,
    repository_root: Path,
    violations: list[dict[str, Any]],
) -> None:
    path = resolve_repo_path(repository_root, item.path)
    if not path.is_file():
        violations.append(
            _violation(
                "file_missing",
                item.path,
                {"size_bytes": item.size_bytes, "sha256": item.sha256},
                None,
            )
        )
        return
    size = path.stat().st_size
    if size != item.size_bytes:
        violations.append(
            _violation("file_size_mismatch", item.path, item.size_bytes, size)
        )
    digest = sha256_file(path)
    if digest != item.sha256:
        violations.append(_violation("file_tampered", item.path, item.sha256, digest))


def _load_manifest(
    path: Path, violations: list[dict[str, Any]]
) -> RunManifestV1 | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        violations.append(
            _violation(
                "manifest_unreadable",
                _display_path(path, path.parent),
                "valid JSON",
                type(exc).__name__,
            )
        )
        return None
    if not isinstance(payload, dict):
        violations.append(
            _violation("manifest_schema_error", "$", "JSON object", type(payload).__name__)
        )
        return None
    if (
        payload.get("manifest_kind") != RUN_MANIFEST_KIND
        or payload.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
    ):
        violations.append(
            _violation(
                "schema_version_incompatible",
                "$",
                {
                    "manifest_kind": RUN_MANIFEST_KIND,
                    "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                },
                {
                    "manifest_kind": payload.get("manifest_kind"),
                    "schema_version": payload.get("schema_version"),
                },
            )
        )
        return None
    try:
        return RunManifestV1.model_validate(payload)
    except ValidationError as exc:
        for error in exc.errors(
            include_url=False, include_context=False, include_input=False
        ):
            violations.append(
                _violation(
                    "manifest_schema_error",
                    ".".join(str(value) for value in error["loc"]),
                    "valid field",
                    error["type"],
                )
            )
        return None


def _json_pointer(payload: Any, pointer: str) -> Any:
    current = payload
    for part in pointer.lstrip("/").split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        current = current[int(key)] if isinstance(current, list) else current[key]
    return current


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise RunProvenanceError(f"spec section missing:{name}")
    return value


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ValueError(f"path must be repository-relative:{value}")


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _violation(kind: str, path: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {"kind": kind, "path": path, "expected": expected, "observed": observed}


def _validation_report(
    *, status: ReportStatus, violations: Sequence[Mapping[str, Any]], network_count: int
) -> dict[str, Any]:
    ordered = sorted(
        (dict(item) for item in violations),
        key=lambda item: (str(item["kind"]), str(item["path"]), json.dumps(item, sort_keys=True)),
    )
    return {
        "schema_version": "1",
        "gate": RUN_PROVENANCE_GATE,
        "status": status,
        "exit_code": EXIT_OK if status == "passed" else EXIT_INTEGRITY_FAILURE,
        "violation_count": len(ordered),
        "violations": ordered,
        "execution": {
            "network_request_count": network_count,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_fields_accessed": False,
        },
    }


def _legacy_report(
    status: ReportStatus,
    violations: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    report = _validation_report(
        status=(
            "invalid" if status == "invalid" else "legacy_metadata_incomplete"
        ),
        violations=violations,
        network_count=0,
    )
    report["status"] = status
    report["exit_code"] = (
        EXIT_INTEGRITY_FAILURE
        if status == "invalid"
        else EXIT_LEGACY_METADATA_INCOMPLETE
    )
    report["profiles"] = list(profiles)
    return report


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise RunProvenanceError("network access forbidden")

    with patch.object(socket, "create_connection", blocked), patch.object(
        socket.socket, "connect", blocked
    ):
        yield
