"""Synthetic-only integration rehearsal for the formal validation lifecycle.

The coordinator deliberately reuses the production offline gates.  It stores
only hashes and structural counts from synthetic fixtures; labels, scorer
values, receipts, archives, and run directories remain in temporary storage.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from scholar_agent.evaluation.evidence_revocation import (
    audit_current as audit_revocation_current,
    load_current as load_revocation_current,
    simulate_incidents,
)
from scholar_agent.evaluation.external_scorer_handoff import (
    audit_real_readiness as audit_scorer_readiness,
    canonical_handoff,
    create_package_manifest,
    load_protocol as load_scorer_protocol,
    run_scorer,
    stable_json_bytes,
    synthetic_scorer_source,
)
from scholar_agent.evaluation.formal_evidence_quarantine import (
    FormalEvidenceIntakeManifestV1,
    audit_contamination,
    build_intake_manifest,
    consume_for_evaluation,
    load_protocol as load_quarantine_protocol,
    current_readiness as quarantine_current_readiness,
)
from scholar_agent.evaluation.formal_run_disaster_recovery import (
    load_protocol as load_recovery_protocol,
    simulate_disaster,
)
from scholar_agent.evaluation.formal_validation_clearance import (
    build_current_evidence,
    conformance_evidence,
    evaluate as evaluate_clearance,
    issue_receipt,
    load_protocol as load_clearance_protocol,
    verify_receipt,
)
from scholar_agent.evaluation.formal_validation_preregistration import (
    audit_readiness as audit_preregistration_readiness,
    evaluate_amendment,
    load_protocol as load_preregistration_protocol,
    read_json as read_preregistration_json,
    verify_seal,
)
from scholar_agent.evaluation.full1000_execution_readiness import (
    dry_run as full1000_dry_run,
    verify_plan,
)
from scholar_agent.evaluation.full1000_launch_control import (
    load_protocol as load_launch_protocol,
    simulate_operations,
)
from scholar_agent.evaluation.human_annotation_delivery import (
    load_delivery_protocol,
    readiness as human_delivery_readiness,
    synthetic_dry_run as human_synthetic_dry_run,
)
from scholar_agent.evaluation.provider_ingest_provenance import (
    deterministic_fixture_matrix as provider_fixture_matrix,
)
from scholar_agent.evaluation.validation_evidence_freshness import (
    load_contract as load_freshness_contract,
    verify_current as verify_freshness_current,
)


PROTOCOL = "formal_validation_dress_rehearsal_v1"
SCHEMA_VERSION = "1"
EXIT_COMPLETED = 0
EXIT_VIOLATION = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 4
SOURCE_COMMIT = "a3ba36422b69294cd70bde83b102bd576a2c4d1d"
SYNTHETIC_PREFIX = "synthetic_rehearsal_only:"
TEMP_PREFIX = "synthetic_rehearsal_only-"
REAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
STAGE_ORDER = (
    "preregistration_sealed",
    "launch_authorized",
    "full1000_executed",
    "provider_provenance_verified",
    "disaster_recovery_verified",
    "evidence_intake_locked",
    "unblind_and_scoring_verified",
    "quarantine_audited",
    "freshness_verified",
    "eligible_for_clearance",
    "test_receipt_issued",
    "standalone_bundle_verified",
)
FAILURE_SCENARIOS = (
    "missing_stage",
    "reordered_stage",
    "cross_commit_mix",
    "cross_protocol_mix",
    "duplicate_intake",
    "duplicate_receipt",
    "old_attempt",
    "partial_human_labels",
    "partial_scorer_output",
    "posthoc_prompt_change",
    "posthoc_threshold_change",
    "posthoc_sample_change",
    "posthoc_statistics_change",
    "posthoc_default_policy_change",
    "revoked_upstream_evidence",
    "synthetic_real_state_pollution",
)
EXECUTION = {
    "academic_api_request_count": 0,
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
REQUIRED_BINDINGS = {
    "clearance_protocol": (
        "benchmark/formal_validation_clearance_v1_protocol.json",
        "formal_validation_clearance_v1",
    ),
    "external_scorer_protocol": (
        "benchmark/external_scorer_handoff_v1_protocol.json",
        "external_scorer_handoff_v1",
    ),
    "freshness_spec": (
        "benchmark/validation_evidence_freshness_v1_spec.json",
        "validation_evidence_freshness_v1",
    ),
    "full1000_execution_plan": (
        "benchmark/full1000_execution_plan_v1.json",
        "full1000_execution_plan_v1",
    ),
    "full1000_execution_protocol": (
        "benchmark/full1000_execution_readiness_v1_protocol.json",
        "full1000_execution_readiness_v1",
    ),
    "human_adjudication_protocol": (
        "benchmark/human_precision_adjudication_v1_protocol.json",
        "human_precision_adjudication_v1",
    ),
    "human_delivery_protocol": (
        "benchmark/human_annotation_delivery_v1_protocol.json",
        "human_annotation_delivery_v1",
    ),
    "launch_protocol": (
        "benchmark/full1000_launch_control_v1_protocol.json",
        "full1000_launch_control_v1",
    ),
    "preregistration_protocol": (
        "benchmark/formal_validation_preregistration_v1_protocol.json",
        "formal_validation_preregistration_v1",
    ),
    "preregistration_seal": (
        "benchmark/formal_validation_preregistration_v1_seal.json",
        "formal_validation_preregistration_seal_v1",
    ),
    "provider_provenance_protocol": (
        "benchmark/provider_ingest_provenance_v1_protocol.json",
        "provider_ingest_provenance_v1",
    ),
    "quarantine_protocol": (
        "benchmark/formal_evidence_quarantine_v1_protocol.json",
        "formal_evidence_quarantine_v1",
    ),
    "recovery_protocol": (
        "benchmark/formal_run_disaster_recovery_v1_protocol.json",
        "formal_run_disaster_recovery_v1",
    ),
    "revocation_protocol": (
        "benchmark/evidence_revocation_response_v1_protocol.json",
        "evidence_revocation_response_v1",
    ),
    "standalone_contract": (
        "benchmark/standalone_auditor_bundle_v1_contract.json",
        "standalone_auditor_bundle_v1",
    ),
}
_HEX = frozenset("0123456789abcdef")
_REPORT_KEYS = {
    "cleanup",
    "execution",
    "exit_code",
    "formal_validation_complete",
    "handoff_checklist_sha256",
    "human_item_count",
    "namespace",
    "protocol",
    "protocol_sha256",
    "query_count",
    "real_blockers",
    "real_state_mutation_count",
    "rehearsal_id",
    "report_sha256",
    "schema_version",
    "scorer_query_count",
    "shard_count",
    "source_commit",
    "stage_count",
    "stages",
    "status",
    "synthetic_rehearsal_only",
    "test_receipt",
}


class DressRehearsalError(RuntimeError):
    """An integration, ordering, isolation, or evidence invariant failed."""


class DressRehearsalBlocked(DressRehearsalError):
    """Controls are ready while real external evidence remains unavailable."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in rows:
        if key in value:
            raise DressRehearsalError("duplicate_json_key")
        value[key] = child
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                DressRehearsalError("nonfinite_json_number")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DressRehearsalError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise DressRehearsalError("json_root_not_object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(canonical_json(value))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise DressRehearsalError("json_output_unavailable") from exc


def _safe_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
        or value.parts[0] == "third_party"
        or value.name == ".env"
    ):
        raise DressRehearsalError("unsafe_registered_path")
    path = (root / Path(*value.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DressRehearsalError("registered_path_escape") from exc
    return path


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=20,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        },
    )
    if completed.returncode != 0:
        raise DressRehearsalError("registered_file_not_present_at_source_commit")
    return hashlib.sha256(completed.stdout).hexdigest()


def _protocol_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["protocol_sha256"] = "0" * 64
    return payload


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = read_json(path)
    required = {
        "allowed_outputs",
        "bindings",
        "execution",
        "failure_scenarios",
        "formal_validation_complete",
        "handoff_steps",
        "human",
        "namespace",
        "population",
        "protocol",
        "protocol_sha256",
        "real_blockers",
        "schema_version",
        "scorer",
        "source_commit",
        "stage_order",
    }
    if set(value) != required:
        raise DressRehearsalError("protocol_schema_invalid")
    if (
        value.get("protocol") != PROTOCOL
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("source_commit") != SOURCE_COMMIT
        or value.get("formal_validation_complete") is not False
        or value.get("execution") != EXECUTION
        or value.get("stage_order") != list(STAGE_ORDER)
        or value.get("failure_scenarios") != list(FAILURE_SCENARIOS)
        or value.get("real_blockers") != list(REAL_BLOCKERS)
    ):
        raise DressRehearsalError("protocol_policy_drift")
    if value.get("namespace") != {
        "identity_prefix": SYNTHETIC_PREFIX,
        "persist_labels": False,
        "persist_receipts": False,
        "persist_run_artifacts": False,
        "temporary_directory_prefix": TEMP_PREFIX,
    }:
        raise DressRehearsalError("synthetic_namespace_policy_drift")
    if value.get("population") != {
        "query_count": 1000,
        "shard_count": 20,
        "start_mode": "full_restart_all_1000",
    }:
        raise DressRehearsalError("population_policy_drift")
    if value.get("human") != {
        "annotator_count": 2,
        "item_count": 471,
        "statistics_persisted": False,
    }:
        raise DressRehearsalError("human_policy_drift")
    if value.get("scorer") != {
        "metric_namespace": "synthetic_handoff",
        "metric_values_persisted": False,
        "official_schema_inferred": False,
        "package_type": "strict_conformance_fixture_not_official_scorer",
        "query_count": 1000,
    }:
        raise DressRehearsalError("scorer_policy_drift")
    if not isinstance(value.get("handoff_steps"), list) or not value["handoff_steps"]:
        raise DressRehearsalError("handoff_steps_invalid")
    if not isinstance(value.get("allowed_outputs"), list) or sorted(
        set(value["allowed_outputs"])
    ) != value["allowed_outputs"]:
        raise DressRehearsalError("allowed_outputs_invalid")
    bindings = value.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != len(REQUIRED_BINDINGS):
        raise DressRehearsalError("binding_inventory_invalid")
    observed_roles: set[str] = set()
    for row in bindings:
        if not isinstance(row, Mapping) or set(row) != {
            "contract",
            "path",
            "role",
            "sha256",
        }:
            raise DressRehearsalError("binding_entry_invalid")
        role = str(row["role"])
        if role in observed_roles or role not in REQUIRED_BINDINGS:
            raise DressRehearsalError("binding_role_invalid")
        observed_roles.add(role)
        expected_path, expected_contract = REQUIRED_BINDINGS[role]
        if (
            row["path"] != expected_path
            or row["contract"] != expected_contract
            or not _is_digest(row["sha256"])
        ):
            raise DressRehearsalError("binding_contract_drift")
        bound_path = _safe_path(repository_root, expected_path)
        if (
            not bound_path.is_file()
            or sha256_file(bound_path) != row["sha256"]
            or _git_blob_sha256(repository_root, SOURCE_COMMIT, expected_path)
            != row["sha256"]
        ):
            raise DressRehearsalError("binding_hash_or_source_commit_drift")
    if observed_roles != set(REQUIRED_BINDINGS):
        raise DressRehearsalError("binding_inventory_invalid")
    if not _is_digest(value.get("protocol_sha256")) or stable_hash(
        _protocol_payload(value)
    ) != value["protocol_sha256"]:
        raise DressRehearsalError("protocol_digest_invalid")
    return value


def _binding_path(
    protocol: Mapping[str, Any], repository_root: Path, role: str
) -> Path:
    row = next(item for item in protocol["bindings"] if item["role"] == role)
    return _safe_path(repository_root, str(row["path"]))


class RehearsalMachine:
    """Strict ordered state and uniqueness guard for synthetic orchestration."""

    def __init__(self, protocol: Mapping[str, Any]) -> None:
        self.protocol = protocol
        self.stage_index = 0
        self.stages: list[dict[str, Any]] = []
        self.intake_ids: set[str] = set()
        self.receipt_issued = False

    def advance(self, stage: str, summary: Mapping[str, Any]) -> None:
        if self.stage_index >= len(STAGE_ORDER) or stage != STAGE_ORDER[self.stage_index]:
            raise DressRehearsalError("stage_order_or_presence_violation")
        _assert_no_sensitive_or_quality_payload(summary)
        self.stages.append(
            {
                "index": self.stage_index,
                "name": stage,
                "status": "passed",
                "summary": dict(summary),
                "summary_sha256": stable_hash(summary),
            }
        )
        self.stage_index += 1

    def register_intake(self, manifest: FormalEvidenceIntakeManifestV1) -> str:
        if not manifest.synthetic_only:
            raise DressRehearsalError("non_synthetic_intake_forbidden")
        identity = SYNTHETIC_PREFIX + manifest.intake_id
        if identity in self.intake_ids:
            raise DressRehearsalError("duplicate_intake")
        self.intake_ids.add(identity)
        return identity

    def register_receipt(self, receipt: Mapping[str, Any]) -> str:
        if self.receipt_issued:
            raise DressRehearsalError("duplicate_receipt")
        if (
            receipt.get("synthetic_test_only") is not True
            or receipt.get("formal_validation_complete") is not False
        ):
            raise DressRehearsalError("real_clearance_from_rehearsal_forbidden")
        self.receipt_issued = True
        return SYNTHETIC_PREFIX + str(receipt["receipt_sha256"])

    def finish(self) -> None:
        if self.stage_index != len(STAGE_ORDER):
            raise DressRehearsalError("stage_order_or_presence_violation")


def _assert_no_sensitive_or_quality_payload(value: Any, path: str = "$") -> None:
    forbidden_keys = {
        "case_id",
        "gold",
        "labels",
        "metric_values",
        "precision",
        "qrels",
        "query",
        "query_identity",
        "query_text",
        "recall",
        "score_values",
        "target_paper",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden_keys:
                raise DressRehearsalError("forbidden_payload_field")
            _assert_no_sensitive_or_quality_payload(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_sensitive_or_quality_payload(child, f"{path}/{index}")
    elif isinstance(value, str):
        lowered = value.lower()
        if (
            value.startswith("/")
            or ".env" in lowered
            or "api_key" in lowered
            or "authorization:" in lowered
        ):
            raise DressRehearsalError("sensitive_or_absolute_value")


def _stage_summary(**values: Any) -> dict[str, Any]:
    return dict(sorted(values.items()))


def _build_intake(
    *,
    artifact: Path,
    evidence_root: Path,
    evidence_type: str,
    evidence_protocol_version: str,
    quarantine_protocol: Mapping[str, Any],
    plan_sha256: str,
) -> FormalEvidenceIntakeManifestV1:
    digest = hashlib.sha256(
        f"{evidence_type}:{artifact.name}".encode("utf-8")
    ).hexdigest()
    return build_intake_manifest(
        evidence_path=artifact,
        evidence_root=evidence_root,
        evidence_type=evidence_type,
        evidence_protocol_version=evidence_protocol_version,
        input_binding={
            "contract": "comparison_plan_v1",
            "plan_sha256": plan_sha256,
            "query_order_sha256": digest,
            "run_manifest_sha256": digest,
        },
        chronology={
            "preregistration_commit": "1" * 40,
            "execution_commit": "2" * 40,
            "intake_commit": "3" * 40,
            "report_code_commit": "4" * 40,
            "proof": "synthetic_fixture_only",
        },
        protocol=quarantine_protocol,
        synthetic_only=True,
    )


def _scorer_fixture(
    *,
    root: Path,
    repository_root: Path,
    query_identities: Sequence[str],
    scorer_protocol: Mapping[str, Any],
    quarantine_protocol: Mapping[str, Any],
    plan_sha256: str,
    machine: RehearsalMachine,
) -> tuple[dict[str, Any], list[tuple[FormalEvidenceIntakeManifestV1, Path]]]:
    package = root / "scorer-package"
    queries = [
        {
            "query_identity": identity,
            "query_order": index,
            "results": [
                {
                    "authority_digest": stable_hash(
                        {"authority": identity, "rank": 1}
                    ),
                    "rank": 1,
                    "result_identity": stable_hash(
                        {"result": identity, "rank": 1}
                    ),
                }
            ],
        }
        for index, identity in enumerate(query_identities)
    ]
    handoff = canonical_handoff(
        queries,
        run_manifest_sha256=stable_hash({"synthetic": "run-manifest"}),
        commit_generation_sha256=stable_hash({"synthetic": "generation"}),
        source_scope="synthetic_rehearsal_only",
    )
    handoff_path = root / "canonical-handoff.json"
    handoff_path.write_bytes(stable_json_bytes(handoff))
    first = run_scorer(
        package,
        handoff_path,
        scorer_protocol,
        repository_root=repository_root,
        run_ordinal=1,
    )
    second = run_scorer(
        package,
        handoff_path,
        scorer_protocol,
        repository_root=repository_root,
        run_ordinal=2,
    )
    if first["output_bytes"] != second["output_bytes"]:
        raise DressRehearsalError("synthetic_scorer_not_deterministic")
    output_path = root / "scorer-output.json"
    output_path.write_bytes(first["output_bytes"])
    output_manifest = _build_intake(
        artifact=output_path,
        evidence_root=root,
        evidence_type="official_scorer_output",
        evidence_protocol_version="synthetic_scorer_output_v1",
        quarantine_protocol=quarantine_protocol,
        plan_sha256=plan_sha256,
    )
    machine.register_intake(output_manifest)
    consumed = consume_for_evaluation(
        output_manifest,
        evidence_root=root,
        consumer="scholar_agent.evaluation.formal_validation_dress_rehearsal",
        purpose="evaluation",
        protocol=quarantine_protocol,
    )
    if hashlib.sha256(consumed).hexdigest() != output_manifest.artifact.sha256:
        raise DressRehearsalError("scorer_output_intake_consumption_drift")
    return (
        {
            "handoff_query_count": len(handoff["queries"]),
            "handoff_sha256": handoff["content_sha256"],
            "output_query_count": len(first["output"]["query_results"]),
            "output_sha256": first["output_sha256"],
            "repeat_output_byte_identical": True,
            "synthetic_metric_values_persisted": False,
        },
        [(output_manifest, root)],
    )


def _standalone_fixture(repository_root: Path, temporary_root: Path) -> dict[str, Any]:
    archive = temporary_root / "standalone-auditor.zip"
    script = repository_root / "scripts/check_standalone_auditor_bundle.py"
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    build = subprocess.run(
        [sys.executable, str(script), "build", "--output", str(archive)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        check=False,
        timeout=60,
    )
    verify = subprocess.run(
        [sys.executable, str(script), "verify", str(archive)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if build.returncode != 0 or verify.returncode != 0 or build.stderr or verify.stderr:
        raise DressRehearsalError("standalone_bundle_gate_failed")
    build_report = json.loads(build.stdout)
    verify_report = json.loads(verify.stdout)
    if (
        build_report.get("archive_sha256") != verify_report.get("archive_sha256")
        or verify_report.get("formal_validation_complete") is not False
        or verify_report.get("blocker_count") != 3
    ):
        raise DressRehearsalError("standalone_bundle_boundary_drift")
    return {
        "archive_sha256": verify_report["archive_sha256"],
        "blocker_count": verify_report["blocker_count"],
        "formal_validation_complete": False,
        "status": verify_report["status"],
    }


def build_handoff_checklist(protocol: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for index, row in enumerate(protocol["handoff_steps"]):
        if not isinstance(row, Mapping) or set(row) != {
            "command",
            "expected_exit_codes",
            "failure_rollback",
            "human_checkpoint",
            "input_prerequisites",
            "step",
        }:
            raise DressRehearsalError("handoff_step_schema_invalid")
        command = str(row["command"])
        if (
            command.startswith("/")
            or ".env" in command
            or "official_schema" in command.lower()
        ):
            raise DressRehearsalError("handoff_command_unsafe_or_speculative")
        rows.append({"order": index + 1, **dict(row)})
    checklist = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "synthetic_rehearsal_handoff_ready",
        "synthetic_rehearsal_only": True,
        "formal_validation_complete": False,
        "steps": rows,
        "real_blockers": list(REAL_BLOCKERS),
        "execution": EXECUTION,
    }
    checklist["checklist_sha256"] = stable_hash(checklist)
    return checklist


def run_rehearsal(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    root = repository_root.resolve()
    machine = RehearsalMachine(protocol)
    checklist = build_handoff_checklist(protocol)
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as temporary:
        temporary_path = Path(temporary)
        if not temporary_path.name.startswith(TEMP_PREFIX):
            raise DressRehearsalError("synthetic_namespace_directory_invalid")

        preregistration = load_preregistration_protocol(
            _binding_path(protocol, root, "preregistration_protocol")
        )
        seal = read_preregistration_json(
            _binding_path(protocol, root, "preregistration_seal")
        )
        prereg_report = verify_seal(seal, preregistration, repository_root=root)
        machine.advance(
            "preregistration_sealed",
            _stage_summary(
                registered_file_count=prereg_report["registered_file_count"],
                seal_sha256=prereg_report["seal_sha256"],
            ),
        )

        launch_protocol = load_launch_protocol(
            _binding_path(protocol, root, "launch_protocol")
        )
        launch = simulate_operations(root, launch_protocol)
        if (
            launch.get("exit_code") != 0
            or launch.get("query_count") != 1000
            or launch.get("shard_count") != 20
            or launch.get("violations")
        ):
            raise DressRehearsalError("launch_simulation_failed")
        machine.advance(
            "launch_authorized",
            _stage_summary(
                audit_sha256=launch["audit_sha256"],
                operation_count=launch["operation_count"],
                scenario_count=launch["scenario_count"],
            ),
        )

        execution_protocol = read_json(
            _binding_path(protocol, root, "full1000_execution_protocol")
        )
        plan = read_json(_binding_path(protocol, root, "full1000_execution_plan"))
        plan_verification = verify_plan(root, execution_protocol, plan)
        execution = full1000_dry_run(plan)
        if (
            plan_verification.get("exit_code") != 0
            or execution.get("exit_code") != 0
            or execution.get("query_count") != 1000
            or execution.get("shard_count") != 20
            or execution.get("violations")
        ):
            raise DressRehearsalError("full1000_fixture_failed")
        machine.advance(
            "full1000_executed",
            _stage_summary(
                aggregate_sha256=execution["aggregate_sha256"],
                capsule_verified=execution["stages"]["reproduction_capsule"],
                checkpoint_resume_verified=execution["stages"][
                    "resume_supersession"
                ],
                query_count=1000,
                resource_ledger_verified=execution["stages"][
                    "resource_ledger_conformance"
                ],
                shard_count=20,
                top20_delivery_verified=execution["stages"][
                    "top20_delivery_contract"
                ],
            ),
        )

        provider = provider_fixture_matrix(temporary_path / "provider")
        if provider.get("exit_code") != 0:
            raise DressRehearsalError("provider_provenance_fixture_failed")
        machine.advance(
            "provider_provenance_verified",
            _stage_summary(
                scenario_count=provider["scenario_count"],
                source_count=len(
                    {row["source"] for row in provider["scenarios"]}
                ),
                status=provider["status"],
            ),
        )

        recovery_protocol = load_recovery_protocol(
            _binding_path(protocol, root, "recovery_protocol"),
            repository_root=root,
        )
        recovery = simulate_disaster(root, recovery_protocol)
        if (
            recovery.get("exit_code") != 0
            or recovery.get("query_count") != 1000
            or recovery.get("duplicate_request_count") != 0
            or not all(recovery.get("equivalence", {}).values())
        ):
            raise DressRehearsalError("disaster_recovery_fixture_failed")
        machine.advance(
            "disaster_recovery_verified",
            _stage_summary(
                duplicate_request_count=0,
                final_request_count=recovery["final_request_count"],
                parent_chain_length=recovery["parent_chain_length"],
                replacement_shard=recovery["replacement_shard"],
                scenario_count=recovery["scenario_count"],
            ),
        )

        quarantine_protocol = load_quarantine_protocol(
            _binding_path(protocol, root, "quarantine_protocol")
        )
        scorer_protocol = load_scorer_protocol(
            _binding_path(protocol, root, "external_scorer_protocol"),
            repository_root=root,
        )
        scorer_package = temporary_path / "scorer-package"
        create_package_manifest(
            scorer_package,
            scorer_name="synthetic-strict-scorer",
            scorer_version="1",
            entrypoint_source=synthetic_scorer_source("valid"),
        )
        package_manifest = _build_intake(
            artifact=scorer_package / "manifest.json",
            evidence_root=temporary_path,
            evidence_type="official_scorer_package",
            evidence_protocol_version="external_scorer_handoff_v1",
            quarantine_protocol=quarantine_protocol,
            plan_sha256=plan["plan_sha256"],
        )
        machine.register_intake(package_manifest)
        package_bytes = consume_for_evaluation(
            package_manifest,
            evidence_root=temporary_path,
            consumer="scholar_agent.evaluation.formal_validation_dress_rehearsal",
            purpose="evaluation",
            protocol=quarantine_protocol,
        )
        if hashlib.sha256(package_bytes).hexdigest() != package_manifest.artifact.sha256:
            raise DressRehearsalError("scorer_package_intake_consumption_drift")
        intake_records: list[tuple[FormalEvidenceIntakeManifestV1, Path]] = [
            (package_manifest, temporary_path)
        ]
        human_adjudication_manifest: FormalEvidenceIntakeManifestV1 | None = None

        def locked_human_intake(base: Path, annotator_a: Path, annotator_b: Path) -> None:
            for path in (annotator_a, annotator_b):
                manifest = _build_intake(
                    artifact=path,
                    evidence_root=base,
                    evidence_type="human_annotation_labels",
                    evidence_protocol_version="human_annotation_delivery_v1",
                    quarantine_protocol=quarantine_protocol,
                    plan_sha256=plan["plan_sha256"],
                )
                machine.register_intake(manifest)
                consumed = consume_for_evaluation(
                    manifest,
                    evidence_root=base,
                    consumer=(
                        "scholar_agent.evaluation."
                        "formal_validation_dress_rehearsal"
                    ),
                    purpose="evaluation",
                    protocol=quarantine_protocol,
                )
                intake_records.append((manifest, base))
                if hashlib.sha256(consumed).hexdigest() != manifest.artifact.sha256:
                    raise DressRehearsalError("human_intake_consumption_drift")
            machine.advance(
                "evidence_intake_locked",
                _stage_summary(
                    intake_count=len(machine.intake_ids),
                    package_count=1,
                    submission_count=2,
                    unique_intake_count=len(machine.intake_ids),
                ),
            )

        def adjudication_intake(
            base: Path, human_gate: Mapping[str, Any]
        ) -> None:
            nonlocal human_adjudication_manifest
            if human_gate.get("state") != "validated":
                raise DressRehearsalError("human_adjudication_not_validated")
            human_adjudication_manifest = _build_intake(
                artifact=base / "adjudication.json",
                evidence_root=base,
                evidence_type="human_adjudication_result",
                evidence_protocol_version="human_precision_adjudication_v1",
                quarantine_protocol=quarantine_protocol,
                plan_sha256=plan["plan_sha256"],
            )
            machine.register_intake(human_adjudication_manifest)
            consumed = consume_for_evaluation(
                human_adjudication_manifest,
                evidence_root=base,
                consumer=(
                    "scholar_agent.evaluation.formal_validation_dress_rehearsal"
                ),
                purpose="evaluation",
                protocol=quarantine_protocol,
            )
            if (
                hashlib.sha256(consumed).hexdigest()
                != human_adjudication_manifest.artifact.sha256
            ):
                raise DressRehearsalError("adjudication_intake_consumption_drift")
            intake_records.append((human_adjudication_manifest, base))

        human_protocol = load_delivery_protocol(
            _binding_path(protocol, root, "human_delivery_protocol"), root
        )
        human = human_synthetic_dry_run(
            human_protocol,
            repository_root=root,
            locked_submission_callback=locked_human_intake,
            adjudication_callback=adjudication_intake,
        )
        if (
            human.get("synthetic_item_count") != 471
            or human.get("synthetic_gate_state") != "validated"
            or human.get("statistics") is not None
        ):
            raise DressRehearsalError("human_fixture_failed")

        scorer_summary, scorer_intakes = _scorer_fixture(
            root=temporary_path,
            repository_root=root,
            query_identities=plan["population"]["identities"],
            scorer_protocol=scorer_protocol,
            quarantine_protocol=quarantine_protocol,
            plan_sha256=plan["plan_sha256"],
            machine=machine,
        )
        intake_records.extend(scorer_intakes)
        machine.advance(
            "unblind_and_scoring_verified",
            _stage_summary(
                adjudication_validated=True,
                annotator_count=2,
                human_item_count=471,
                scorer_output_sha256=scorer_summary["output_sha256"],
                scorer_query_count=scorer_summary["output_query_count"],
                scorer_repeat_byte_identical=True,
                statistics_persisted=False,
                synthetic_metric_values_persisted=False,
            ),
        )

        consumed_count = len(intake_records)
        if human_adjudication_manifest is None:
            raise DressRehearsalError("adjudication_intake_missing")
        if len(machine.intake_ids) != 5 or consumed_count != 5:
            raise DressRehearsalError("intake_coverage_invalid")
        machine.advance(
            "quarantine_audited",
            _stage_summary(
                intake_count=len(machine.intake_ids),
                intake_ids_sha256=stable_hash(sorted(machine.intake_ids)),
                synthetic_only=True,
                verified_consumption_count=consumed_count,
            ),
        )

        freshness_contract = load_freshness_contract(
            root / "benchmark/validation_evidence_freshness_v1_contract.json",
            repository_root=root,
        )
        freshness = verify_freshness_current(
            freshness_contract, repository_root=root
        )
        if (
            freshness.get("status") != "fresh_with_declared_blockers"
            or freshness.get("state_counts", {}).get("stale") != 0
        ):
            raise DressRehearsalError("freshness_gate_failed")
        machine.advance(
            "freshness_verified",
            _stage_summary(
                stale_count=0,
                status="fresh_with_declared_blockers",
            ),
        )

        clearance_protocol = load_clearance_protocol(
            _binding_path(protocol, root, "clearance_protocol")
        )
        clearance_evidence = conformance_evidence()
        clearance = evaluate_clearance(clearance_evidence)
        if (
            clearance.get("status") != "eligible_for_clearance"
            or clearance.get("synthetic_test_only") is not True
        ):
            raise DressRehearsalError("synthetic_clearance_not_eligible")
        machine.advance(
            "eligible_for_clearance",
            _stage_summary(
                blocker_count=3,
                evidence_sha256=clearance["evidence_sha256"],
                synthetic_test_only=True,
            ),
        )
        receipt = issue_receipt(clearance_evidence, clearance_protocol)
        receipt_identity = machine.register_receipt(receipt)
        receipt_verification = verify_receipt(
            receipt, clearance_evidence, clearance_protocol
        )
        if (
            receipt_verification.get("exit_code") != 0
            or receipt_verification.get("synthetic_test_only") is not True
            or receipt_verification.get("formal_validation_complete") is not False
        ):
            raise DressRehearsalError("synthetic_receipt_verification_failed")
        machine.advance(
            "test_receipt_issued",
            _stage_summary(
                formal_validation_complete=False,
                receipt_identity_sha256=stable_hash(receipt_identity),
                synthetic_test_only=True,
            ),
        )

        standalone = _standalone_fixture(root, temporary_path)
        machine.advance(
            "standalone_bundle_verified",
            _stage_summary(
                blocker_count=standalone["blocker_count"],
                formal_validation_complete=False,
                status=standalone["status"],
            ),
        )
        machine.finish()

    cleanup = {
        "temporary_namespace_cleaned": temporary_path is not None
        and not temporary_path.exists(),
        "labels_persisted": False,
        "receipt_persisted": False,
        "run_artifacts_persisted": False,
    }
    if not cleanup["temporary_namespace_cleaned"]:
        raise DressRehearsalError("synthetic_temporary_cleanup_failed")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "rehearsal_completed",
        "exit_code": EXIT_COMPLETED,
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "namespace": SYNTHETIC_PREFIX,
        "rehearsal_id": SYNTHETIC_PREFIX
        + stable_hash(
            {
                "protocol": protocol["protocol_sha256"],
                "stages": machine.stages,
            }
        ),
        "synthetic_rehearsal_only": True,
        "query_count": 1000,
        "shard_count": 20,
        "human_item_count": 471,
        "scorer_query_count": 1000,
        "stage_count": len(machine.stages),
        "stages": machine.stages,
        "test_receipt": {
            "synthetic_test_only": True,
            "formal_validation_complete": False,
            "receipt_identity_sha256": stable_hash(receipt_identity),
        },
        "handoff_checklist_sha256": checklist["checklist_sha256"],
        "real_blockers": list(REAL_BLOCKERS),
        "real_state_mutation_count": 0,
        "cleanup": cleanup,
        "formal_validation_complete": False,
        "execution": EXECUTION,
        "report_sha256": "0" * 64,
    }
    report["report_sha256"] = stable_hash(report)
    verify_rehearsal_report(report, protocol)
    return report


def verify_rehearsal_report(
    report: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if set(report) != _REPORT_KEYS:
        raise DressRehearsalError("rehearsal_report_schema_invalid")
    claimed = report.get("report_sha256")
    content = dict(report)
    content["report_sha256"] = "0" * 64
    if not _is_digest(claimed) or stable_hash(content) != claimed:
        raise DressRehearsalError("rehearsal_report_hash_mismatch")
    if (
        report.get("protocol") != PROTOCOL
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("source_commit") != SOURCE_COMMIT
        or report.get("protocol_sha256") != protocol["protocol_sha256"]
        or report.get("status") != "rehearsal_completed"
        or report.get("exit_code") != EXIT_COMPLETED
        or report.get("synthetic_rehearsal_only") is not True
        or report.get("formal_validation_complete") is not False
        or report.get("namespace") != SYNTHETIC_PREFIX
        or not str(report.get("rehearsal_id") or "").startswith(SYNTHETIC_PREFIX)
        or report.get("real_blockers") != list(REAL_BLOCKERS)
        or report.get("real_state_mutation_count") != 0
    ):
        raise DressRehearsalError("rehearsal_boundary_invalid")
    if (
        report.get("query_count") != 1000
        or report.get("shard_count") != 20
        or report.get("human_item_count") != 471
        or report.get("scorer_query_count") != 1000
    ):
        raise DressRehearsalError("rehearsal_population_invalid")
    stages = report.get("stages")
    if not isinstance(stages, list) or report.get("stage_count") != len(STAGE_ORDER):
        raise DressRehearsalError("rehearsal_stage_count_invalid")
    if [row.get("name") for row in stages] != list(STAGE_ORDER):
        raise DressRehearsalError("rehearsal_stage_order_invalid")
    for index, row in enumerate(stages):
        if not isinstance(row, Mapping) or set(row) != {
            "index",
            "name",
            "status",
            "summary",
            "summary_sha256",
        }:
            raise DressRehearsalError("rehearsal_stage_schema_invalid")
        if (
            row["index"] != index
            or row["status"] != "passed"
            or stable_hash(row["summary"]) != row["summary_sha256"]
        ):
            raise DressRehearsalError("rehearsal_stage_integrity_invalid")
    cleanup = report.get("cleanup")
    if cleanup != {
        "labels_persisted": False,
        "receipt_persisted": False,
        "run_artifacts_persisted": False,
        "temporary_namespace_cleaned": True,
    }:
        raise DressRehearsalError("rehearsal_cleanup_invalid")
    receipt = report.get("test_receipt")
    if receipt != {
        "formal_validation_complete": False,
        "receipt_identity_sha256": receipt.get("receipt_identity_sha256")
        if isinstance(receipt, Mapping)
        else None,
        "synthetic_test_only": True,
    } or not _is_digest(receipt.get("receipt_identity_sha256")):
        raise DressRehearsalError("rehearsal_receipt_boundary_invalid")
    if report.get("handoff_checklist_sha256") != build_handoff_checklist(protocol)[
        "checklist_sha256"
    ]:
        raise DressRehearsalError("handoff_checklist_binding_drift")
    _assert_no_sensitive_or_quality_payload(report)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "rehearsal_completed",
        "exit_code": EXIT_COMPLETED,
        "report_sha256": report["report_sha256"],
        "stage_count": len(stages),
        "synthetic_rehearsal_only": True,
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }


def simulate_failures(
    repository_root: Path,
    protocol: Mapping[str, Any],
    rehearsal_report: Mapping[str, Any],
) -> dict[str, Any]:
    root = repository_root.resolve()
    rejected: dict[str, bool] = {}

    def expect(name: str, operation: Any) -> None:
        try:
            operation()
        except (DressRehearsalError, ValueError, TypeError, KeyError):
            rejected[name] = True
        else:
            rejected[name] = False

    machine = RehearsalMachine(protocol)
    expect("reordered_stage", lambda: machine.advance("launch_authorized", {}))
    machine = RehearsalMachine(protocol)
    machine.advance("preregistration_sealed", {})
    expect("missing_stage", machine.finish)

    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as temporary:
        temporary_root = Path(temporary)
        drifted = copy.deepcopy(dict(protocol))
        drifted["source_commit"] = "0" * 40
        commit_drift = temporary_root / "commit-drift.json"
        write_json(commit_drift, drifted)
        expect(
            "cross_commit_mix",
            lambda: load_protocol(commit_drift, repository_root=root),
        )
        drifted = copy.deepcopy(dict(protocol))
        drifted["stage_order"][0] = "different_protocol_stage"
        protocol_drift = temporary_root / "protocol-drift.json"
        write_json(protocol_drift, drifted)
        expect(
            "cross_protocol_mix",
            lambda: load_protocol(protocol_drift, repository_root=root),
        )

    quarantine = load_quarantine_protocol(
        _binding_path(protocol, root, "quarantine_protocol")
    )
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX) as temporary:
        temp = Path(temporary)
        artifact = temp / "locked.json"
        artifact.write_bytes(canonical_json({"synthetic_rehearsal_only": True}))
        manifest = _build_intake(
            artifact=artifact,
            evidence_root=temp,
            evidence_type="human_annotation_labels",
            evidence_protocol_version="human_annotation_delivery_v1",
            quarantine_protocol=quarantine,
            plan_sha256="a" * 64,
        )
        machine = RehearsalMachine(protocol)
        machine.register_intake(manifest)
        expect("duplicate_intake", lambda: machine.register_intake(manifest))

    clearance_protocol = load_clearance_protocol(
        _binding_path(protocol, root, "clearance_protocol")
    )
    evidence = conformance_evidence()
    receipt = issue_receipt(evidence, clearance_protocol)
    machine = RehearsalMachine(protocol)
    machine.register_receipt(receipt)
    expect("duplicate_receipt", lambda: machine.register_receipt(receipt))

    launch_evidence = read_json(
        root / "benchmark/full1000_launch_control_v1_evidence/simulation.json"
    )
    rejected["old_attempt"] = any(
        row.get("scenario") == "stale_attempt" and row.get("blocked") is True
        for row in launch_evidence.get("scenarios", [])
    )
    partial_human = evaluate_clearance(
        conformance_evidence(satisfied=("full1000", "official_scorer"))
    )
    rejected["partial_human_labels"] = (
        partial_human.get("status") != "eligible_for_clearance"
    )
    partial_scorer = evaluate_clearance(
        conformance_evidence(satisfied=("full1000", "human_precision"))
    )
    rejected["partial_scorer_output"] = (
        partial_scorer.get("status") != "eligible_for_clearance"
    )

    before = "a" * 64
    after = "b" * 64
    for name, pointer in (
        ("posthoc_threshold_change", "/human_annotation/coverage_threshold"),
        ("posthoc_sample_change", "/population/exclusion_rules"),
        ("posthoc_statistics_change", "/statistics/resampling_unit"),
    ):
        amended = evaluate_amendment(
            changed_pointers=[pointer],
            evidence_intake_present=True,
            semantic_digest_before=before,
            semantic_digest_after=after,
        )
        rejected[name] = amended["state"] == "invalid_post_evidence_change"
    for name, path in (
        ("posthoc_prompt_change", "src/scholar_agent/prompts/query.txt"),
        ("posthoc_default_policy_change", "src/scholar_agent/core/config.py"),
    ):
        contamination = audit_contamination(manifest, [path], quarantine)
        rejected[name] = contamination["status"] == "stale_for_claim"

    revocation_protocol, _ledger, freshness, readiness = load_revocation_current(
        root
    )
    revocation = simulate_incidents(revocation_protocol, freshness, readiness)
    publication_scenario = next(
        row
        for row in revocation["scenarios"]
        if row["scenario"] == "default_policy_evidence_revoked"
    )
    rejected["revoked_upstream_evidence"] = {
        "clearance_receipt",
        "standalone_auditor_bundle",
        "validation_readiness_bundle",
    }.issubset(set(publication_scenario["invalidated_publication_targets"]))

    tampered = copy.deepcopy(dict(rehearsal_report))
    tampered["synthetic_rehearsal_only"] = False
    tampered["report_sha256"] = "0" * 64
    tampered["report_sha256"] = stable_hash(tampered)
    expect(
        "synthetic_real_state_pollution",
        lambda: verify_rehearsal_report(tampered, protocol),
    )

    missing = sorted(set(FAILURE_SCENARIOS) - set(rejected))
    failed = sorted(name for name, blocked in rejected.items() if not blocked)
    if missing or failed:
        raise DressRehearsalError("failure_matrix_expectation_mismatch")
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "rehearsal_completed",
        "exit_code": EXIT_COMPLETED,
        "synthetic_rehearsal_only": True,
        "scenario_count": len(rejected),
        "scenarios": [
            {"scenario": name, "rejected": rejected[name]}
            for name in FAILURE_SCENARIOS
        ],
        "real_state_mutation_count": 0,
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }
    report["report_sha256"] = stable_hash(report)
    return report


def audit_readiness(
    repository_root: Path,
    protocol: Mapping[str, Any],
    rehearsal_report: Mapping[str, Any],
) -> dict[str, Any]:
    root = repository_root.resolve()
    verified = verify_rehearsal_report(rehearsal_report, protocol)
    preregistration = load_preregistration_protocol(
        _binding_path(protocol, root, "preregistration_protocol")
    )
    seal = read_preregistration_json(
        _binding_path(protocol, root, "preregistration_seal")
    )
    preregistration_status = audit_preregistration_readiness(
        preregistration, seal, repository_root=root
    )
    human_protocol = load_delivery_protocol(
        _binding_path(protocol, root, "human_delivery_protocol"), root
    )
    human_status = human_delivery_readiness(
        human_protocol, repository_root=root
    )
    scorer_protocol = load_scorer_protocol(
        _binding_path(protocol, root, "external_scorer_protocol"),
        repository_root=root,
    )
    scorer_status = audit_scorer_readiness(
        scorer_protocol, repository_root=root
    )
    quarantine_protocol = load_quarantine_protocol(
        _binding_path(protocol, root, "quarantine_protocol")
    )
    quarantine_status = quarantine_current_readiness(root, quarantine_protocol)
    clearance_protocol = load_clearance_protocol(
        _binding_path(protocol, root, "clearance_protocol")
    )
    clearance_status = evaluate_clearance(
        build_current_evidence(clearance_protocol, repository_root=root)
    )
    revocation_status = audit_revocation_current(root)
    if (
        preregistration_status.get("exit_code") != EXIT_BLOCKED
        or human_status.get("exit_code") != EXIT_BLOCKED
        or scorer_status.get("exit_code") != EXIT_BLOCKED
        or quarantine_status.get("exit_code") != EXIT_BLOCKED
        or clearance_status.get("exit_code") != EXIT_BLOCKED
        or revocation_status.get("active_incident_count") != 0
    ):
        raise DressRehearsalError("real_readiness_boundary_drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "real_external_evidence_still_blocked",
        "exit_code": EXIT_BLOCKED,
        "controls_ready": True,
        "rehearsal_report_sha256": verified["report_sha256"],
        "synthetic_rehearsal_only": True,
        "real_blockers": list(REAL_BLOCKERS),
        "real_blocker_count": 3,
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }
