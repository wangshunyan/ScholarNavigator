"""Immutable, gold-blind execution readiness plan for AutoScholarQuery Full1000."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scholar_agent.evaluation.experiment_pairing import opaque_query_identity
from scholar_agent.evaluation.resource_accounting import (
    deterministic_fixture_report as resource_fixture_report,
)
from scholar_agent.evaluation.reproduction_capsule import (
    export_capsule,
    materialize_local_replay_run,
    replay_capsule,
    verify_capsule,
)
from scholar_agent.evaluation.sharded_execution import (
    build_local_fixture,
    deterministic_assignments,
    deterministic_fixture_report as shard_fixture_report,
    validate_and_merge,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash


PROTOCOL = "full1000_execution_readiness_v1"
PLAN_CONTRACT = "full1000_execution_plan_v1"
SCHEMA_VERSION = "1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4

EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
FORBIDDEN_KEYS = frozenset(
    {"gold", "qrels", "case_id", "target_paper", "quality_metric", "official_score"}
)
_OPAQUE_QUERY = re.compile(r"^query:[0-9a-f]{64}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Full1000ReadinessError(RuntimeError):
    """The plan or preflight evidence is inconsistent."""


class Full1000NotReady(Full1000ReadinessError):
    """A required immutable input is unavailable."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Full1000NotReady("required_json_unavailable") from exc
    if not isinstance(value, dict):
        raise Full1000ReadinessError("required_json_not_object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise Full1000ReadinessError("query_row_not_object")
                    rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Full1000NotReady("required_jsonl_unavailable") from exc
    return rows


def _normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _assert_no_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise Full1000ReadinessError(f"forbidden_field:{path}/{key}")
            _assert_no_forbidden_keys(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_forbidden_keys(item, f"{path}/{index}")


def _hashes(root: Path, paths: Sequence[str]) -> dict[str, str]:
    values = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise Full1000NotReady(f"required_input_missing:{relative}")
        values[relative] = sha256_file(path)
    return values


def _query_population(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    relative = str(protocol["inputs"]["query_input"])
    rows = _read_jsonl(root / relative)
    expected = int(protocol["population"]["query_count"])
    if len(rows) != expected:
        raise Full1000NotReady("query_count_mismatch")
    if any(set(row) != {"query_id", "query"} for row in rows):
        raise Full1000ReadinessError("query_input_field_drift")
    raw_ids = [str(row["query_id"]) for row in rows]
    if not all(raw_ids) or len(set(raw_ids)) != expected:
        raise Full1000ReadinessError("query_identity_not_unique")
    identities = [opaque_query_identity(value) for value in raw_ids]
    if len(set(identities)) != expected:
        raise Full1000ReadinessError("opaque_query_identity_collision")

    component_members: dict[str, list[str]] = defaultdict(list)
    for identity, row in zip(identities, rows, strict=True):
        normalized = _normalize_query(str(row["query"]))
        component = "component:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        component_members[component].append(identity)
    components = [
        {
            "component_identity": component,
            "query_count": len(members),
            "query_identities_sha256": stable_hash(members),
        }
        for component, members in sorted(component_members.items())
    ]
    return {
        "count": expected,
        "identities": identities,
        "stable_identity_sha256": stable_hash(sorted(identities)),
        "order_sha256": stable_hash(identities),
        "input_sha256": sha256_file(root / relative),
        "component_contract": "pre_retrieval_exact_normalized_query_components_v1",
        "component_count": len(components),
        "component_query_count": sum(item["query_count"] for item in components),
        "components_sha256": stable_hash(components),
        "components": components,
    }


def _planned_resources(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(root / str(protocol["inputs"]["planning_baseline"]))
    source_slots: Counter[str] = Counter()
    subquery_count = 0
    for row in rows:
        if row.get("status") != "success" or not isinstance(row.get("plan"), dict):
            raise Full1000ReadinessError("planning_baseline_incomplete")
        plan = row["plan"]
        subqueries = plan.get("subqueries") or []
        sources = plan.get("selected_sources") or []
        subquery_count += len(subqueries)
        for source in sources:
            source_slots[str(source)] += len(subqueries)
    if len(rows) != int(protocol["population"]["query_count"]):
        raise Full1000ReadinessError("planning_population_mismatch")
    expected_sources = list(protocol["execution_contract"]["sources"])
    if sorted(source_slots) != sorted(expected_sources):
        raise Full1000ReadinessError("planned_source_set_drift")
    logical = sum(source_slots.values())
    retry_sources = {"arxiv", "openalex", "semantic_scholar"}
    retry_upper = sum(source_slots[source] for source in retry_sources)
    http_upper = logical + retry_upper + source_slots["pubmed"]
    shard_count = int(protocol["sharding"]["shard_count"])
    generations_selected = len(rows) + (2 * shard_count)
    attempts_per_shard = int(protocol["attempts"]["max_attempts_per_shard"])
    generations_all_attempts = generations_selected * attempts_per_shard
    generation_files_upper = (6 * len(rows) + 15 * shard_count) * attempts_per_shard
    return {
        "query_count": len(rows),
        "subquery_count": subquery_count,
        "source_logical_request_upper": dict(sorted(source_slots.items())),
        "logical_source_request_upper": logical,
        "http_request_attempt_upper": http_upper,
        "retry_upper": retry_upper,
        "pagination_upper": 0,
        "candidate_records_before_global_budget_upper": logical
        * int(protocol["execution_contract"]["limit_per_source"]),
        "checkpoint_generation_selected_attempt_upper": generations_selected,
        "checkpoint_generation_all_attempts_upper": generations_all_attempts,
        "checkpoint_file_all_attempts_upper": generation_files_upper,
        "snapshot_key_file_upper": logical + 1,
        "disk_bytes_upper": "not_available",
        "provider_token_upper": "not_available",
        "provider_cost_upper": "not_available",
        "provider_rate_limit": "not_available",
        "derivation": "frozen_plan_slots_and_versioned_connector_request_semantics_v1",
    }


def build_plan(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    _assert_no_forbidden_keys(protocol)
    if protocol.get("protocol") != PROTOCOL or protocol.get("schema_version") != SCHEMA_VERSION:
        raise Full1000ReadinessError("protocol_version_mismatch")
    inputs = list(protocol["input_hashes"])
    observed_hashes = _hashes(root, inputs)
    if observed_hashes != protocol["input_hashes"]:
        raise Full1000ReadinessError("frozen_input_hash_drift")
    population = _query_population(root, protocol)
    identities = population["identities"]
    shard_count = int(protocol["sharding"]["shard_count"])
    assignments = deterministic_assignments(identities, shard_count)
    shards = [
        {
            "shard_index": index,
            "query_count": len(values),
            "query_identities": values,
            "query_identities_sha256": stable_hash(values),
            "attempts": [
                {
                    "attempt_id": f"shard-{index:02d}-attempt-0",
                    "supersedes": None,
                    "selection": "initial",
                },
                {
                    "attempt_id": f"shard-{index:02d}-attempt-1",
                    "supersedes": f"shard-{index:02d}-attempt-0",
                    "selection": "uniform_retry_only_if_initial_not_completed",
                },
            ],
        }
        for index, values in enumerate(assignments)
    ]
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": PLAN_CONTRACT,
        "source_commit": protocol["source_commit"],
        "status": "execution_plan_ready_but_network_blocked",
        "scope": "execution_readiness_only_not_completed_baseline_or_quality_score",
        "population": population,
        "input_hashes": observed_hashes,
        "execution_contract": protocol["execution_contract"],
        "sharding": {
            **protocol["sharding"],
            "shards": shards,
            "assignment_sha256": stable_hash(
                [item["query_identities_sha256"] for item in shards]
            ),
        },
        "attempts": protocol["attempts"],
        "resume": protocol["resume"],
        "legacy_artifacts": protocol["legacy_artifacts"],
        "resource_upper_bounds": _planned_resources(root, protocol),
        "network_status": "network_not_checked",
        "credential_status": "deferred_to_application_configuration_loader_not_inspected",
        "execution": EXECUTION,
        "formal_validation_complete": False,
    }
    _assert_no_forbidden_keys(plan)
    plan["plan_sha256"] = stable_hash(plan)
    return plan


def verify_plan(root: Path, protocol: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    try:
        expected = build_plan(root, protocol)
    except Full1000ReadinessError as exc:
        expected = {}
        violations.append(str(exc))
    if plan != expected:
        violations.append("plan_content_or_digest_drift")
    if plan.get("contract") != PLAN_CONTRACT:
        violations.append("plan_contract_mismatch")
    if plan.get("source_commit") != protocol.get("source_commit"):
        violations.append("source_commit_drift")
    population = plan.get("population") or {}
    identities = population.get("identities") or []
    if len(identities) != 1000 or len(set(identities)) != 1000:
        violations.append("query_population_not_closed")
    if any(not _OPAQUE_QUERY.fullmatch(str(value)) for value in identities):
        violations.append("query_identity_not_opaque")
    shards = ((plan.get("sharding") or {}).get("shards") or [])
    flattened = [identity for shard in shards for identity in shard.get("query_identities", [])]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(identities):
        violations.append("shard_partition_not_closed")
    if (plan.get("resume") or {}).get("start_mode") != "full_restart_all_1000":
        violations.append("legacy_partial_run_reuse_forbidden")
    status = "plan_or_preflight_violation" if violations else "execution_plan_ready_but_network_blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": EXIT_VIOLATION if violations else EXIT_READY,
        "query_count": len(identities),
        "shard_count": len(shards),
        "violations": sorted(set(violations)),
        "network_status": "network_not_checked",
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }


def preflight(root: Path, protocol: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    verification = verify_plan(root, protocol, plan)
    snapshot_relative = str(protocol["paths"]["snapshot_directory"])
    snapshot_path = root / snapshot_relative
    parent = snapshot_path.parent
    disk = shutil.disk_usage(parent if parent.exists() else root)
    minimum = int(protocol["preflight"]["minimum_free_disk_bytes"])
    checks = {
        "plan_verified": verification["exit_code"] == 0,
        "query_identity_and_order_closed": verification["query_count"] == 1000,
        "component_membership_closed": int(plan["population"]["component_query_count"]) == 1000,
        "snapshot_parent_exists": parent.is_dir(),
        "snapshot_parent_writable": os.access(parent, os.W_OK),
        "minimum_disk_budget_available": disk.free >= minimum,
        "network_not_checked": True,
        "environment_file_not_read": True,
        "legacy_record_reuse_disabled": plan["resume"]["start_mode"] == "full_restart_all_1000",
        "default_experimental_features_disabled": not any(
            plan["execution_contract"]["experimental_features"].values()
        ),
    }
    violations = sorted(key for key, value in checks.items() if not value)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "execution_plan_ready_but_network_blocked" if not violations else "plan_or_preflight_violation",
        "exit_code": EXIT_READY if not violations else EXIT_VIOLATION,
        "checks": checks,
        "violations": violations,
        "network_status": "network_not_checked",
        "credential_status": "deferred_to_application_configuration_loader_not_inspected",
        "minimum_free_disk_bytes": minimum,
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }


def dry_run(plan: Mapping[str, Any]) -> dict[str, Any]:
    identities = list(plan["population"]["identities"])
    shard_count = int(plan["sharding"]["shard_count"])
    with tempfile.TemporaryDirectory(prefix="full1000-readiness-") as temporary:
        root = Path(temporary)
        plan_path, attempts, monolithic, _ = build_local_fixture(
            root,
            shard_count=shard_count,
            query_identities=identities,
        )
        aggregate_path = root / "aggregate.json"
        merge = validate_and_merge(
            plan_path,
            attempts,
            repository_root=root,
            output_path=aggregate_path,
            monolithic_manifest_path=monolithic,
        )
        aggregate = _read_json(aggregate_path)
        top20_valid = all(
            len(row.get("normalized_results") or []) <= 20
            and [item.get("rank") for item in row.get("normalized_results") or []]
            == list(range(1, len(row.get("normalized_results") or []) + 1))
            for row in aggregate["records"]
        )
        capsule_source = materialize_local_replay_run(
            root / "capsule-source",
            host_repository_root=REPOSITORY_ROOT,
            execution_protocol_path=(
                REPOSITORY_ROOT / "benchmark/execution_determinism_v1_protocol.json"
            ),
        )
        capsule_path = root / "fixture-capsule.tar"
        capsule_export = export_capsule(
            capsule_source,
            capsule_path,
            host_repository_root=REPOSITORY_ROOT,
        )
        capsule_verify = verify_capsule(capsule_path)
        capsule_replay = replay_capsule(
            capsule_path,
            host_repository_root=REPOSITORY_ROOT,
        )
        capsule_sha256 = sha256_file(capsule_path)
    resume = shard_fixture_report(retry_shard=1)
    ledger = resource_fixture_report(shard_resume=True)
    capsule_limit = int(plan["execution_contract"]["protocol_limits"]["capsule_max_files"])
    per_shard_queries = max(item["query_count"] for item in plan["sharding"]["shards"])
    per_shard_generation_files = 6 * per_shard_queries + 15
    stages = {
        "sharded_execution": merge.get("exit_code") == 0 and merge.get("query_count") == 1000,
        "checkpoint_and_aggregate": len(aggregate.get("records") or []) == 1000,
        "resume_supersession": resume.get("exit_code") == 0,
        "resource_ledger_conformance": ledger.get("exit_code") == 0,
        "run_manifest_binding": merge.get("selected_shard_count") == shard_count,
        "capsule_file_limit_preflight": per_shard_generation_files < capsule_limit,
        "reproduction_capsule": all(
            item.get("exit_code") == 0
            for item in (capsule_export, capsule_verify, capsule_replay)
        ),
        "top20_delivery_contract": top20_valid,
    }
    violations = sorted(key for key, value in stages.items() if not value)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "execution_plan_ready_but_network_blocked" if not violations else "plan_or_preflight_violation",
        "exit_code": EXIT_READY if not violations else EXIT_VIOLATION,
        "query_count": len(identities),
        "shard_count": shard_count,
        "stages": stages,
        "terminal_counts": merge.get("terminal_counts"),
        "aggregate_sha256": merge.get("aggregate_sha256"),
        "capsule_preflight": {
            "scope": "per_shard_file_limit_plus_ephemeral_export_verify_replay",
            "generation_file_upper_per_selected_shard": per_shard_generation_files,
            "protocol_file_limit": capsule_limit,
            "ephemeral_capsule_sha256": capsule_sha256,
            "ephemeral_replay_query_count": (
                capsule_replay.get("replay") or {}
            ).get("query_count"),
        },
        "fixture_only": True,
        "fixture_results_are_not_formal_evidence_or_quality_statistics": True,
        "violations": violations,
        "network_status": "network_not_checked",
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }
