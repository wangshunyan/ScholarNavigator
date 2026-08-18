"""Gold-blind Full1000 network-request intent manifest and Snapshot gap audit."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scholar_agent.agents.retriever import retrieval_cache_identity
from scholar_agent.connectors.arxiv import describe_arxiv_search_request
from scholar_agent.connectors.openalex import describe_openalex_search_request
from scholar_agent.connectors.pubmed import describe_pubmed_search_request
from scholar_agent.connectors.schemas import ConnectorRequestSpec
from scholar_agent.connectors.semantic_scholar import (
    describe_semantic_scholar_search_request,
)
from scholar_agent.core.search_schemas import QueryConstraint
from scholar_agent.evaluation.experiment_pairing import opaque_query_identity
from scholar_agent.evaluation.relevance_filter_audit import _tree_sha256
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash
from scholar_agent.evaluation.snapshots import SnapshotStore
from scholar_agent.evaluation.snapshots.store import (
    connector_version,
    retrieval_snapshot_key,
)
from scholar_agent.evaluation.source_reliability_diagnostics import (
    audit_retrieval_requests,
)
from scholar_agent.retrieval.query_adapter import adapt_queries_for_source


PROTOCOL = "formal_network_request_manifest_v1"
MANIFEST_CONTRACT = "formal_network_request_manifest_v1"
INTENT_CONTRACT = "formal_network_request_intent_v1"
SNAPSHOT_AUDIT_CONTRACT = "formal_network_snapshot_gap_audit_v1"
LAUNCH_ADDENDUM_CONTRACT = "full1000_network_request_manifest_addendum_v1"
SCHEMA_VERSION = "1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCES = ("openalex", "arxiv", "semantic_scholar", "pubmed")
OPAQUE_QUERY_RE = re.compile(r"^query:[0-9a-f]{64}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "credential",
        "environment",
        "headers",
        "query",
        "raw_url",
        "request_url",
    }
)
DESCRIBERS = {
    "arxiv": describe_arxiv_search_request,
    "openalex": describe_openalex_search_request,
    "pubmed": describe_pubmed_search_request,
    "semantic_scholar": describe_semantic_scholar_search_request,
}


class NetworkRequestManifestError(RuntimeError):
    """The request identity, frozen plan, or emitted artifact is invalid."""


class NetworkRequestManifestNotReady(NetworkRequestManifestError):
    """A required frozen planning or request artifact is unavailable."""


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


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise NetworkRequestManifestError("duplicate_json_key")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non_finite_json")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NetworkRequestManifestNotReady("required_json_unavailable") from exc
    if not isinstance(value, dict):
        raise NetworkRequestManifestError("json_root_not_object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise NetworkRequestManifestNotReady("required_jsonl_unavailable") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                object_pairs_hook=_pairs_no_duplicates,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ValueError("non_finite_json")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise NetworkRequestManifestError("jsonl_row_invalid") from exc
        if not isinstance(value, dict):
            raise NetworkRequestManifestError("jsonl_row_not_object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(dict(value)))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_jsonl(rows))


def load_protocol(
    path: Path, *, repository_root: Path = REPOSITORY_ROOT
) -> dict[str, Any]:
    protocol = load_json(path)
    if (
        protocol.get("protocol") != PROTOCOL
        or protocol.get("schema_version") != SCHEMA_VERSION
    ):
        raise NetworkRequestManifestError("protocol_version_mismatch")
    if protocol.get("source_commit") != (
        "fae8a7a56c2c6ef877e36dd8901e8a19de6af318"
    ):
        raise NetworkRequestManifestError("protocol_source_commit_drift")
    if protocol.get("sources") != list(SOURCES):
        raise NetworkRequestManifestError("protocol_source_order_drift")
    if protocol.get("execution") != EXECUTION_ZERO:
        raise NetworkRequestManifestError("offline_execution_contract_drift")
    if protocol.get("selection_prohibitions") != [
        "gold",
        "qrels",
        "case_id",
        "target_paper",
        "retrieval_result",
        "source_yield",
        "quality_metric",
    ]:
        raise NetworkRequestManifestError("selection_prohibition_drift")
    for relative, expected in protocol.get("input_hashes", {}).items():
        target = repository_root / str(relative)
        if not target.is_file():
            raise NetworkRequestManifestNotReady("required_input_missing")
        if sha256_file(target) != expected:
            raise NetworkRequestManifestError("frozen_input_hash_drift")
    return protocol


def _parameter_evidence(parameters: Mapping[str, str]) -> dict[str, Any]:
    ordered = [(str(key), str(parameters[key])) for key in sorted(parameters)]
    return {
        "names": [key for key, _value in ordered],
        "value_sha256": {
            key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in ordered
        },
        "parameters_sha256": stable_hash(ordered),
        "serialization": "sorted_name_value_pairs_utf8_v1",
    }


def _safe_child_rules(spec: ConnectorRequestSpec) -> list[dict[str, Any]]:
    allowed = {
        "accept",
        "cursor_or_ids",
        "endpoint_alias",
        "max_retries",
        "method",
        "parameter_rule",
        "parent_field",
        "response_media_type",
    }
    values: list[dict[str, Any]] = []
    for child in spec.response_dependent_children:
        if set(child) != allowed:
            raise NetworkRequestManifestError("response_child_schema_drift")
        values.append({key: child[key] for key in sorted(child)})
    return values


def _describe_source_request(
    source: str,
    query: str,
    *,
    limit: int,
    constraints: QueryConstraint,
    combination_mode: str,
    adapter_policy: str,
) -> tuple[ConnectorRequestSpec, str]:
    variants = adapt_queries_for_source(
        query,
        source,
        constraints=constraints,
        policy=adapter_policy,
        combination_mode=combination_mode,  # type: ignore[arg-type]
    )
    if not variants or not variants[0].query:
        raise NetworkRequestManifestError("initial_adapted_query_missing")
    initial = variants[0]
    spec = DESCRIBERS[source](initial.query, limit)
    if spec.adapted_query != initial.query:
        raise NetworkRequestManifestError("connector_adapter_roundtrip_drift")
    return spec, initial.strategy


def _query_rows(
    root: Path, protocol: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = load_jsonl(root / str(protocol["inputs"]["query_input"]))
    planning = load_jsonl(root / str(protocol["inputs"]["planning_baseline"]))
    if len(inputs) != 1000 or len(planning) != 1000:
        raise NetworkRequestManifestNotReady("full1000_population_incomplete")
    if any(set(row) != {"query_id", "query"} for row in inputs):
        raise NetworkRequestManifestError("query_input_schema_drift")
    identities = [opaque_query_identity(str(row["query_id"])) for row in inputs]
    population = plan.get("population") or {}
    if identities != population.get("identities"):
        raise NetworkRequestManifestError("query_identity_or_order_drift")
    if len(set(identities)) != 1000 or any(
        not OPAQUE_QUERY_RE.fullmatch(value) for value in identities
    ):
        raise NetworkRequestManifestError("query_identity_not_closed")
    planning_by_id = {str(row.get("query_id")): row for row in planning}
    if len(planning_by_id) != 1000:
        raise NetworkRequestManifestError("planning_identity_duplicate")
    ordered_planning: list[dict[str, Any]] = []
    for row in inputs:
        planned = planning_by_id.get(str(row["query_id"]))
        if planned is None or planned.get("status") != "success":
            raise NetworkRequestManifestNotReady("planning_row_missing")
        if planned.get("input_query_sha256") != hashlib.sha256(
            str(row["query"]).encode("utf-8")
        ).hexdigest():
            raise NetworkRequestManifestError("planning_query_binding_drift")
        ordered_planning.append(planned)
    return inputs, ordered_planning


def build_request_manifest(
    repository_root: Path,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand all frozen source slots without opening sockets or reading config."""

    plan = load_json(repository_root / str(protocol["inputs"]["full1000_plan"]))
    if plan.get("plan_sha256") != protocol.get("full1000_plan_sha256"):
        raise NetworkRequestManifestError("full1000_plan_binding_drift")
    inputs, planning_rows = _query_rows(repository_root, protocol, plan)
    execution = plan["execution_contract"]
    sources = list(execution["sources"])
    if sources != list(SOURCES):
        raise NetworkRequestManifestError("execution_source_order_drift")
    limit = int(execution["limit_per_source"])
    adapter_policy = str(execution["query_adapter_policy"])
    planner_policy = str(execution["query_planning_policy"])
    planner_version = str(execution["query_planner_version"])
    retry_limits = execution["protocol_limits"]["connector_max_retries"]
    intents: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    shard_counts: Counter[int] = Counter()
    subquery_count = 0
    request_spec_to_cache: dict[str, set[str]] = defaultdict(set)
    cache_to_specs: dict[str, set[str]] = defaultdict(set)
    request_ids: set[str] = set()

    for query_order, (input_row, planning_row) in enumerate(
        zip(inputs, planning_rows, strict=True)
    ):
        query_identity = opaque_query_identity(str(input_row["query_id"]))
        shard_index = query_order % int(protocol["shard_count"])
        planned = planning_row["plan"]
        if planned.get("selected_sources") != sources:
            raise NetworkRequestManifestError("planning_source_set_drift")
        constraints = QueryConstraint.model_validate(
            planned["query_analysis"]["constraints"]
        )
        subqueries = planned.get("subqueries")
        if not isinstance(subqueries, list) or not 1 <= len(subqueries) <= 3:
            raise NetworkRequestManifestError("subquery_count_out_of_contract")
        subquery_count += len(subqueries)
        for subquery_index, subquery in enumerate(subqueries):
            if not isinstance(subquery, dict):
                raise NetworkRequestManifestError("subquery_schema_invalid")
            text = str(subquery.get("query") or "")
            if not text:
                raise NetworkRequestManifestError("subquery_text_missing")
            subquery_identity = "subquery:" + stable_hash(
                {
                    "query_identity": query_identity,
                    "subquery_index": subquery_index,
                    "plan_sha256": planning_row["plan_sha256"],
                }
            )
            for source_order, source in enumerate(sources):
                spec, strategy = _describe_source_request(
                    source,
                    text,
                    limit=limit,
                    constraints=constraints,
                    combination_mode=str(subquery["combination_mode"]),
                    adapter_policy=adapter_policy,
                )
                if spec.max_retries != int(retry_limits[source]):
                    raise NetworkRequestManifestError("connector_retry_plan_drift")
                version = connector_version(source)
                snapshot_key, normalized_query = retrieval_snapshot_key(
                    source=source,
                    adapted_query=spec.adapted_query,
                    limit=limit,
                    adapter_policy=adapter_policy,
                    connector_version=version,
                    query_planning_policy=planner_policy,  # type: ignore[arg-type]
                    query_planner_version=planner_version,
                )
                cache_tuple = retrieval_cache_identity(
                    source, spec.adapted_query, limit
                )
                cache_key_sha256 = stable_hash(
                    {
                        "cache_contract": "retriever_cache_key_v1",
                        "source": cache_tuple[0],
                        "adapted_query_sha256": hashlib.sha256(
                            cache_tuple[1].encode("utf-8")
                        ).hexdigest(),
                        "limit": cache_tuple[2],
                    }
                )
                parameter_evidence = _parameter_evidence(spec.parameters)
                safe_spec = {
                    "source": source,
                    "endpoint_alias": spec.endpoint_alias,
                    "method": spec.method,
                    "parameter_evidence": parameter_evidence,
                    "timeout_seconds": spec.timeout_seconds,
                    "max_retries": spec.max_retries,
                    "auth_scope_alias": spec.auth_scope_alias,
                    "auth_affects_response_semantics": (
                        spec.auth_affects_response_semantics
                    ),
                    "accept": spec.accept,
                    "response_media_type": spec.response_media_type,
                    "page_budget": spec.page_budget,
                    "pagination_strategy": spec.pagination_strategy,
                    "response_dependent_children": _safe_child_rules(spec),
                    "adapted_query_sha256": hashlib.sha256(
                        normalized_query.encode("utf-8")
                    ).hexdigest(),
                }
                request_spec_sha256 = stable_hash(safe_spec)
                identity_payload = {
                    "protocol": PROTOCOL,
                    "query_identity": query_identity,
                    "subquery_identity": subquery_identity,
                    "source": source,
                    "request_spec_sha256": request_spec_sha256,
                }
                intent_id = "request:" + stable_hash(identity_payload)
                if intent_id in request_ids:
                    raise NetworkRequestManifestError("request_identity_duplicate")
                request_ids.add(intent_id)
                child_attempts = sum(
                    int(item["max_retries"]) + 1
                    for item in spec.response_dependent_children
                )
                http_attempt_upper = spec.max_retries + 1 + child_attempts
                intent = {
                    "schema_version": SCHEMA_VERSION,
                    "contract": INTENT_CONTRACT,
                    "intent_id": intent_id,
                    "query_identity": query_identity,
                    "query_order": query_order,
                    "shard_index": shard_index,
                    "subquery_identity": subquery_identity,
                    "subquery_index": subquery_index,
                    "subquery_plan_sha256": str(planning_row["plan_sha256"]),
                    "source": source,
                    "source_order": source_order,
                    "adapter_policy": adapter_policy,
                    "adaptation_strategy": strategy,
                    "query_adapter_version": str(
                        protocol["query_adapter_version"]
                    ),
                    "connector_version": version,
                    "request_spec": safe_spec,
                    "request_spec_sha256": request_spec_sha256,
                    "production_cache_key_sha256": cache_key_sha256,
                    "snapshot_key": snapshot_key,
                    "http_attempt_upper": http_attempt_upper,
                    "execution_condition": "initial_source_slot",
                    "historical_authority": "new_full1000_intent_not_completed_work",
                }
                _assert_safe_output(intent)
                intents.append(intent)
                source_counts[source] += 1
                shard_counts[shard_index] += 1
                request_spec_to_cache[request_spec_sha256].add(cache_key_sha256)
                cache_to_specs[cache_key_sha256].add(request_spec_sha256)

    expected = plan["resource_upper_bounds"]
    http_upper = sum(int(item["http_attempt_upper"]) for item in intents)
    if (
        len(intents) != int(expected["logical_source_request_upper"])
        or subquery_count != int(expected["subquery_count"])
        or dict(sorted(source_counts.items()))
        != expected["source_logical_request_upper"]
        or http_upper != int(expected["http_request_attempt_upper"])
    ):
        raise NetworkRequestManifestError("request_plan_count_drift")
    if len(shard_counts) != 20 or sum(shard_counts.values()) != len(intents):
        raise NetworkRequestManifestError("shard_request_coverage_drift")
    collisions = {
        key: sorted(values)
        for key, values in cache_to_specs.items()
        if len(values) > 1
    }
    same_request_multiple_keys = {
        key: sorted(values)
        for key, values in request_spec_to_cache.items()
        if len(values) > 1
    }
    if collisions:
        raise NetworkRequestManifestError("semantic_cache_collision")
    if same_request_multiple_keys:
        raise NetworkRequestManifestError("same_request_multiple_cache_keys")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": MANIFEST_CONTRACT,
        "status": "request_manifest_ready_network_blocked",
        "source_commit": protocol["source_commit"],
        "full1000_plan_sha256": protocol["full1000_plan_sha256"],
        "query_count": 1000,
        "query_identity_order_sha256": plan["population"]["order_sha256"],
        "shard_count": int(protocol["shard_count"]),
        "subquery_count": subquery_count,
        "logical_source_request_count": len(intents),
        "http_attempt_upper": http_upper,
        "pagination_followup_count_materialized": 0,
        "source_counts": dict(sorted(source_counts.items())),
        "shard_counts": {
            str(key): shard_counts[key] for key in sorted(shard_counts)
        },
        "unique_snapshot_key_count": len(
            {str(item["snapshot_key"]) for item in intents}
        ),
        "shared_cache_intent_count": len(intents)
        - len({str(item["production_cache_key_sha256"]) for item in intents}),
        "cache_collision_count": 0,
        "same_request_multiple_cache_key_count": 0,
        "intents_sha256": stable_hash(intents),
        "input_hashes": dict(protocol["input_hashes"]),
        "network_status": "network_not_checked",
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    _assert_safe_output(manifest)
    return intents, manifest


def _historical_snapshot_keys(
    repository_root: Path, protocol: Mapping[str, Any]
) -> tuple[set[str], dict[str, Any]]:
    historical = protocol["historical_snapshot_audit"]
    reliability_protocol = load_json(
        repository_root / str(historical["source_reliability_protocol"])
    )
    frozen = reliability_protocol["frozen_input"]
    run_dir = repository_root / str(frozen["run_dir"])
    results_path = run_dir / "results.jsonl"
    config_path = run_dir / "config.json"
    snapshot_dir = repository_root / str(frozen["snapshot_dir"])
    if (
        sha256_file(results_path) != frozen["record_results_sha256"]
        or sha256_file(config_path) != frozen["config_sha256"]
        or _tree_sha256(snapshot_dir) != frozen["snapshot_tree_sha256"]
    ):
        raise NetworkRequestManifestNotReady("historical_snapshot_input_drift")
    config = load_json(config_path)
    store = SnapshotStore(snapshot_dir)
    observed: set[str] = set()
    rows = load_jsonl(results_path)
    for row in rows:
        diagnostics = row.get("stage_diagnostics")
        if not isinstance(diagnostics, dict):
            raise NetworkRequestManifestNotReady("historical_stage_missing")
        snapshots = diagnostics.get("snapshots")
        if not isinstance(snapshots, list):
            raise NetworkRequestManifestNotReady("historical_snapshot_stage_missing")
        stages = {
            str(item.get("stage")): item
            for item in snapshots
            if isinstance(item, dict)
        }
        initial = stages.get("initial_retrieval")
        if initial is None:
            raise NetworkRequestManifestNotReady("historical_initial_stage_missing")
        audit = audit_retrieval_requests(
            initial,
            config=config,
            store=store,
            sources=list(SOURCES),
        )
        observed.update(audit.observed_keys)
    expected = int(historical["expected_key_count"])
    if len(observed) != expected:
        raise NetworkRequestManifestNotReady("historical_key_count_drift")
    if any(not HEX64_RE.fullmatch(key) for key in observed):
        raise NetworkRequestManifestError("historical_key_invalid")
    return observed, {
        "snapshot_dir_identity": stable_hash(str(frozen["snapshot_dir"])),
        "snapshot_tree_sha256": frozen["snapshot_tree_sha256"],
        "historical_key_count": len(observed),
        "authority": "historical_reference_only_not_checkpoint_or_completion",
    }


def audit_snapshots(
    repository_root: Path,
    protocol: Mapping[str, Any],
    intents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed, evidence = _historical_snapshot_keys(repository_root, protocol)
    by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for intent in intents:
        by_key[str(intent["snapshot_key"])].append(intent)
    planned = set(by_key)
    exact = observed & planned
    orphaned = observed - planned
    uncovered = planned - observed
    conflicts: list[str] = []
    frozen = load_json(
        repository_root
        / str(protocol["historical_snapshot_audit"]["source_reliability_protocol"])
    )["frozen_input"]
    store = SnapshotStore(repository_root / str(frozen["snapshot_dir"]))
    exact_by_source: Counter[str] = Counter()
    historical_orphan_by_source: Counter[str] = Counter()
    planned_uncovered_by_source: Counter[str] = Counter()
    for key in sorted(exact):
        entry = store.read_retrieval(key)
        candidate = by_key[key][0]
        exact_by_source[str(candidate["source"])] += 1
        expected = {
            "source": candidate["source"],
            "limit": int(protocol["limit_per_source"]),
            "adapter_policy": candidate["adapter_policy"],
            "connector_version": candidate["connector_version"],
        }
        actual = {
            "source": entry.source,
            "limit": entry.limit,
            "adapter_policy": entry.adapter_policy,
            "connector_version": entry.connector_version,
        }
        if actual != expected:
            conflicts.append(key)
    for key in sorted(orphaned):
        historical_orphan_by_source[store.read_retrieval(key).source] += 1
    for key in sorted(uncovered):
        planned_uncovered_by_source[str(by_key[key][0]["source"])] += 1
    if conflicts:
        raise NetworkRequestManifestError("historical_snapshot_binding_conflict")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": SNAPSHOT_AUDIT_CONTRACT,
        "status": "completed",
        "authority": "historical_reference_only_not_checkpoint_or_completion",
        **evidence,
        "planned_unique_snapshot_key_count": len(planned),
        "exact_match_count": len(exact),
        "historical_orphan_count": len(orphaned),
        "planned_uncovered_count": len(uncovered),
        "conflict_count": 0,
        "by_source": {
            source: {
                "exact_match_count": exact_by_source[source],
                "historical_orphan_count": historical_orphan_by_source[source],
                "planned_uncovered_count": planned_uncovered_by_source[source],
            }
            for source in SOURCES
        },
        "exact_keys_sha256": stable_hash(sorted(exact)),
        "historical_orphan_keys_sha256": stable_hash(sorted(orphaned)),
        "planned_uncovered_keys_sha256": stable_hash(sorted(uncovered)),
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }
    report["report_sha256"] = stable_hash(report)
    _assert_safe_output(report)
    return report


def build_launch_addendum(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    snapshot_audit: Mapping[str, Any],
) -> dict[str, Any]:
    addendum: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": LAUNCH_ADDENDUM_CONTRACT,
        "source_commit": protocol["source_commit"],
        "full1000_plan_sha256": protocol["full1000_plan_sha256"],
        "request_manifest_sha256": manifest["manifest_sha256"],
        "intents_sha256": manifest["intents_sha256"],
        "logical_source_request_count": manifest["logical_source_request_count"],
        "http_attempt_upper": manifest["http_attempt_upper"],
        "snapshot_gap_report_sha256": snapshot_audit["report_sha256"],
        "launch_control_protocol_sha256": protocol["input_hashes"][
            "benchmark/full1000_launch_control_v1_protocol.json"
        ],
        "provider_ingest_provenance_protocol_sha256": protocol["input_hashes"][
            "benchmark/provider_ingest_provenance_v1_protocol.json"
        ],
        "historical_snapshot_authority": "historical_reference_only",
        "launch_requirement": (
            "regenerate_and_verify_identical_request_manifest_before_authorization"
        ),
        "network_status": "network_not_checked",
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }
    addendum["addendum_sha256"] = stable_hash(addendum)
    return addendum


def write_bundle(
    output_dir: Path,
    intents: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    snapshot_audit: Mapping[str, Any],
    launch_addendum: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise NetworkRequestManifestError("output_directory_not_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "intents.jsonl": canonical_jsonl(intents),
        "manifest.json": canonical_json(manifest),
        "snapshot_audit.json": canonical_json(snapshot_audit),
        "launch_addendum.json": canonical_json(launch_addendum),
    }
    for relative, content in files.items():
        (output_dir / relative).write_bytes(content)
    inventory = {
        relative: {
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for relative, content in sorted(files.items())
    }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "contract": "formal_network_request_manifest_bundle_v1",
        "status": "request_manifest_ready_network_blocked",
        "files": inventory,
        "bundle_sha256": stable_hash(inventory),
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }
    write_json(output_dir / "bundle.json", bundle)
    return bundle


def verify_bundle(
    output_dir: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    bundle = load_json(output_dir / "bundle.json")
    if (
        bundle.get("contract")
        != "formal_network_request_manifest_bundle_v1"
        or bundle.get("schema_version") != SCHEMA_VERSION
        or bundle.get("status") != "request_manifest_ready_network_blocked"
        or bundle.get("execution") != EXECUTION_ZERO
        or bundle.get("formal_validation_complete") is not False
    ):
        raise NetworkRequestManifestError("bundle_contract_drift")
    expected_files = {
        "intents.jsonl",
        "launch_addendum.json",
        "manifest.json",
        "snapshot_audit.json",
    }
    inventory = bundle.get("files")
    if not isinstance(inventory, dict) or set(inventory) != expected_files:
        raise NetworkRequestManifestError("bundle_inventory_drift")
    if bundle.get("bundle_sha256") != stable_hash(inventory):
        raise NetworkRequestManifestError("bundle_self_hash_drift")
    actual_members = {
        item.name
        for item in output_dir.iterdir()
        if item.is_file() and item.name != "bundle.json"
    }
    if actual_members != expected_files:
        raise NetworkRequestManifestError("bundle_member_drift")
    for relative, metadata in inventory.items():
        path = output_dir / relative
        content = path.read_bytes()
        if (
            metadata.get("size") != len(content)
            or metadata.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise NetworkRequestManifestError("bundle_file_hash_drift")
    intents = load_jsonl(output_dir / "intents.jsonl")
    manifest = load_json(output_dir / "manifest.json")
    audit = load_json(output_dir / "snapshot_audit.json")
    addendum = load_json(output_dir / "launch_addendum.json")
    rebuilt_intents, rebuilt_manifest = build_request_manifest(
        repository_root, protocol
    )
    if intents != rebuilt_intents or manifest != rebuilt_manifest:
        raise NetworkRequestManifestError("request_manifest_rebuild_drift")
    rebuilt_audit = audit_snapshots(repository_root, protocol, intents)
    if audit != rebuilt_audit:
        raise NetworkRequestManifestError("snapshot_audit_rebuild_drift")
    if addendum != build_launch_addendum(protocol, manifest, audit):
        raise NetworkRequestManifestError("launch_addendum_drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "request_manifest_ready_network_blocked",
        "exit_code": EXIT_READY,
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "logical_source_request_count": manifest["logical_source_request_count"],
        "http_attempt_upper": manifest["http_attempt_upper"],
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }


def audit_readiness(
    repository_root: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    intents, manifest = build_request_manifest(repository_root, protocol)
    snapshot_audit = audit_snapshots(repository_root, protocol, intents)
    addendum = build_launch_addendum(protocol, manifest, snapshot_audit)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "request_manifest_ready_network_blocked",
        "exit_code": EXIT_READY,
        "manifest_sha256": manifest["manifest_sha256"],
        "launch_addendum_sha256": addendum["addendum_sha256"],
        "query_count": manifest["query_count"],
        "subquery_count": manifest["subquery_count"],
        "logical_source_request_count": manifest[
            "logical_source_request_count"
        ],
        "http_attempt_upper": manifest["http_attempt_upper"],
        "snapshot_audit": {
            key: snapshot_audit[key]
            for key in (
                "historical_key_count",
                "exact_match_count",
                "historical_orphan_count",
                "planned_uncovered_count",
                "conflict_count",
            )
        },
        "network_status": "network_not_checked",
        "execution": EXECUTION_ZERO,
        "formal_validation_complete": False,
    }


def _assert_safe_output(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            parameter_hash_name = (
                normalized == "query"
                and path.endswith("/parameter_evidence/value_sha256")
            )
            if normalized in FORBIDDEN_OUTPUT_KEYS and not parameter_hash_name:
                raise NetworkRequestManifestError(
                    f"forbidden_output_field:{path}/{key}"
                )
            _assert_safe_output(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_safe_output(item, f"{path}/{index}")
    elif isinstance(value, str):
        registered_auth_alias = path.endswith("/auth_scope_alias") and value in {
            "public_anonymous",
            "openalex_polite_pool_optional",
            "semantic_scholar_api_key_optional",
            "ncbi_api_key_optional",
        }
        if value.startswith("/") or (
            not registered_auth_alias
            and re.search(
                r"(?i)(api[_-]?key|authorization|bearer\s+|[?&](?:key|token)=)",
                value,
            )
        ):
            raise NetworkRequestManifestError(f"sensitive_output_value:{path}")
