"""Gold-blind, offline regression audit for deterministic query planning.

This module intentionally does not import dataset adapters, SearchService,
connectors, evaluators, or Snapshot runtimes.  Its only input is a query-only
JSONL projection containing ``query_id`` and ``query``.
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_agent.agents import query_understanding
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.core.search_schemas import (
    DEFAULT_SEARCH_SOURCES,
    QUERY_PLANNER_VERSION,
    SearchBudget,
    SearchPlan,
)
from scholar_agent.evaluation.current_rules_regression import compare_profiles


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GATE_NAME = "autoscholar_query_planning_regression"
SCHEMA_VERSION = "1"
BASELINE_APPROVAL_TOKEN = "PROPOSE_QUERY_PLANNING_BASELINE"
DEFAULT_SNAPSHOT_ROOT = REPOSITORY_ROOT / "outputs" / "benchmark_snapshots"
REQUIRED_PLAN_FIELDS = frozenset(
    {
        "query_analysis",
        "subqueries",
        "selected_sources",
        "limit_per_source",
        "top_k",
        "run_profile",
        "enable_refchain",
        "enable_semantic_seed_expansion",
        "enable_query_evolution",
        "query_evolution_policy",
        "query_planning_policy",
        "ranking_policy",
        "query_planning",
        "warnings",
    }
)


class QueryPlanningAuditError(RuntimeError):
    """Raised when an offline audit invariant is violated."""


class QueryOnlyRecord(BaseModel):
    """The complete allowed dataset contract for this gold-blind audit."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    query: str


def project_query_only_manifest(source: Path, destination: Path) -> dict[str, Any]:
    """Project only qid/question from AutoScholarQuery JSONL.

    Non-query fields are neither inspected nor copied.  In particular, this
    function has no dependency on the dataset evaluator or its gold schema.
    """

    if destination.exists():
        raise QueryPlanningAuditError(
            "query-only manifest already exists; write a review artifact elsewhere"
        )
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = _select_top_level_string_fields(
                raw_line,
                fields=frozenset({"qid", "question"}),
                line_number=line_number,
            )
            query_id = payload.get("qid")
            query = payload.get("question")
            if not isinstance(query_id, str) or not query_id.strip():
                raise QueryPlanningAuditError(
                    f"invalid source record at line {line_number}:missing_qid"
                )
            if not isinstance(query, str):
                raise QueryPlanningAuditError(
                    f"invalid source record at line {line_number}:missing_question"
                )
            if query_id in seen:
                raise QueryPlanningAuditError(f"duplicate query_id:{query_id}")
            seen.add(query_id)
            rows.append({"query_id": query_id, "query": query})
    _write_jsonl(destination, rows)
    return {
        "case_count": len(rows),
        "query_manifest_sha256": sha256_file(destination),
        "allowed_fields": ["query_id", "query"],
        "gold_fields_accessed": False,
    }


def _select_top_level_string_fields(
    raw: str,
    *,
    fields: frozenset[str],
    line_number: int,
) -> dict[str, str]:
    """Decode selected top-level strings while structurally skipping all else."""

    decoder = json.JSONDecoder()
    selected: dict[str, str] = {}
    index = _skip_whitespace(raw, 0)
    if index >= len(raw) or raw[index] != "{":
        raise QueryPlanningAuditError(
            f"invalid source record at line {line_number}:expected_object"
        )
    index += 1
    while True:
        index = _skip_whitespace(raw, index)
        if index < len(raw) and raw[index] == "}":
            index += 1
            break
        try:
            key, index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError as exc:
            raise QueryPlanningAuditError(
                f"invalid source record at line {line_number}:invalid_key"
            ) from exc
        if not isinstance(key, str):
            raise QueryPlanningAuditError(
                f"invalid source record at line {line_number}:invalid_key"
            )
        index = _skip_whitespace(raw, index)
        if index >= len(raw) or raw[index] != ":":
            raise QueryPlanningAuditError(
                f"invalid source record at line {line_number}:missing_colon"
            )
        index = _skip_whitespace(raw, index + 1)
        if key in fields:
            try:
                value, index = decoder.raw_decode(raw, index)
            except json.JSONDecodeError as exc:
                raise QueryPlanningAuditError(
                    f"invalid source record at line {line_number}:invalid_{key}"
                ) from exc
            if not isinstance(value, str):
                raise QueryPlanningAuditError(
                    f"invalid source record at line {line_number}:invalid_{key}"
                )
            selected[key] = value
        else:
            index = _skip_json_value(raw, index, line_number=line_number)
        index = _skip_whitespace(raw, index)
        if index < len(raw) and raw[index] == ",":
            index += 1
            continue
        if index < len(raw) and raw[index] == "}":
            index += 1
            break
        raise QueryPlanningAuditError(
            f"invalid source record at line {line_number}:missing_delimiter"
        )
    if raw[index:].strip():
        raise QueryPlanningAuditError(
            f"invalid source record at line {line_number}:trailing_content"
        )
    return selected


def _skip_json_value(raw: str, index: int, *, line_number: int) -> int:
    if index >= len(raw):
        raise QueryPlanningAuditError(
            f"invalid source record at line {line_number}:missing_value"
        )
    first = raw[index]
    if first == '"':
        return _skip_json_string(raw, index, line_number=line_number)
    if first in "[{":
        stack = ["]" if first == "[" else "}"]
        index += 1
        while index < len(raw) and stack:
            char = raw[index]
            if char == '"':
                index = _skip_json_string(raw, index, line_number=line_number)
                continue
            if char == "[":
                stack.append("]")
            elif char == "{":
                stack.append("}")
            elif char in "]}":
                if char != stack[-1]:
                    raise QueryPlanningAuditError(
                        f"invalid source record at line {line_number}:unbalanced_value"
                    )
                stack.pop()
            index += 1
        if stack:
            raise QueryPlanningAuditError(
                f"invalid source record at line {line_number}:unterminated_value"
            )
        return index
    end = index
    while end < len(raw) and raw[end] not in ",}]\t\r\n ":
        end += 1
    if end == index:
        raise QueryPlanningAuditError(
            f"invalid source record at line {line_number}:missing_value"
        )
    return end


def _skip_json_string(raw: str, index: int, *, line_number: int) -> int:
    index += 1
    while index < len(raw):
        char = raw[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    raise QueryPlanningAuditError(
        f"invalid source record at line {line_number}:unterminated_string"
    )


def _skip_whitespace(raw: str, index: int) -> int:
    while index < len(raw) and raw[index].isspace():
        index += 1
    return index


def load_query_only_manifest(path: Path) -> list[QueryOnlyRecord]:
    rows: list[QueryOnlyRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = QueryOnlyRecord.model_validate_json(raw_line)
            except ValidationError as exc:
                raise QueryPlanningAuditError(
                    f"invalid query-only manifest at line {line_number}"
                ) from exc
            if row.query_id in seen:
                raise QueryPlanningAuditError(f"duplicate query_id:{row.query_id}")
            seen.add(row.query_id)
            rows.append(row)
    if not rows:
        raise QueryPlanningAuditError("query-only manifest is empty")
    return rows


def build_planning_audit(
    manifest: Mapping[str, Any],
    *,
    measure_latency: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Build deterministic plans plus a non-regression latency sidecar."""

    _validate_manifest(manifest, require_baseline=False)
    input_path = _repo_path(manifest["query_input"]["path"])
    if sha256_file(input_path) != manifest["query_input"]["sha256"]:
        raise QueryPlanningAuditError("query-only input fingerprint drifted")
    prompt_path = _repo_path(manifest["prompt_state"]["manifest_path"])
    if sha256_file(prompt_path) != manifest["prompt_state"]["manifest_sha256"]:
        raise QueryPlanningAuditError("prompt manifest fingerprint drifted")
    rows = load_query_only_manifest(input_path)
    if len(rows) != int(manifest["dataset"]["case_count"]):
        raise QueryPlanningAuditError("query-only input case count drifted")
    config = dict(manifest["plan_config"])
    snapshot_before = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    attempts = {"network": 0, "llm": 0}
    plan_rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    with _forbid_external_calls(attempts):
        for index, row in enumerate(rows):
            started = time.perf_counter_ns()
            result = _plan_one(index, row, config)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            plan_rows.append(result)
            timings.append(
                {
                    "index": index,
                    "query_id": row.query_id,
                    "latency_ms": elapsed_ms if measure_latency else 0.0,
                    "query_length": len(row.query),
                    "subquery_count": int(result.get("quality", {}).get("subquery_count", 0)),
                    "status": result["status"],
                }
            )
    snapshot_after = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    execution = {
        "network_request_count": attempts["network"],
        "llm_request_count": attempts["llm"],
        "snapshot_write_count": int(snapshot_before != snapshot_after),
        "connector_invoked": False,
        "evaluator_invoked": False,
        "gold_fields_accessed": False,
    }
    if any(execution[key] for key in (
        "network_request_count",
        "llm_request_count",
        "snapshot_write_count",
    )):
        raise QueryPlanningAuditError(f"offline execution invariant failed:{execution}")
    summary = _summarize(plan_rows, manifest, execution)
    runtime = _runtime_summary(timings)
    return plan_rows, summary, runtime


def check_planning_regression(
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest, require_baseline=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary, runtime = build_planning_audit(manifest)
    plans_path = output_dir / "plans.jsonl"
    summary_path = output_dir / "summary.json"
    runtime_path = output_dir / "runtime.json"
    _write_jsonl(plans_path, rows)
    _write_json(summary_path, summary)
    _write_json(runtime_path, runtime)

    baseline_plans_path = _repo_path(manifest["baseline"]["plans_path"])
    baseline_summary_path = _repo_path(manifest["baseline"]["summary_path"])
    expected_rows = _read_jsonl(baseline_plans_path)
    expected_summary = _read_json(baseline_summary_path)
    drifts: list[dict[str, Any]] = []
    drifts.extend(_fingerprint_drifts(manifest, manifest_path))
    drifts.extend(compare_profiles(expected_rows, rows, max_diffs=100))
    summary_drifts = compare_profiles(expected_summary, summary, max_diffs=100)
    for drift in summary_drifts:
        drift["path"] = drift["path"].replace("$", "$.summary", 1)
    drifts.extend(summary_drifts)
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "passed": not drifts,
        "case_count": len(rows),
        "success_count": int(summary["terminal_counts"].get("success", 0)),
        "error_count": int(summary["terminal_counts"].get("error", 0)),
        "drift_count": len(drifts),
        "drifts": drifts[:100],
        "plans_sha256": sha256_file(plans_path),
        "summary_sha256": sha256_file(summary_path),
        "manifest_sha256": sha256_file(manifest_path),
        "execution": summary["execution"],
        "official_effectiveness_score": False,
    }
    _write_json(output_dir / "regression_report.json", report)
    return report


def propose_planning_baseline(
    manifest_path: Path,
    output_dir: Path,
    *,
    approval_token: str,
    reason: str,
) -> dict[str, Any]:
    """Create review-only artifacts; never mutate tracked baselines."""

    if approval_token != BASELINE_APPROVAL_TOKEN:
        raise QueryPlanningAuditError("baseline proposal approval token rejected")
    if len(reason.strip()) < 12:
        raise QueryPlanningAuditError("baseline proposal reason is too short")
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest, require_baseline=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, summary, runtime = build_planning_audit(manifest)
    plans_path = output_dir / "proposed_plans.jsonl"
    summary_path = output_dir / "proposed_summary.json"
    _write_jsonl(plans_path, rows)
    _write_json(summary_path, summary)
    _write_json(output_dir / "runtime.json", runtime)
    hash_rows = [
        {"query_id": row["query_id"], "plan_sha256": row["plan_sha256"]}
        for row in rows
    ]
    hashes_path = output_dir / "proposed_plan_hashes.jsonl"
    _write_jsonl(hashes_path, hash_rows)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "reason": reason.strip(),
        "approval_token_verified": True,
        "tracked_files_mutated": False,
        "proposed_plans_sha256": sha256_file(plans_path),
        "proposed_summary_sha256": sha256_file(summary_path),
        "proposed_plan_hashes_sha256": sha256_file(hashes_path),
        "query_plan_hash_count": len(hash_rows),
    }
    _write_json(output_dir / "baseline_update_audit.json", audit)
    return audit


def _plan_one(
    index: int,
    row: QueryOnlyRecord,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "index": index,
        "query_id": row.query_id,
        "input_query_sha256": sha256_text(row.query),
    }
    try:
        plan = analyze_query(
            row.query,
            top_k=int(config["top_k"]),
            run_profile=str(config["run_profile"]),
            enable_refchain=False,
            enable_semantic_seed_expansion=False,
            enable_query_evolution=False,
            query_planning_policy="current_rules",
            current_year=int(config["effective_current_year"]),
            use_llm=False,
        )
        plan = _apply_sources(plan, list(config["sources"]))
        payload = plan.model_dump(mode="json")
        missing_fields = sorted(REQUIRED_PLAN_FIELDS - set(payload))
        round_trip = SearchPlan.model_validate_json(stable_json(payload))
        round_payload = round_trip.model_dump(mode="json")
        if payload != round_payload:
            raise QueryPlanningAuditError("schema round-trip changed plan")
        quality = _quality(plan, payload, config, missing_fields)
        plan_sha = sha256_json(payload)
        return {
            **base,
            "status": "success",
            "error": None,
            "plan_sha256": plan_sha,
            "plan": payload,
            "quality": quality,
        }
    except Exception as exc:  # noqa: BLE001 - an error row is an audit terminal
        return {
            **base,
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "code": _stable_error_code(exc),
            },
            "plan_sha256": None,
            "plan": None,
            "quality": {
                "schema_valid": False,
                "serialization_valid": False,
                "budget_consistent": False,
                "subquery_count": 0,
                "empty_subquery_count": 0,
                "duplicate_subquery_count": 0,
                "missing_plan_fields": [],
            },
        }


def _quality(
    plan: SearchPlan,
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    missing_fields: list[str],
) -> dict[str, Any]:
    query_keys = [item.query.strip().casefold() for item in plan.subqueries]
    empty_count = sum(not key for key in query_keys)
    duplicate_count = len(query_keys) - len(set(query_keys))
    max_subqueries = int(config["max_subqueries"])
    sources = list(config["sources"])
    priorities = [item.priority for item in plan.subqueries]
    budget_consistent = all(
        (
            len(plan.subqueries) <= max_subqueries,
            plan.selected_sources == sources,
            all(item.source_hints == sources for item in plan.subqueries),
            plan.limit_per_source == int(config["limit_per_source"]),
            priorities == list(range(1, len(priorities) + 1)),
            empty_count == 0,
            duplicate_count == 0,
            not missing_fields,
        )
    )
    constraints = plan.query_analysis.constraints
    return {
        "schema_valid": True,
        "serialization_valid": bool(stable_json(payload)),
        "budget_consistent": budget_consistent,
        "subquery_count": len(plan.subqueries),
        "planned_source_query_slots": sum(
            len(item.source_hints) for item in plan.subqueries
        ),
        "planned_result_capacity_before_global_budget": sum(
            len(item.source_hints) * plan.limit_per_source
            for item in plan.subqueries
        ),
        "global_candidate_budget": int(config["budgets"]["max_candidate_papers"]),
        "empty_subquery_count": empty_count,
        "duplicate_subquery_count": duplicate_count,
        "missing_plan_fields": missing_fields,
        "external_explicit_constraint_fields": list(constraints.explicit_fields),
        "text_extracted_constraint_fields": _constraint_fields(constraints),
    }


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    terminal_counts = Counter(str(row["status"]) for row in rows)
    successful = [row for row in rows if row["status"] == "success"]
    subquery_counts = Counter(
        int(row["quality"]["subquery_count"]) for row in successful
    )
    constraint_counts: Counter[str] = Counter()
    purposes: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    query_lengths: list[int] = []
    for row in successful:
        plan = row["plan"]
        query_lengths.append(len(str(plan["query_analysis"]["original_query"])))
        constraint_counts.update(row["quality"]["text_extracted_constraint_fields"])
        purposes.update(item["purpose"] for item in plan["subqueries"])
        warning_counts.update(plan["warnings"])
    quality = {
        "schema_valid_count": sum(row["quality"]["schema_valid"] for row in rows),
        "serialization_valid_count": sum(
            row["quality"]["serialization_valid"] for row in rows
        ),
        "budget_consistent_count": sum(
            row["quality"]["budget_consistent"] for row in rows
        ),
        "empty_subquery_total": sum(
            row["quality"]["empty_subquery_count"] for row in rows
        ),
        "duplicate_subquery_total": sum(
            row["quality"]["duplicate_subquery_count"] for row in rows
        ),
        "missing_plan_field_total": sum(
            len(row["quality"]["missing_plan_fields"]) for row in rows
        ),
        "external_explicit_constraint_case_count": sum(
            bool(row["quality"].get("external_explicit_constraint_fields"))
            for row in successful
        ),
        "text_extracted_constraint_case_count": sum(
            bool(row["quality"].get("text_extracted_constraint_fields"))
            for row in successful
        ),
        "subquery_total": sum(
            int(row["quality"]["subquery_count"]) for row in successful
        ),
        "planned_source_query_slot_total": sum(
            int(row["quality"]["planned_source_query_slots"])
            for row in successful
        ),
        "planned_result_capacity_before_global_budget_total": sum(
            int(row["quality"]["planned_result_capacity_before_global_budget"])
            for row in successful
        ),
        "cases_above_global_candidate_budget": sum(
            int(row["quality"]["planned_result_capacity_before_global_budget"])
            > int(row["quality"]["global_candidate_budget"])
            for row in successful
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": GATE_NAME,
        "dataset": manifest["dataset"],
        "query_input_sha256": manifest["query_input"]["sha256"],
        "plan_config_sha256": sha256_json(manifest["plan_config"]),
        "prompt_manifest_sha256": manifest["prompt_state"]["manifest_sha256"],
        "query_planner_version": QUERY_PLANNER_VERSION,
        "case_count": len(rows),
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "quality": quality,
        "subquery_count_distribution": {
            str(key): value for key, value in sorted(subquery_counts.items())
        },
        "subquery_purpose_counts": dict(sorted(purposes.items())),
        "text_extracted_constraint_field_counts": dict(
            sorted(constraint_counts.items())
        ),
        "warning_counts": dict(sorted(warning_counts.items())),
        "query_length": _distribution(query_lengths),
        "per_query_plan_hashes_sha256": sha256_json(
            [
                {"query_id": row["query_id"], "plan_sha256": row["plan_sha256"]}
                for row in rows
            ]
        ),
        "execution": dict(execution),
        "latency": {
            "included_in_regression": False,
            "artifact": "runtime.json",
            "reason": "wall_clock_is_nondeterministic",
        },
        "effectiveness_metrics_generated": False,
        "official_score": False,
    }


def _runtime_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "excluded_from_regression": True,
        "reason": "wall_clock_is_nondeterministic",
        "case_count": len(rows),
        "latency_ms": {
            "min": min(latencies, default=0.0),
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies, default=0.0),
        },
        "per_query": list(rows),
        "long_tail": sorted(
            rows,
            key=lambda item: (-float(item["latency_ms"]), int(item["index"])),
        )[:20],
    }


def _fingerprint_drifts(
    manifest: Mapping[str, Any], manifest_path: Path
) -> list[dict[str, Any]]:
    expected = {
        "query_input_sha256": manifest["query_input"]["sha256"],
        "prompt_manifest_sha256": manifest["prompt_state"]["manifest_sha256"],
        "baseline_plans_sha256": manifest["baseline"]["plans_sha256"],
        "baseline_summary_sha256": manifest["baseline"]["summary_sha256"],
    }
    actual = {
        "query_input_sha256": sha256_file(_repo_path(manifest["query_input"]["path"])),
        "prompt_manifest_sha256": sha256_file(
            _repo_path(manifest["prompt_state"]["manifest_path"])
        ),
        "baseline_plans_sha256": sha256_file(
            _repo_path(manifest["baseline"]["plans_path"])
        ),
        "baseline_summary_sha256": sha256_file(
            _repo_path(manifest["baseline"]["summary_path"])
        ),
    }
    drifts = compare_profiles(expected, actual)
    for drift in drifts:
        drift["path"] = drift["path"].replace("$", "$.fingerprints", 1)
    baseline_rows = _read_jsonl(_repo_path(manifest["baseline"]["plans_path"]))
    actual_plan_hashes = [
        {"query_id": row.get("query_id"), "plan_sha256": row.get("plan_sha256")}
        for row in baseline_rows
    ]
    expected_spec = manifest.get("expected") or {}
    expected_hash_path = _repo_path(expected_spec["query_plan_hashes_path"])
    expected_plan_hashes = _read_jsonl(expected_hash_path)
    if sha256_file(expected_hash_path) != expected_spec["query_plan_hashes_sha256"]:
        drifts.append(
            {
                "path": "$.fingerprints.query_plan_hashes_sha256",
                "kind": "value_changed",
                "expected": expected_spec["query_plan_hashes_sha256"],
                "actual": sha256_file(expected_hash_path),
            }
        )
    hash_drifts = compare_profiles(expected_plan_hashes, actual_plan_hashes)
    for drift in hash_drifts:
        drift["path"] = drift["path"].replace(
            "$", "$.fingerprints.query_plan_hashes", 1
        )
    drifts.extend(hash_drifts)
    expected_manifest_hash = manifest.get("self_sha256_excluding_field")
    if expected_manifest_hash and expected_manifest_hash != _manifest_semantic_sha(manifest):
        drifts.append(
            {
                "path": "$.fingerprints.manifest_semantic_sha256",
                "kind": "value_changed",
                "expected": expected_manifest_hash,
                "actual": _manifest_semantic_sha(manifest),
            }
        )
    return drifts


def _validate_manifest(manifest: Mapping[str, Any], *, require_baseline: bool) -> None:
    if manifest.get("gate") != GATE_NAME:
        raise QueryPlanningAuditError("unexpected planning regression manifest")
    config = manifest.get("plan_config") or {}
    required_config = {
        "query_planning_policy": "current_rules",
        "run_profile": "balanced",
        "top_k": 20,
        "sources": list(DEFAULT_SEARCH_SOURCES),
        "enable_llm": False,
        "enable_query_evolution": False,
        "enable_refchain": False,
        "enable_semantic_seed_expansion": False,
        "ranking_policy": "current_rules",
        "judgement_policy": "current_rules",
        "query_adapter_policy": "adaptive",
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise QueryPlanningAuditError(f"default planning config drift:{key}")
    if int(config.get("effective_current_year") or 0) < 1900:
        raise QueryPlanningAuditError("effective_current_year must be frozen")
    if int(config.get("max_subqueries") or 0) != 3:
        raise QueryPlanningAuditError("balanced max_subqueries must remain 3")
    if int(config.get("limit_per_source") or 0) != 20:
        raise QueryPlanningAuditError("balanced source limit must remain 20")
    if config.get("query_planner_version") != QUERY_PLANNER_VERSION:
        raise QueryPlanningAuditError("query planner version drifted")
    SearchBudget.model_validate(config.get("budgets"))
    features = config.get("experimental_features") or {}
    if not features or any(bool(value) for value in features.values()):
        raise QueryPlanningAuditError("all experimental features must remain disabled")
    if require_baseline:
        baseline = manifest.get("baseline") or {}
        for key in ("plans_path", "plans_sha256", "summary_path", "summary_sha256"):
            if not baseline.get(key):
                raise QueryPlanningAuditError(f"missing baseline field:{key}")
        expected = manifest.get("expected") or {}
        for key in (
            "query_plan_hashes_path",
            "query_plan_hashes_sha256",
            "query_plan_hash_count",
        ):
            if not expected.get(key):
                raise QueryPlanningAuditError(f"missing expected field:{key}")
        if int(expected["query_plan_hash_count"]) != int(
            manifest.get("dataset", {}).get("case_count") or 0
        ):
            raise QueryPlanningAuditError("per-query expected hashes are incomplete")


def _apply_sources(plan: SearchPlan, sources: list[str]) -> SearchPlan:
    subqueries = [
        item.model_copy(update={"source_hints": list(sources)})
        for item in plan.subqueries
    ]
    return plan.model_copy(
        update={"selected_sources": list(sources), "subqueries": subqueries}
    )


def _constraint_fields(constraints: Any) -> list[str]:
    fields: list[str] = []
    if constraints.time_range is not None:
        fields.append("time_range")
    for field in (
        "venues",
        "methods",
        "datasets",
        "domains",
        "must_include_terms",
        "exclude_terms",
        "paper_types",
    ):
        if getattr(constraints, field):
            fields.append(field)
    return fields


@contextmanager
def _forbid_external_calls(attempts: dict[str, int]) -> Iterator[None]:
    def reject_network(*_args: Any, **_kwargs: Any) -> Any:
        attempts["network"] += 1
        raise QueryPlanningAuditError("network access forbidden in planning audit")

    def reject_llm(*_args: Any, **_kwargs: Any) -> Any:
        attempts["llm"] += 1
        raise QueryPlanningAuditError("LLM access forbidden in planning audit")

    with (
        patch("socket.create_connection", reject_network),
        patch.object(socket.socket, "connect", reject_network),
        patch.object(socket, "getaddrinfo", reject_network),
        patch.object(query_understanding, "provider_chat_json", reject_llm),
    ):
        yield


def _tree_signature(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": min(ordered, default=0),
        "mean": sum(ordered) / len(ordered) if ordered else 0.0,
        "p50": _percentile([float(item) for item in ordered], 0.50),
        "p95": _percentile([float(item) for item in ordered], 0.95),
        "p99": _percentile([float(item) for item in ordered], 0.99),
        "max": max(ordered, default=0),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return float(values[index])


def _stable_error_code(exc: Exception) -> str:
    text = " ".join(str(exc).split()).casefold()
    if "empty" in text or "must not be empty" in text:
        return "empty_query"
    if "schema" in text or isinstance(exc, ValidationError):
        return "schema_error"
    if isinstance(exc, QueryPlanningAuditError):
        return "audit_invariant_error"
    return "planning_error"


def _manifest_semantic_sha(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("self_sha256_excluding_field", None)
    return sha256_json(payload)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return sha256_text(stable_json(value))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QueryPlanningAuditError(f"expected JSON object:{path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(stable_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
