"""Offline gate for ``result_lineage_v1`` field reconstruction.

The gate invokes production paper deduplication with its opt-in observation
hook.  It never loads gold, connectors, runtime configuration, or Snapshot
writers.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from scholar_agent.core.dedup import deduplicate_papers_with_lineage
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.result_lineage import (
    RESULT_LINEAGE_CONTRACT,
    RESULT_LINEAGE_SCHEMA_VERSION,
    ResultLineageDocument,
    restrict_result_lineage_document,
    stable_json_bytes,
    stable_sha256,
)
from scholar_agent.evaluation.execution_determinism import forbid_network, tree_signature
from scholar_agent.evaluation.snapshot_resume import sha256_file


GATE_NAME = "result_lineage_gate"
EXIT_PASSED = 0
EXIT_LINEAGE_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
DEFAULT_SNAPSHOT_ROOT = Path(__file__).resolve().parents[3] / "outputs" / "benchmark_snapshots"

_OPAQUE_QUERY_RE = re.compile(r"^query:[0-9a-f]{64}$")
_FORBIDDEN_OUTPUT_FRAGMENTS = (
    ".env",
    "authorization",
    "api_key",
    "bearer ",
    "/users/",
)


class ResultLineageError(RuntimeError):
    """The local fixture or lineage protocol is malformed."""


class ResultLineageNotEligible(ResultLineageError):
    """A frozen run lacks the evidence needed for field reconstruction."""


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    value = _load_object(path)
    if value.get("contract") != RESULT_LINEAGE_CONTRACT:
        raise ResultLineageError("protocol_contract_invalid")
    if value.get("schema_version") != RESULT_LINEAGE_SCHEMA_VERSION:
        raise ResultLineageError("protocol_schema_version_invalid")
    fixture = _mapping(value, "fixture")
    _verify_file_identity(fixture, repository_root, "fixture")
    legacy = _mapping(value, "frozen_baseline_eligibility")
    _verify_file_identity(legacy, repository_root, "legacy_audit")
    query_identity = str(value.get("query_identity") or "")
    if not _OPAQUE_QUERY_RE.fullmatch(query_identity):
        raise ResultLineageError("opaque_query_identity_invalid")
    if value.get("score_scope") != "lineage_only_not_quality_or_official_score":
        raise ResultLineageError("score_scope_invalid")
    return value


def run_result_lineage_gate(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    controlled_fault: str | None = None,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    before = tree_signature(snapshot_root)
    attempts = {"network": 0}
    with forbid_network(attempts):
        papers, terminals = _load_fixture(protocol, repository_root)
        production_papers, audit, document = deduplicate_papers_with_lineage(
            papers,
            query_identity=str(protocol["query_identity"]),
            source_terminals=terminals,
        )
        if controlled_fault == "field_injection":
            document = copy.deepcopy(document)
            document["results"][0]["field_decisions"][0]["selected_value"] = (
                "fabricated field value"
            )
        violations = validate_result_lineage_document(document)
    after = tree_signature(snapshot_root)
    if before != after:
        violations.append(
            _violation(
                invariant="snapshot_tree_immutable",
                path="execution.snapshot_tree",
                query_identity=str(protocol["query_identity"]),
            )
        )
    serialized = json.dumps(document, ensure_ascii=False, sort_keys=True).casefold()
    for fragment in _FORBIDDEN_OUTPUT_FRAGMENTS:
        if fragment in serialized:
            violations.append(
                _violation(
                    invariant="sensitive_or_machine_specific_content_forbidden",
                    path="lineage",
                    query_identity=str(protocol["query_identity"]),
                )
            )
            break
    result_count = len(document.get("results") or [])
    source_record_count = len(document.get("source_records") or [])
    report = {
        "schema_version": RESULT_LINEAGE_SCHEMA_VERSION,
        "contract": RESULT_LINEAGE_CONTRACT,
        "gate": GATE_NAME,
        "status": "passed" if not violations else "lineage_or_reconstruction_violation",
        "exit_code": EXIT_PASSED if not violations else EXIT_LINEAGE_VIOLATION,
        "score_scope": "lineage_only_not_quality_or_official_score",
        "query_identity": str(protocol["query_identity"]),
        "source_record_count": source_record_count,
        "source_terminal_counts": dict(
            sorted(Counter(item["status"] for item in terminals).items())
        ),
        "result_count": result_count,
        "accepted_merge_count": len(audit),
        "field_decision_count": sum(
            len(item.get("field_decisions") or [])
            for item in document.get("results") or []
        ),
        "observational_equivalence": {
            "production_results_sha256": stable_sha256(
                [paper.model_dump(mode="json") for paper in production_papers]
            ),
            "lineage_results_sha256": document.get("final_results_sha256"),
            "equal": stable_sha256(
                [paper.model_dump(mode="json") for paper in production_papers]
            )
            == document.get("final_results_sha256"),
        },
        "lineage_sha256": stable_sha256(document),
        "violation_count": len(violations),
        "violations": violations,
        "execution": {
            "network_request_count": attempts["network"],
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
            "controlled_fault": controlled_fault,
        },
    }
    return report


def validate_result_lineage_document(
    value: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct a lineage document from its registered source records."""

    try:
        document = ResultLineageDocument.model_validate(value)
    except ValidationError:
        return [
            _violation(
                invariant="lineage_schema_valid",
                path="$",
                query_identity=str(value.get("query_identity") or "unknown"),
            )
        ]
    violations: list[dict[str, Any]] = []
    if not _OPAQUE_QUERY_RE.fullmatch(document.query_identity):
        violations.append(
            _violation(
                invariant="opaque_query_identity",
                path="query_identity",
                query_identity=document.query_identity,
            )
        )
    ordered_records = sorted(document.source_records, key=lambda item: item.input_index)
    expected_indexes = list(range(len(ordered_records)))
    if [item.input_index for item in ordered_records] != expected_indexes:
        violations.append(
            _violation(
                invariant="source_record_order_contiguous",
                path="source_records.*.input_index",
                query_identity=document.query_identity,
            )
        )
        return violations
    refs = [item.record_ref for item in ordered_records]
    if len(refs) != len(set(refs)):
        violations.append(
            _violation(
                invariant="source_record_ref_unique",
                path="source_records.*.record_ref",
                query_identity=document.query_identity,
            )
        )
        return violations
    contributed_counts = Counter(
        source for record in ordered_records for source in record.sources
    )
    terminal_sources = {item.source for item in document.source_terminals}
    if set(contributed_counts) - terminal_sources:
        violations.append(
            _violation(
                invariant="every_contributing_source_has_terminal",
                path="source_terminals",
                query_identity=document.query_identity,
            )
        )
        return violations
    for index, terminal in enumerate(document.source_terminals):
        expected_count = contributed_counts.get(terminal.source, 0)
        if terminal.contributed_record_count != expected_count:
            violations.append(
                _violation(
                    invariant="source_terminal_contribution_count_matches",
                    path=f"source_terminals.{index}.contributed_record_count",
                    query_identity=document.query_identity,
                    expected=expected_count,
                    observed=terminal.contributed_record_count,
                )
            )
            return violations
    for index, record in enumerate(ordered_records):
        observed = stable_sha256(record.paper.model_dump(mode="json"))
        if observed != record.source_record_sha256:
            violations.append(
                _violation(
                    invariant="source_record_hash_matches",
                    path=f"source_records.{index}.source_record_sha256",
                    query_identity=document.query_identity,
                    result_identity=None,
                    field=None,
                    expected=record.source_record_sha256,
                    observed=observed,
                )
            )
            return violations
    terminals = [item.model_dump(mode="json") for item in document.source_terminals]
    _, _, expected = deduplicate_papers_with_lineage(
        [item.paper for item in ordered_records],
        query_identity=document.query_identity,
        source_terminals=terminals,
    )
    actual = document.model_dump(mode="json")
    expected_results = {
        item["result_identity"]: Paper.model_validate(item["final_result"])
        for item in expected["results"]
    }
    if document.final_result_order != [
        item["result_identity"] for item in expected["results"]
    ] and all(identity in expected_results for identity in document.final_result_order):
        expected = restrict_result_lineage_document(
            expected,
            [expected_results[identity] for identity in document.final_result_order],
        )
    difference = _first_difference(expected, actual)
    if difference is not None:
        path, expected_value, observed_value = difference
        result_index, field = _difference_context(path, actual)
        result_identity = None
        if result_index is not None and result_index < len(document.results):
            result_identity = document.results[result_index].result_identity
        violations.append(
            _violation(
                invariant="exact_reconstruction_from_registered_sources",
                path=path,
                query_identity=document.query_identity,
                result_identity=result_identity,
                field=field,
                expected=expected_value,
                observed=observed_value,
            )
        )
    return violations


def audit_frozen_baseline_eligibility(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    legacy = _mapping(protocol, "frozen_baseline_eligibility")
    audit = _load_object(repository_root / str(legacy["path"]))
    wanted = {
        "autoscholar_full1000_frozen_baseline",
        "autoscholar_record160_analysis_input",
    }
    profiles = []
    for item in audit.get("profiles") or []:
        profile_id = str(item.get("profile_id"))
        if profile_id not in wanted:
            continue
        profiles.append(
            {
                "profile_id": profile_id,
                "status": "not_eligible",
                "reason": "field_level_candidate_and_merge_lineage_unavailable",
                "observed_record_count": item.get("observed_record_count"),
                "historical_artifacts_modified": False,
            }
        )
    if {item["profile_id"] for item in profiles} != wanted:
        raise ResultLineageNotEligible("frozen_profile_evidence_missing")
    return {
        "schema_version": RESULT_LINEAGE_SCHEMA_VERSION,
        "contract": RESULT_LINEAGE_CONTRACT,
        "gate": GATE_NAME,
        "status": "not_eligible",
        "exit_code": EXIT_NOT_ELIGIBLE,
        "score_scope": "lineage_only_not_quality_or_official_score",
        "profile_count": len(profiles),
        "profiles": sorted(profiles, key=lambda item: item["profile_id"]),
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "quality_metric_count": 0,
        },
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(value))


def _load_fixture(
    protocol: Mapping[str, Any], repository_root: Path
) -> tuple[list[Paper], list[dict[str, Any]]]:
    fixture = _mapping(protocol, "fixture")
    payload = _load_object(repository_root / str(fixture["path"]))
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ResultLineageError("fixture_sources_invalid")
    papers: list[Paper] = []
    terminals: list[dict[str, Any]] = []
    for source in sorted(sources):
        values = sources[source]
        if not isinstance(values, list):
            raise ResultLineageError("fixture_source_records_invalid")
        source_papers = [Paper.model_validate(item) for item in values]
        if any(source not in paper.sources for paper in source_papers):
            raise ResultLineageError("fixture_source_record_mismatch")
        papers.extend(source_papers)
        status_overrides = protocol.get("source_terminal_overrides") or {}
        status = str(status_overrides.get(source) or ("success" if values else "success_empty"))
        terminals.append(
            {
                "source": source,
                "status": status,
                "reason": (
                    "fixture_partial_source_failure"
                    if status == "partial_completion"
                    else None
                ),
                "contributed_record_count": len(values),
            }
        )
    return papers, terminals


def _first_difference(
    expected: Any, observed: Any, path: str = "$"
) -> tuple[str, Any, Any] | None:
    if type(expected) is not type(observed):
        return path, expected, observed
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(observed)):
            child = f"{path}.{key}"
            if key not in expected:
                return child, None, observed[key]
            if key not in observed:
                return child, expected[key], None
            difference = _first_difference(expected[key], observed[key], child)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}.length", len(expected), len(observed)
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            difference = _first_difference(left, right, f"{path}.{index}")
            if difference is not None:
                return difference
        return None
    if expected != observed:
        return path, expected, observed
    return None


def _difference_context(
    path: str, document: Mapping[str, Any]
) -> tuple[int | None, str | None]:
    match = re.search(r"\$\.results\.(\d+)", path)
    if match is None:
        return None, None
    result_index = int(match.group(1))
    field_match = re.search(r"\.field_decisions\.(\d+)", path)
    if field_match is None:
        return result_index, None
    decisions = document["results"][result_index]["field_decisions"]
    decision_index = int(field_match.group(1))
    return result_index, str(decisions[decision_index]["field"])


def _violation(
    *,
    invariant: str,
    path: str,
    query_identity: str,
    result_identity: str | None = None,
    field: str | None = None,
    expected: Any = None,
    observed: Any = None,
) -> dict[str, Any]:
    return {
        "query_identity": query_identity,
        "result_identity": result_identity,
        "field": field,
        "invariant": invariant,
        "first_difference_path": path,
        "expected_sha256": stable_sha256(expected),
        "observed_sha256": stable_sha256(observed),
    }


def _verify_file_identity(
    value: Mapping[str, Any], repository_root: Path, label: str
) -> None:
    path = repository_root / str(value.get("path") or "")
    if not path.is_file():
        raise ResultLineageNotEligible(f"{label}_missing")
    if path.stat().st_size != int(value.get("size_bytes", path.stat().st_size)):
        raise ResultLineageNotEligible(f"{label}_size_mismatch")
    if sha256_file(path) != str(value.get("sha256") or ""):
        raise ResultLineageNotEligible(f"{label}_hash_mismatch")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResultLineageError("json_object_required")
    return value


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ResultLineageError(f"{key}_object_required")
    return item
