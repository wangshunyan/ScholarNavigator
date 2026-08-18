"""Gold-free Top-20 delivery fidelity audit for frozen Record160 Replay.

The audit reconstructs ``final_returned`` through the production retrieval,
canonicalization, deduplication, Judgement, rerank, and selection path.  It then
exercises the real public API mapper and serialization schemas.  It never
invents an export surface: absent CSV/table exporters and legacy runs without a
reproduction capsule are reported explicitly as unsupported or not eligible.
"""

from __future__ import annotations

import csv
import io
import json
import socket
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch
from urllib.parse import urlsplit

from scholar_agent.agents.reranker import (
    DEFAULT_TIEBREAK_POLICY,
    deterministic_tiebreak_v2_catalog,
)
from scholar_agent.core import api_schemas as api
from scholar_agent.core.dedup import deduplicate_papers_with_lineage
from scholar_agent.core.result_lineage import (
    ranked_result_authority_digest,
    result_identity,
)
from scholar_agent.core.search_schemas import QueryAnalysis, RankedPaper
from scholar_agent.evaluation.constraint_decision_audit import (
    _load_component_assignments,
    _opaque_identity,
    _read_json,
    _read_rows,
    _repo_path,
    _sha256,
    _stable_json_sha256,
    _validate_config,
    _validate_file_hash,
    _validate_population,
)
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    align_papers_to_diagnostics,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.relevance_filter_audit import _tree_sha256
from scholar_agent.evaluation.source_fusion_ablation import (
    rank_variant,
    validate_full_reconstruction,
)
from scholar_agent.evaluation.source_reliability_diagnostics import (
    audit_retrieval_requests,
)
from scholar_agent.evaluation.snapshots import SnapshotStore
from scholar_agent.services.api_mapper import map_final_ranked_papers


SCHEMA_VERSION = "1"
CONTRACT_VERSION = "top20_delivery_fidelity_v1"
EXIT_PASSED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class Top20DeliveryError(RuntimeError):
    """A delivery contract or round-trip invariant was violated."""


class Top20DeliveryNotEligible(Top20DeliveryError):
    """Frozen evidence cannot support authoritative reconstruction."""


def load_contract(path: str | Path) -> dict[str, Any]:
    value = _read_json(Path(path).expanduser().resolve())
    if value.get("analysis") != CONTRACT_VERSION or value.get("schema_version") != "1":
        raise Top20DeliveryError("unsupported_contract")
    if value.get("execution") != {
        "gold_access": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }:
        raise Top20DeliveryError("offline_contract_drift")
    if value.get("analysis_population", {}).get("selection_prohibitions") != [
        "gold",
        "qrels",
        "case_id",
        "target_paper",
        "quality_score",
        "observed_delivery_result",
    ]:
        raise Top20DeliveryError("selection_contract_drift")
    authority = value.get("authoritative_input") or {}
    if authority.get("stage") != "final_returned":
        raise Top20DeliveryError("authoritative_stage_drift")
    if authority.get("operation_order") != (
        "rerank_all_then_take_first_20_then_apply_category_gate"
    ):
        raise Top20DeliveryError("selection_operation_order_drift")
    if int(authority.get("maximum_results_per_query") or 0) != 20:
        raise Top20DeliveryError("top_k_contract_drift")
    if DEFAULT_TIEBREAK_POLICY != "original_index_v1":
        raise Top20DeliveryError("production_default_tiebreak_drift")
    v2 = deterministic_tiebreak_v2_catalog()
    frozen_v2 = value.get("policy_isolation", {}).get("deterministic_tiebreak_v2") or {}
    if frozen_v2.get("default_enabled") is not False or v2["default_enabled"] is not False:
        raise Top20DeliveryError("deterministic_tiebreak_v2_enabled")
    return value


def run_top20_delivery_fidelity(
    contract_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the frozen audit with network and evaluator access blocked."""

    root = Path(repository_root).expanduser().resolve()
    contract_file = Path(contract_path).expanduser().resolve()
    contract = load_contract(contract_file)
    frozen = contract["frozen_input"]
    run_dir = _repo_path(root, frozen["run_dir"])
    snapshot_dir = _repo_path(root, frozen["snapshot_dir"])
    config_path = run_dir / "config.json"
    results_path = run_dir / "results.jsonl"
    assignments_path = _repo_path(root, frozen["component_assignments"]["path"])
    _validate_file_hash(config_path, frozen["config_sha256"])
    _validate_file_hash(results_path, frozen["record_results_sha256"])
    _validate_file_hash(assignments_path, frozen["component_assignments"]["sha256"])
    before_tree = _tree_sha256(snapshot_dir)
    if before_tree != frozen["snapshot_tree_sha256"]:
        raise Top20DeliveryNotEligible("snapshot_tree_hash_drift")
    if sum(path.is_file() for path in snapshot_dir.rglob("*")) != int(
        frozen["snapshot_file_count"]
    ):
        raise Top20DeliveryNotEligible("snapshot_file_count_drift")

    config = _read_json(config_path)
    _validate_config(config, contract)
    rows = _read_rows(results_path)
    if len(rows) != int(contract["analysis_population"]["record_case_count"]):
        raise Top20DeliveryNotEligible("record_case_count_drift")
    configured_order = [str(value) for value in config.get("case_ids") or []]
    row_order = [str(row["case_id"]) for row in rows]
    if row_order != configured_order[: len(rows)]:
        raise Top20DeliveryNotEligible("record_prefix_or_order_drift")
    components = _load_component_assignments(assignments_path)
    if any(case_id not in components for case_id in row_order):
        raise Top20DeliveryNotEligible("missing_component_assignment")

    frontend = audit_frontend_contract(root)
    export_eligibility = audit_export_eligibility(root, run_dir)
    store = SnapshotStore(snapshot_dir)
    attempts = {"network": 0}
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    observed_keys: set[str] = set()
    with _forbid_network(attempts):
        for case_order, row in enumerate(rows):
            case, results, keys = analyze_case(
                row,
                config=config,
                contract=contract,
                store=store,
                component_id=components[str(row["case_id"])],
                case_order=case_order,
            )
            observed_keys.update(keys)
            if case["analysis_status"] == "excluded_no_successful_source":
                excluded.append(case)
            else:
                included.append(case)
                result_rows.extend(results)

    _validate_population(included, excluded, contract)
    if len(result_rows) != int(
        contract["analysis_population"]["expected_final_result_count"]
    ):
        raise Top20DeliveryNotEligible("final_result_population_drift")
    if len(observed_keys) != int(frozen["observed_snapshot_key_count"]):
        raise Top20DeliveryNotEligible("observed_snapshot_key_count_drift")
    if _tree_sha256(snapshot_dir) != before_tree:
        raise Top20DeliveryError("snapshot_tree_changed")
    if attempts["network"]:
        raise Top20DeliveryError("network_attempt_detected")

    cases = sorted([*included, *excluded], key=lambda item: int(item["case_order"]))
    aggregate = aggregate_analysis(
        included,
        excluded,
        result_rows,
        contract,
        frontend=frontend,
        export_eligibility=export_eligibility,
        contract_sha256=_sha256(contract_file),
        input_hashes={
            "component_assignments_sha256": _sha256(assignments_path),
            "config_sha256": _sha256(config_path),
            "record_results_sha256": _sha256(results_path),
            "snapshot_tree_sha256": before_tree,
        },
        observed_snapshot_key_count=len(observed_keys),
    )
    return cases, result_rows, aggregate


def analyze_case(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    store: Any,
    component_id: str,
    case_order: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    stages = {
        str(item.get("stage")): item
        for item in row["stage_diagnostics"]["snapshots"]
    }
    required = set(contract["reconstruction"]["required_exact_stages"]) | {
        "initial_retrieval"
    }
    if not required.issubset(stages):
        raise Top20DeliveryNotEligible("required_frozen_stage_missing")
    sources = [str(value) for value in config["sources"]]
    requests = audit_retrieval_requests(
        stages["initial_retrieval"], config=config, store=store, sources=sources
    )
    successful_source_count = sum(
        int(requests.source_records[source]["snapshot_success_count"]) > 0
        for source in sources
    )
    query_identity = _opaque_identity("query", str(row["case_id"]))
    base = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": (
            "included_main_analysis"
            if successful_source_count
            else "excluded_no_successful_source"
        ),
        "case_order": case_order,
        "query_identity": query_identity,
        "component_identity": _opaque_identity("component", str(component_id)),
        "successful_source_count": successful_source_count,
    }
    if not successful_source_count:
        return base, [], requests.observed_keys

    analysis = QueryAnalysis.model_validate(row["query_analysis"])
    raw = [
        paper.model_copy(deep=True)
        for _source, batch in requests.ordered_batches
        for paper in batch
    ]
    deduplicated, _dedup_audit, lineage = deduplicate_papers_with_lineage(
        raw, query_identity=query_identity
    )
    candidates = list(deduplicated)
    limit = int(contract["reconstruction"]["candidate_limit"])
    if len(candidates) > limit:
        candidates = stable_source_coverage_truncate(
            candidates, limit=limit, source_order=sources
        )
    candidates = align_papers_to_diagnostics(
        candidates, stages["initial_deduplicated"]["candidates"]
    )
    top_k = int(contract["reconstruction"]["top_k"])
    full = rank_variant(analysis, candidates, top_k=top_k)
    validate_full_reconstruction(full, stages)
    if len(full.returned) > top_k:
        raise Top20DeliveryError("authoritative_top20_overflow")

    mapped = map_final_ranked_papers(list(full.ranked[:top_k]))
    validate_authority_mapping(full.returned, mapped, query_identity=query_identity)
    canonical = delivery_projection(mapped)
    json_roundtrip = roundtrip_json(mapped)
    jsonl_roundtrip = roundtrip_jsonl(mapped)
    assert_same_delivery(canonical, json_roundtrip, export_name="json", query_identity=query_identity)
    assert_same_delivery(canonical, jsonl_roundtrip, export_name="jsonl", query_identity=query_identity)
    for page_size in (1, 7, 20):
        paged = paginate_delivery(canonical, page_size=page_size)
        assert_same_delivery(
            canonical,
            paged,
            export_name=f"pagination:{page_size}",
            query_identity=query_identity,
        )
    validate_frontend_keys(canonical, query_identity=query_identity)

    authority_by_identity = {
        result_identity(item.paper): item for item in full.returned
    }
    lineage_by_identity = {
        str(item["result_identity"]): item for item in lineage["results"]
    }
    results: list[dict[str, Any]] = []
    unsafe_url_count = 0
    for position, value in enumerate(canonical, start=1):
        identity = str(value["result_identity"])
        internal = authority_by_identity[identity]
        raw_urls = internal.paper.urls.model_dump(mode="json")
        mapped_urls = value["paper"]["urls"]
        unsafe_fields = [
            field
            for field, raw_url in raw_urls.items()
            if raw_url is not None and not is_safe_clickable_url(str(raw_url))
        ]
        unsafe_url_count += len(unsafe_fields)
        if any(mapped_urls[field] is not None for field in unsafe_fields):
            raise Top20DeliveryError(
                f"unsafe_url_remained_clickable:{query_identity}:{position}"
            )
        lineage_item = lineage_by_identity.get(identity)
        if lineage_item is None:
            raise Top20DeliveryError(
                f"final_result_lineage_missing:{query_identity}:{position}"
            )
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_order": case_order,
                "query_identity": query_identity,
                "result_position": position,
                "result_identity": identity,
                "authority_digest": value["authority_digest"],
                "delivery_digest": _stable_json_sha256(value),
                "field_lineage_sha256": _stable_json_sha256(lineage_item),
                "null_fields": sorted(null_field_paths(value)),
                "empty_string_fields": sorted(empty_string_field_paths(value)),
                "unsafe_url_fields": unsafe_fields,
                "frontend_key": identity,
            }
        )
    return (
        {
            **base,
            "authoritative_result_count": len(canonical),
            "authoritative_result_digest": _stable_json_sha256(canonical),
            "delivery": {
                "api_exact": True,
                "frontend_key_unique": True,
                "json_exact": True,
                "jsonl_exact": True,
                "pagination_profiles_exact": [1, 7, 20],
                "unsafe_url_field_count": unsafe_url_count,
            },
            "reconstruction": {
                "initial_deduplicated_exact": True,
                "initial_judged_exact": True,
                "initial_reranked_exact": True,
                "final_returned_exact": True,
            },
        },
        results,
        requests.observed_keys,
    )


def validate_authority_mapping(
    authoritative: Sequence[RankedPaper],
    mapped: Sequence[api.RankedPaper],
    *,
    query_identity: str,
) -> None:
    if len(authoritative) != len(mapped):
        raise Top20DeliveryError(
            f"api_count_drift:{query_identity}:expected={len(authoritative)}:actual={len(mapped)}"
        )
    for position, (source, delivered) in enumerate(
        zip(authoritative, mapped, strict=True), start=1
    ):
        expected_identity = result_identity(source.paper)
        expected_digest = ranked_result_authority_digest(source)
        for field, expected, actual in (
            ("result_identity", expected_identity, delivered.result_identity),
            ("authority_digest", expected_digest, delivered.authority_digest),
            ("rank", source.rank, delivered.rank),
            ("category", source.category, delivered.category),
        ):
            if expected != actual:
                raise Top20DeliveryError(
                    f"api_field_drift:{query_identity}:{position}:{field}"
                )


def delivery_projection(values: Sequence[api.RankedPaper | Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the canonical public delivery document without reordering it."""

    return [
        api.RankedPaper.model_validate(value).model_dump(mode="json")
        for value in values
    ]


def roundtrip_json(values: Sequence[api.RankedPaper]) -> list[dict[str, Any]]:
    payload = json.dumps(
        [value.model_dump(mode="json") for value in values],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    parsed = json.loads(payload)
    return delivery_projection(parsed)


def roundtrip_jsonl(values: Sequence[api.RankedPaper]) -> list[dict[str, Any]]:
    payload = "".join(
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for value in values
    )
    return delivery_projection(
        [json.loads(line) for line in payload.splitlines() if line]
    )


def paginate_delivery(
    values: Sequence[Mapping[str, Any]], *, page_size: int
) -> list[dict[str, Any]]:
    if page_size < 1:
        raise Top20DeliveryError("invalid_page_size")
    pages = [values[index : index + page_size] for index in range(0, len(values), page_size)]
    return [dict(item) for page in pages for item in page]


def assert_same_delivery(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    *,
    export_name: str,
    query_identity: str,
) -> None:
    if len(expected) != len(actual):
        raise Top20DeliveryError(
            f"delivery_count_drift:{export_name}:{query_identity}"
        )
    for position, (left, right) in enumerate(zip(expected, actual, strict=True), start=1):
        difference = first_difference(left, right)
        if difference is not None:
            raise Top20DeliveryError(
                f"delivery_roundtrip_drift:{export_name}:{query_identity}:{position}:{difference}"
            )


def validate_frontend_keys(
    values: Sequence[Mapping[str, Any]], *, query_identity: str
) -> None:
    keys = [str(item.get("result_identity") or "") for item in values]
    if any(not key for key in keys):
        raise Top20DeliveryError(f"frontend_key_missing:{query_identity}")
    if len(keys) != len(set(keys)):
        raise Top20DeliveryError(f"frontend_key_collision:{query_identity}")


def audit_frontend_contract(repository_root: Path) -> dict[str, Any]:
    helper = repository_root / "frontend/src/lib/top20-delivery.ts"
    component = repository_root / "frontend/src/components/scholar-navigator-app.tsx"
    formatter = repository_root / "frontend/src/lib/format.ts"
    for path in (helper, component, formatter):
        if not path.is_file():
            raise Top20DeliveryNotEligible(f"frontend_contract_file_missing:{path.name}")
    helper_text = helper.read_text(encoding="utf-8")
    component_text = component.read_text(encoding="utf-8")
    formatter_text = formatter.read_text(encoding="utf-8")
    required_fragments = {
        "helper_version": "frontend_top20_delivery_v1",
        "stable_key": "return paper.result_identity;",
        "component_key": "key={top20PaperKey(paper)}",
        "direct_mapping": "papers.map((paper) =>",
        "url_allowlist": '["http:", "https:"]',
    }
    observed = {
        "helper_version": helper_text,
        "stable_key": helper_text,
        "component_key": component_text,
        "direct_mapping": component_text,
        "url_allowlist": formatter_text,
    }
    missing = [
        name
        for name, fragment in required_fragments.items()
        if fragment not in observed[name]
    ]
    if missing:
        raise Top20DeliveryError("frontend_contract_drift:" + ",".join(missing))
    return {
        "status": "passed",
        "version": "frontend_top20_delivery_v1",
        "key": "result_identity",
        "source_files": {
            str(path.relative_to(repository_root)): {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in (helper, component, formatter)
        },
    }


def audit_export_eligibility(repository_root: Path, run_dir: Path) -> dict[str, Any]:
    frontend_export = repository_root / "frontend/src/lib/export.ts"
    batch_export = repository_root / "scripts/run_search_batch.py"
    if not frontend_export.is_file() or not batch_export.is_file():
        raise Top20DeliveryNotEligible("declared_json_export_missing")
    export_text = frontend_export.read_text(encoding="utf-8")
    batch_text = batch_export.read_text(encoding="utf-8")
    if "JSON.stringify(result, null, 2)" not in export_text:
        raise Top20DeliveryError("frontend_json_export_drift")
    if "json.dumps(result, ensure_ascii=False)" not in batch_text:
        raise Top20DeliveryError("batch_jsonl_export_drift")
    capsule_missing = [
        name
        for name, path in (
            ("run_manifest_v1", run_dir / "run_manifest.json"),
            ("committed_generation_chain", run_dir / ".run_commits"),
        )
        if not path.exists()
    ]
    return {
        "backend_public_api": {"status": "eligible"},
        "json": {"status": "eligible"},
        "jsonl": {"status": "eligible"},
        "frontend_display": {"status": "eligible"},
        "csv_table": {
            "status": "unsupported_export",
            "reason": "no_existing_production_csv_or_table_result_export",
        },
        "reproduction_capsule": {
            "status": "not_eligible" if capsule_missing else "eligible",
            "reason": (
                "legacy_metadata_incomplete:" + ",".join(capsule_missing)
                if capsule_missing
                else None
            ),
        },
    }


def csv_formula_safe_cell(value: str) -> str:
    """Return the RFC4180 cell value a future registered CSV export must use.

    This helper is validation-only.  Its presence does not create or claim a
    production CSV export.
    """

    if value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def csv_roundtrip_row(values: Sequence[str]) -> list[str]:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, dialect="excel", lineterminator="\r\n")
    writer.writerow([csv_formula_safe_cell(value) for value in values])
    buffer.seek(0)
    return next(csv.reader(buffer, dialect="excel"))


def is_safe_clickable_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return bool(
        parsed.scheme.casefold() in {"http", "https"}
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
    )


def null_field_paths(value: Any, path: str = "$") -> list[str]:
    if value is None:
        return [path]
    if isinstance(value, Mapping):
        return [
            item
            for key in sorted(value)
            for item in null_field_paths(value[key], f"{path}.{key}")
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            item
            for index, child in enumerate(value)
            for item in null_field_paths(child, f"{path}[{index}]")
        ]
    return []


def empty_string_field_paths(value: Any, path: str = "$") -> list[str]:
    if value == "":
        return [path]
    if isinstance(value, Mapping):
        return [
            item
            for key in sorted(value)
            for item in empty_string_field_paths(value[key], f"{path}.{key}")
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            item
            for index, child in enumerate(value)
            for item in empty_string_field_paths(child, f"{path}[{index}]")
        ]
    return []


def first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return path
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return path
        for key in sorted(left):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if len(left) != len(right):
            return path
        for index, (first, second) in enumerate(zip(left, right, strict=True)):
            difference = first_difference(first, second, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if left == right else path


def aggregate_analysis(
    included: Sequence[Mapping[str, Any]],
    excluded: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    frontend: Mapping[str, Any],
    export_eligibility: Mapping[str, Any],
    contract_sha256: str,
    input_hashes: Mapping[str, str],
    observed_snapshot_key_count: int,
) -> dict[str, Any]:
    unsupported = sorted(
        name
        for name, value in export_eligibility.items()
        if value["status"] in {"unsupported_export", "not_eligible"}
    )
    counts = [int(item["authoritative_result_count"]) for item in included]
    case_digests = [
        {
            "case_order": int(item["case_order"]),
            "query_identity": item["query_identity"],
            "sha256": item["authoritative_result_digest"],
        }
        for item in included
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": (
            "completed_with_unsupported_exports" if unsupported else "passed"
        ),
        "exit_code": EXIT_NOT_ELIGIBLE if unsupported else EXIT_PASSED,
        "implementation_base_commit": contract["implementation_base_commit"],
        "contract_sha256": contract_sha256,
        "inputs": dict(sorted(input_hashes.items())),
        "closure": {
            "record_case_count": len(included) + len(excluded),
            "included_main_case_count": len(included),
            "excluded_no_successful_source_count": len(excluded),
            "authoritative_final_result_count": len(results),
            "minimum_results_per_query": min(counts) if counts else 0,
            "maximum_results_per_query": max(counts) if counts else 0,
            "queries_below_20_count": sum(value < 20 for value in counts),
            "observed_snapshot_key_count": observed_snapshot_key_count,
        },
        "delivery": {
            "api_exact_query_count": len(included),
            "json_exact_query_count": len(included),
            "jsonl_exact_query_count": len(included),
            "frontend_key_unique_query_count": len(included),
            "pagination_exact_query_count": len(included),
            "authority_digest_preserved_result_count": len(results),
            "unsafe_url_field_count": sum(
                len(item["unsafe_url_fields"]) for item in results
            ),
            "null_field_occurrence_count": sum(
                len(item["null_fields"]) for item in results
            ),
            "empty_string_field_occurrence_count": sum(
                len(item["empty_string_fields"]) for item in results
            ),
            "case_contract_sha256": _stable_json_sha256(case_digests),
            "result_contract_sha256": _stable_json_sha256(
                [
                    {
                        "query_identity": item["query_identity"],
                        "result_position": item["result_position"],
                        "result_identity": item["result_identity"],
                        "authority_digest": item["authority_digest"],
                        "delivery_digest": item["delivery_digest"],
                    }
                    for item in results
                ]
            ),
        },
        "exports": dict(export_eligibility),
        "unsupported_or_ineligible_exports": unsupported,
        "frontend": dict(frontend),
        "policy_isolation": {
            "production_default": DEFAULT_TIEBREAK_POLICY,
            "deterministic_tiebreak_v2_default_enabled": deterministic_tiebreak_v2_catalog()[
                "default_enabled"
            ],
            "v2_used_for_authoritative_reconstruction": False,
            "current_rules_delivery_query_count": len(included),
        },
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
            "official_submission_generated": False,
        },
        "interpretation": {
            "scope": "delivery_fidelity_only",
            "relevance_claim_permitted": False,
            "precision_recall_f1_or_official_score": False,
            "warnings": list(contract["warnings"]),
        },
    }


def write_analysis(
    output_dir: str | Path,
    cases: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    contract_path: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "aggregate": root / "aggregate.json",
        "case_diagnostics": root / "case_diagnostics.jsonl",
        "contract": root / "contract.json",
        "result_diagnostics": root / "result_diagnostics.jsonl",
    }
    _write_json(paths["aggregate"], aggregate)
    _write_jsonl(
        paths["case_diagnostics"],
        sorted(cases, key=lambda item: int(item["case_order"])),
    )
    _write_json(paths["contract"], _read_json(Path(contract_path).expanduser().resolve()))
    _write_jsonl(
        paths["result_diagnostics"],
        sorted(
            results,
            key=lambda item: (int(item["case_order"]), int(item["result_position"])),
        ),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": aggregate["status"],
        "exit_code": aggregate["exit_code"],
        "files": {
            name: {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sorted(paths.items())
        },
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def verify_analysis(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("analysis") != CONTRACT_VERSION:
        raise Top20DeliveryError("manifest_contract_mismatch")
    for value in manifest.get("files", {}).values():
        path = root / str(value["path"])
        if not path.is_file() or path.stat().st_size != int(value["size"]):
            raise Top20DeliveryError("output_missing_or_size_drift")
        if _sha256(path) != str(value["sha256"]):
            raise Top20DeliveryError("output_hash_drift")
    aggregate = _read_json(root / "aggregate.json")
    if aggregate.get("status") not in {"passed", "completed_with_unsupported_exports"}:
        raise Top20DeliveryError("analysis_not_completed")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": CONTRACT_VERSION,
        "status": aggregate["status"],
        "exit_code": int(aggregate["exit_code"]),
        "manifest_sha256": _sha256(root / "manifest.json"),
        "verified_file_count": len(manifest["files"]),
        "execution": aggregate["execution"],
        "unsupported_or_ineligible_exports": aggregate[
            "unsupported_or_ineligible_exports"
        ],
    }


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise Top20DeliveryError("network_attempt_detected")

    with (
        patch.object(socket, "create_connection", blocked),
        patch.object(socket.socket, "connect", blocked),
    ):
        yield


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
