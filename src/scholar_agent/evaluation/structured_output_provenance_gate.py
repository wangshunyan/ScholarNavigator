"""Offline evidence-traceability gate for frozen structured search results.

The gate reads existing Replay rows and retrieval Snapshots only. It never
loads evaluator gold, invokes a connector, calls an LLM, or mutates a
Snapshot. Publicly returned papers are the only citation scope.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from scholar_agent.core.api_schemas import (
    MethodCluster,
    PaperIdentifiers,
    RankedPaper,
    SearchRunResultResponse,
    TimelineItem,
)
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.search_schemas import QueryAnalysis, QueryPlanningResult
from scholar_agent.core.search_schemas import (
    EvidenceItem as InternalEvidenceItem,
    RankedPaper as InternalRankedPaper,
    RerankScoreBreakdown,
    SearchPlan as InternalSearchPlan,
)
from scholar_agent.agents.retriever import SourceStats
from scholar_agent.agents.synthesis import synthesize_answer
from scholar_agent.evaluation.local_bm25_conversion_audit import (
    _reconstruct_candidates,
)
from scholar_agent.evaluation.snapshots import (
    SnapshotIntegrityError,
    SnapshotMissingError,
    SnapshotStore,
)
from scholar_agent.services.api_mapper import (
    _method_clusters,
    _paper_node_id,
    _timeline,
    map_query_analysis,
    map_synthesis_output,
)
from scholar_agent.services.search_service import SearchServiceOutput


AUDIT_SCHEMA_VERSION = "1"
_CITATION_RE = re.compile(r"\[R(\d+)\]")
_SUPPORTED_EVIDENCE = {"title", "abstract", "venue", "metadata"}


def validate_structured_result(
    payload: Mapping[str, Any],
    *,
    query: str,
    query_analysis_payload: Mapping[str, Any],
    planning_payload: Mapping[str, Any],
    expected_sources: Sequence[str],
    expected_top_k: int,
    final_ranked_candidates: Sequence[Mapping[str, Any]],
    source_candidates: Sequence[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one frozen public result and emit claim-level provenance."""

    issues: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    raw = dict(payload)
    extra_fields = sorted(set(raw) - set(SearchRunResultResponse.model_fields))
    for field in extra_fields:
        _issue(issues, "schema_extra_field", field)
    try:
        result = SearchRunResultResponse.model_validate(raw)
    except ValidationError as exc:
        for error in exc.errors(include_url=False, include_context=False):
            _issue(
                issues,
                "schema_invalid",
                ".".join(str(value) for value in error["loc"]),
                str(error["type"]),
            )
        _append_unverified_issues(issues, provenance)
        return _terminal("schema_invalid", issues, provenance), provenance

    internal_analysis = QueryAnalysis.model_validate(query_analysis_payload)
    expected_analysis = map_query_analysis(internal_analysis).model_dump(mode="json")
    if result.query_analysis.model_dump(mode="json") != expected_analysis:
        _issue(issues, "query_constraints_drift", "query_analysis")
    else:
        _provenance(
            provenance,
            "query_constraints",
            "query_analysis",
            ["stage_diagnostics.initial_query_planning.query_analysis"],
        )

    planning = QueryPlanningResult.model_validate(planning_payload)
    selected = list(planning.selected_subqueries)
    expected_queries = _dedupe(item.query for item in selected)
    if result.search_plan.expanded_queries != expected_queries:
        _issue(issues, "planned_query_order_drift", "search_plan.expanded_queries")
    if result.search_plan.source_preferences != list(expected_sources):
        _issue(issues, "source_preferences_drift", "search_plan.source_preferences")
    if result.search_plan.query_planning.model_dump(mode="json") != planning.model_dump(
        mode="json"
    ):
        _issue(issues, "query_planning_drift", "search_plan.query_planning")

    visible = [*result.highly_relevant_papers, *result.partially_relevant_papers]
    by_rank = {item.rank: item for item in visible}
    final_by_rank = {
        int(item["rank"]): item
        for item in final_ranked_candidates
        if item.get("rank") is not None
    }
    _validate_rank_order(result, issues)
    _validate_candidate_identities(visible, issues)
    for candidate in visible:
        path = f"returned_papers.R{candidate.rank}"
        expected = final_by_rank.get(candidate.rank)
        if expected is None:
            _issue(issues, "candidate_rank_out_of_bounds", path)
            continue
        if not _diagnostic_matches_public(expected, candidate):
            _issue(issues, "candidate_identity_or_rank_drift", path)
            continue
        source = _find_source_candidate(candidate, source_candidates)
        if source is None:
            _issue(issues, "candidate_not_in_frozen_snapshot", path)
            continue
        field_issues = _source_field_issues(candidate, source)
        for field in field_issues:
            _issue(issues, "candidate_source_field_drift", f"{path}.{field}")
        if not field_issues:
            _provenance(
                provenance,
                "paper",
                path,
                ["retrieval_snapshot.paper", f"stage_diagnostics.final_ranked.R{candidate.rank}"],
            )

    expected_clusters = _method_clusters(internal_analysis, visible)
    _compare_models(
        result.method_clusters,
        expected_clusters,
        "method_cluster_mismatch",
        "method_clusters",
        issues,
    )
    expected_timeline = _timeline(visible)
    _compare_models(
        result.timeline,
        expected_timeline,
        "timeline_mismatch",
        "timeline",
        issues,
    )
    _validate_group_references(result.method_clusters, by_rank, "method", issues, provenance)
    _validate_group_references(result.timeline, by_rank, "timeline", issues, provenance)
    _validate_citation_graph(result, by_rank, issues, provenance)

    synthesis = result.synthesis
    if synthesis is None:
        _issue(issues, "missing_structured_synthesis", "synthesis")
        _append_unverified_issues(issues, provenance)
        return _terminal("blocked_missing_synthesis", issues, provenance), provenance

    regenerated = _regenerate_synthesis(
        result,
        internal_analysis=internal_analysis,
        planning=planning,
        expected_sources=expected_sources,
        expected_top_k=expected_top_k,
        final_ranked_candidates=final_ranked_candidates,
        source_candidates=source_candidates,
    )
    if regenerated != synthesis.model_dump(mode="json"):
        _issue(issues, "synthesis_generator_drift", "synthesis")
    else:
        _provenance(
            provenance,
            "structured_chain",
            "synthesis",
            ["scholar_agent.agents.synthesis.synthesize_answer"],
        )

    evidence_by_id: dict[str, Any] = {}
    evidence_by_key: dict[str, list[Any]] = {}
    for index, evidence in enumerate(synthesis.evidence_table):
        path = f"synthesis.evidence_table[{index}]"
        if evidence.row_id in evidence_by_id:
            _issue(issues, "duplicate_evidence_row_id", path, evidence.row_id)
        evidence_by_id[evidence.row_id] = evidence
        evidence_by_key.setdefault(evidence.citation_key, []).append(evidence)
        candidate = by_rank.get(evidence.rank)
        expected_key = f"R{evidence.rank}"
        if candidate is None:
            _issue(issues, "fabricated_paper_reference", path, expected_key)
            continue
        if evidence.citation_key != expected_key:
            _issue(issues, "citation_key_rank_mismatch", path, evidence.citation_key)
            continue
        if not _evidence_identity_matches(evidence, candidate):
            _issue(issues, "evidence_identity_conflict", path, expected_key)
            continue
        source_index = _matching_evidence_index(evidence, candidate)
        if source_index is None:
            _issue(issues, "evidence_not_grounded", path, evidence.evidence_source)
            continue
        if evidence.supported_terms != candidate.matched_constraints:
            _issue(issues, "supported_terms_drift", path)
            continue
        if evidence.supported_claim != _expected_supported_claim(candidate, evidence):
            _issue(issues, "supported_claim_drift", path)
            continue
        _provenance(
            provenance,
            "evidence",
            path,
            [
                f"returned_papers.R{candidate.rank}",
                f"returned_papers.R{candidate.rank}.evidence[{source_index}]",
            ],
        )

    finding_citations: set[str] = set()
    for index, finding in enumerate(synthesis.key_findings):
        path = f"synthesis.key_findings[{index}]"
        rows = [evidence_by_id.get(value) for value in finding.evidence_row_ids]
        unknown_rows = [
            value
            for value, row in zip(finding.evidence_row_ids, rows, strict=True)
            if row is None
        ]
        unknown_keys = [
            value for value in finding.citation_keys if value not in evidence_by_key
        ]
        if unknown_rows or unknown_keys:
            _issue(
                issues,
                "finding_reference_out_of_bounds",
                path,
                ",".join([*unknown_rows, *unknown_keys]),
            )
            continue
        if not finding.citation_keys or not finding.evidence_row_ids:
            _issue(issues, "finding_missing_provenance", path)
            continue
        row_keys = {row.citation_key for row in rows if row is not None}
        if row_keys != set(finding.citation_keys):
            _issue(issues, "finding_citation_evidence_mismatch", path)
            continue
        text_keys = {f"R{value}" for value in _CITATION_RE.findall(finding.text)}
        if text_keys != set(finding.citation_keys):
            _issue(issues, "finding_text_citation_mismatch", path)
            continue
        first = rows[0]
        if first is None or finding.text != _expected_finding_text(first):
            _issue(issues, "finding_claim_drift", path)
            continue
        if finding.confidence != first.final_score:
            _issue(issues, "finding_confidence_drift", path)
            continue
        finding_citations.update(finding.citation_keys)
        _provenance(
            provenance,
            "finding",
            path,
            [f"synthesis.evidence_table.{value}" for value in finding.evidence_row_ids],
        )

    expected_summary = _expected_summary(
        query,
        internal_analysis,
        synthesis.evidence_table,
        len(synthesis.key_findings),
    )
    summary_keys = {f"R{value}" for value in _CITATION_RE.findall(synthesis.answer_summary)}
    if summary_keys - set(evidence_by_key):
        _issue(
            issues,
            "summary_reference_out_of_bounds",
            "synthesis.answer_summary",
            ",".join(sorted(summary_keys - set(evidence_by_key))),
        )
    elif synthesis.answer_summary != expected_summary:
        _issue(issues, "summary_claim_drift", "synthesis.answer_summary")
    else:
        _provenance(
            provenance,
            "summary",
            "synthesis.answer_summary",
            [
                "input.query",
                "stage_diagnostics.initial_query_planning.query_analysis",
                *[f"synthesis.evidence_table.{key}" for key in sorted(summary_keys)],
            ],
        )

    expected_ranked_count = min(expected_top_k, len(final_ranked_candidates))
    expected_coverage = {
        "ranked_paper_count": expected_ranked_count,
        "cited_paper_count": len(evidence_by_key),
        "evidence_row_count": len(synthesis.evidence_table),
        "cited_evidence_row_count": sum(
            len(evidence_by_key.get(key, [])) for key in finding_citations
        ),
        "missing_evidence_count": len(result.warnings)
        + sum(
            bool(item.error_message)
            for item in result.retrieval_diagnostics.source_stats
        ),
        "source_error_count": sum(
            bool(item.error_message)
            for item in result.retrieval_diagnostics.source_stats
        ),
        "coverage_ratio": round(
            len(evidence_by_key) / expected_ranked_count
            if expected_ranked_count
            else 0.0,
            4,
        ),
    }
    if synthesis.citation_coverage.model_dump(mode="json") != expected_coverage:
        _issue(issues, "citation_coverage_mismatch", "synthesis.citation_coverage")

    _append_unverified_issues(issues, provenance)
    return _terminal("passed" if not issues else "failed_validation", issues, provenance), provenance


def run_structured_output_provenance_gate(
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the gate over all frozen datasets without loading evaluator gold."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest)
    case_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {"manifest_sha256": _sha256(manifest_file)}
    for spec in manifest["frozen_inputs"]:
        dataset_cases, dataset_provenance, fingerprint = _audit_dataset(spec)
        case_rows.extend(dataset_cases)
        provenance_rows.extend(dataset_provenance)
        inputs[str(spec["label"])] = fingerprint
    case_rows.sort(key=lambda item: (str(item["dataset"]), int(item["case_order"])))
    provenance_rows.sort(
        key=lambda item: (
            str(item["dataset"]),
            int(item["case_order"]),
            str(item["kind"]),
            str(item["target"]),
        )
    )
    datasets = {
        str(spec["label"]): _aggregate_cases(
            [row for row in case_rows if row["dataset"] == spec["label"]]
        )
        for spec in manifest["frozen_inputs"]
    }
    status_counts = Counter(str(row["terminal_status"]) for row in case_rows)
    issue_counts = Counter(
        str(issue["code"]) for row in case_rows for issue in row["issues"]
    )
    aggregate = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "structured_output_provenance_gate",
        "implementation_base_commit": manifest["implementation_base_commit"],
        "inputs": inputs,
        "case_count": len(case_rows),
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "terminal_count_closed": sum(status_counts.values()) == len(case_rows),
        "issue_counts": dict(sorted(issue_counts.items())),
        "provenance": {
            "claim_count": len(provenance_rows),
            "verified_claim_count": sum(
                item["status"] == "verified" for item in provenance_rows
            ),
            "unverified_claim_count": sum(
                item["status"] != "verified" for item in provenance_rows
            ),
        },
        "datasets": datasets,
        "gate_passed": status_counts == {"passed": len(case_rows)},
        "formal_execution_blocked": any(
            value.startswith("blocked_") or value == "schema_invalid"
            for value in status_counts
        ),
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_loaded": False,
        },
    }
    if len(case_rows) != sum(int(item["case_count"]) for item in manifest["frozen_inputs"]):
        raise ValueError("structured gate case count is not closed")
    return case_rows, provenance_rows, aggregate


def write_structured_output_provenance_gate(
    output: str | Path,
    case_rows: Sequence[Mapping[str, Any]],
    provenance_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> None:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "case_gate.jsonl", case_rows)
    _write_jsonl(root / "provenance.jsonl", provenance_rows)
    _write_json(root / "aggregate.json", aggregate)


def _audit_dataset(
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    label = str(spec["label"])
    run_root = Path(str(spec["run_dir"])).expanduser().resolve()
    snapshot_root = Path(str(spec["snapshot_dir"])).expanduser().resolve()
    config = _read_json(run_root / "config.json")
    _validate_config(config)
    rows = _read_rows(run_root / "results.jsonl")
    case_ids = [str(value) for value in config["case_ids"]]
    if len(case_ids) != int(spec["case_count"]) or set(rows) != set(case_ids):
        raise ValueError(f"frozen case set mismatch:{label}")
    store = SnapshotStore(snapshot_root)
    cases: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for case_order, case_id in enumerate(case_ids):
        row = rows[case_id]
        if row.get("status") != "succeeded":
            terminal = _terminal(
                "blocked_frozen_case_terminal",
                [{"code": "frozen_case_not_succeeded", "path": "status", "detail": str(row.get("status"))}],
                [],
            )
            case = _case_row(label, case_order, case_id, terminal)
            cases.append(case)
            continue
        stages = {
            str(item["stage"]): item
            for item in row["stage_diagnostics"]["snapshots"]
        }
        required = {"initial_retrieval", "initial_deduplicated", "final_ranked"}
        if not required.issubset(stages):
            terminal = _terminal(
                "blocked_missing_stage",
                [{"code": "missing_required_stage", "path": "stage_diagnostics", "detail": ",".join(sorted(required - set(stages)))}],
                [],
            )
            cases.append(_case_row(label, case_order, case_id, terminal))
            continue
        try:
            source_candidates = _reconstruct_candidates(
                stages["initial_retrieval"],
                stages["initial_deduplicated"],
                config,
                store,
            )
        except (
            SnapshotIntegrityError,
            SnapshotMissingError,
            ValidationError,
            ValueError,
        ) as exc:
            terminal = _terminal(
                "blocked_snapshot",
                [{"code": "snapshot_reconstruction_failed", "path": "snapshot", "detail": type(exc).__name__}],
                [],
            )
            cases.append(_case_row(label, case_order, case_id, terminal))
            continue
        planning = row["stage_diagnostics"]["initial_query_planning"]
        terminal, claim_rows = validate_structured_result(
            row["result"],
            query=str(row["query"]),
            query_analysis_payload=planning["query_analysis"],
            planning_payload=planning["planning"],
            expected_sources=[str(value) for value in config["sources"]],
            expected_top_k=int(config["top_k"]),
            final_ranked_candidates=stages["final_ranked"].get("candidates") or [],
            source_candidates=source_candidates,
        )
        case = _case_row(label, case_order, case_id, terminal)
        case["visible_candidate_count"] = len(row["result"].get("highly_relevant_papers") or []) + len(row["result"].get("partially_relevant_papers") or [])
        case["synthesis_evidence_row_count"] = len((row["result"].get("synthesis") or {}).get("evidence_table") or [])
        cases.append(case)
        for claim in claim_rows:
            provenance.append(
                {"schema_version": AUDIT_SCHEMA_VERSION, "dataset": label, "case_order": case_order, "case_id": case_id, **claim}
            )
    return cases, provenance, {
        "config_sha256": _sha256(run_root / "config.json"),
        "results_sha256": _sha256(run_root / "results.jsonl"),
        "snapshot_tree_sha256": _tree_sha256(snapshot_root),
        "snapshot_file_count": sum(path.is_file() for path in snapshot_root.rglob("*")),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("query_planning_policy") != "current_rules":
        raise ValueError("structured gate requires current_rules planning")
    if config.get("ranking_policy") != "current_rules":
        raise ValueError("structured gate requires current_rules ranking")
    if int(config.get("top_k") or 0) != 20:
        raise ValueError("structured gate requires Top-20")
    for field in (
        "enable_query_evolution",
        "enable_refchain",
        "enable_semantic_seed_expansion",
        "enable_prf",
        "enable_concept_projection",
        "enable_lexical_normalization",
        "enable_local_bm25_original_deepening",
    ):
        if bool(config.get(field)):
            raise ValueError(f"experimental feature enabled:{field}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported structured gate manifest")
    if manifest.get("audit") != "structured_output_provenance_gate":
        raise ValueError("unexpected structured gate audit")
    execution = manifest.get("execution") or {}
    if any(int(execution.get(field) or 0) for field in ("network_request_count", "llm_request_count", "snapshot_write_count")):
        raise ValueError("structured gate must be offline and read-only")
    if sum(int(item["case_count"]) for item in manifest["frozen_inputs"]) != 65:
        raise ValueError("structured gate must cover the fixed 65 cases")


def _validate_rank_order(result: SearchRunResultResponse, issues: list[dict[str, Any]]) -> None:
    ranks: list[int] = []
    for name, values in (
        ("highly_relevant_papers", result.highly_relevant_papers),
        ("partially_relevant_papers", result.partially_relevant_papers),
    ):
        current = [item.rank for item in values]
        if current != sorted(current):
            _issue(issues, "candidate_order_drift", name)
        ranks.extend(current)
    if len(ranks) != len(set(ranks)):
        _issue(issues, "duplicate_candidate_rank", "returned_papers")


def _validate_candidate_identities(
    visible: Sequence[RankedPaper], issues: list[dict[str, Any]]
) -> None:
    profiles = [build_identity_profile(item.paper) for item in visible]
    for left_index, left in enumerate(profiles):
        for right_index in range(left_index + 1, len(profiles)):
            evidence = identity_evidence_from_profiles(left, profiles[right_index])
            if evidence.equivalent:
                _issue(
                    issues,
                    "duplicate_unified_identity",
                    f"returned_papers.R{visible[left_index].rank}/R{visible[right_index].rank}",
                    evidence.rule,
                )


def _diagnostic_matches_public(expected: Mapping[str, Any], actual: RankedPaper) -> bool:
    paper = actual.paper
    expected_identifiers = PaperIdentifiers.model_validate(
        expected.get("identifiers") or {}
    ).model_dump(mode="json")
    return (
        str(expected.get("title") or "") == paper.title
        and (expected.get("year") or 0) == paper.year
        and expected_identifiers == paper.identifiers.model_dump(mode="json")
        and sorted(str(value) for value in expected.get("sources") or []) == sorted(paper.sources)
        and str(expected.get("category") or "") == actual.category
        and float(expected.get("final_score") or 0.0) == actual.relevance_score
    )


def _find_source_candidate(candidate: RankedPaper, source_candidates: Sequence[Any]) -> Any | None:
    target = build_identity_profile(candidate.paper)
    matches = [
        item
        for item in source_candidates
        if identity_evidence_from_profiles(build_identity_profile(item), target).equivalent
    ]
    return matches[0] if len(matches) == 1 else None


def _source_field_issues(candidate: RankedPaper, source: Any) -> list[str]:
    paper = candidate.paper
    fields = {
        "title": (paper.title, source.title),
        "authors": (paper.authors, list(source.authors)),
        "year": (paper.year, source.year or 0),
        "venue": (paper.venue, source.venue),
        "abstract": (paper.abstract, source.abstract),
        "identifiers": (paper.identifiers.model_dump(mode="json"), source.identifiers.model_dump(mode="json")),
        "urls": (paper.urls.model_dump(mode="json"), source.urls.model_dump(mode="json")),
        "sources": (paper.sources, list(source.sources)),
    }
    issues = [field for field, values in fields.items() if values[0] != values[1]]
    for name, value in paper.urls.model_dump(mode="json").items():
        if value and urlparse(value).scheme not in {"http", "https"}:
            issues.append(f"urls.{name}.invalid_scheme")
    return issues


def _evidence_identity_matches(evidence: Any, candidate: RankedPaper) -> bool:
    paper = candidate.paper
    return (
        evidence.paper_title == paper.title
        and evidence.year == (paper.year or None)
        and evidence.venue == paper.venue
        and evidence.sources == paper.sources
        and evidence.identifiers == paper.identifiers
        and evidence.category == candidate.category
        and evidence.final_score == candidate.relevance_score
    )


def _regenerate_synthesis(
    result: SearchRunResultResponse,
    *,
    internal_analysis: QueryAnalysis,
    planning: QueryPlanningResult,
    expected_sources: Sequence[str],
    expected_top_k: int,
    final_ranked_candidates: Sequence[Mapping[str, Any]],
    source_candidates: Sequence[Any],
) -> dict[str, Any] | None:
    visible_by_rank = {
        item.rank: item
        for item in [
            *result.highly_relevant_papers,
            *result.partially_relevant_papers,
        ]
    }
    ranked: list[InternalRankedPaper] = []
    for diagnostic in sorted(
        final_ranked_candidates,
        key=lambda item: int(item.get("rank") or 10**9),
    )[:expected_top_k]:
        rank = int(diagnostic["rank"])
        visible = visible_by_rank.get(rank)
        source = (
            _find_source_candidate(visible, source_candidates)
            if visible is not None
            else None
        )
        if source is None:
            source = {
                "title": str(diagnostic.get("title") or ""),
                "authors": [],
                "year": diagnostic.get("year"),
                "venue": None,
                "abstract": "",
                "identifiers": dict(diagnostic.get("identifiers") or {}),
                "urls": {},
                "sources": list(diagnostic.get("sources") or []),
            }
        paper_payload = (
            source.model_dump(mode="json")
            if hasattr(source, "model_dump")
            else dict(source)
        )
        final_score = float(diagnostic.get("final_score") or 0.0)
        ranked.append(
            InternalRankedPaper(
                rank=rank,
                paper=paper_payload,
                final_score=final_score,
                category=str(diagnostic.get("category") or "irrelevant"),
                score_breakdown=RerankScoreBreakdown(
                    relevance_score=final_score,
                    authority_score=0.0,
                    timeliness_score=0.0,
                    metadata_score=0.0,
                    final_score=final_score,
                    relevance_weight=1.0,
                    authority_weight=0.0,
                    timeliness_weight=0.0,
                    metadata_weight=0.0,
                ),
                ranking_reason=(visible.ranking_reason if visible else "frozen"),
                evidence=(
                    [
                        InternalEvidenceItem.model_validate(item.model_dump(mode="json"))
                        for item in visible.evidence
                    ]
                    if visible
                    else []
                ),
                matched_terms=(
                    list(visible.matched_constraints)
                    if visible
                    else list(diagnostic.get("matched_terms") or [])
                ),
                warnings=list(diagnostic.get("warnings") or []),
            )
        )
    search_plan = InternalSearchPlan(
        query_analysis=internal_analysis,
        subqueries=list(planning.selected_subqueries),
        selected_sources=list(expected_sources),
        top_k=expected_top_k,
        enable_refchain=False,
        enable_semantic_seed_expansion=False,
        enable_query_evolution=False,
        query_evolution_policy="off",
        query_planning_policy="current_rules",
        ranking_policy="current_rules",
        query_planning=planning,
    )
    source_stats = [
        SourceStats(
            source=item.source,
            returned_count=item.returned_count,
            latency_seconds=item.latency_seconds,
            error_message=item.error_message,
            cache_hit=item.cache_hit,
            logical_call_executed=item.logical_call_executed,
            adaptation_strategy=item.adaptation_strategy,
            triggered_by=list(item.triggered_by),
            diagnostics=item.diagnostics,
        )
        for item in result.retrieval_diagnostics.source_stats
    ]
    output = SearchServiceOutput(
        search_plan=search_plan,
        ranked_papers=ranked,
        all_ranked_papers=ranked,
        warnings=list(result.warnings),
        source_stats=source_stats,
        budget_status=result.budget_status,
    )
    return map_synthesis_output(synthesize_answer(output)).model_dump(mode="json")


def _matching_evidence_index(evidence: Any, candidate: RankedPaper) -> int | None:
    if evidence.evidence_source not in _SUPPORTED_EVIDENCE:
        return None
    for index, item in enumerate(candidate.evidence):
        if item.source != evidence.evidence_source:
            continue
        if evidence.evidence_text != _clip(item.text, 240):
            continue
        if not _evidence_item_matches_original_field(item.source, item.text, candidate):
            continue
        return index
    return None


def _evidence_item_matches_original_field(source: str, text: str, candidate: RankedPaper) -> bool:
    normalized = " ".join(text.split())
    paper = candidate.paper
    if source == "title":
        return " ".join(paper.title.split()).startswith(normalized)
    if source == "abstract":
        return normalized in " ".join(paper.abstract.split())
    if source == "venue":
        return normalized == " ".join((paper.venue or "").split())
    if source == "metadata":
        return normalized == f"year={paper.year}"
    return False


def _expected_supported_claim(candidate: RankedPaper, evidence: Any) -> str:
    terms = ", ".join(candidate.matched_constraints[:3])
    if terms:
        return f"{candidate.paper.title} has {evidence.evidence_source} evidence related to {terms}."
    return f"{candidate.paper.title} has {evidence.evidence_source} evidence relevant to the query."


def _expected_finding_text(row: Any) -> str:
    if row.supported_terms:
        topic = ", ".join(row.supported_terms[:3])
    elif row.venue:
        topic = f"evidence from {row.venue}"
    else:
        topic = "the search query"
    return f"{row.paper_title} provides {row.evidence_source} evidence for {topic} [{row.citation_key}]."


def _expected_summary(
    query: str,
    analysis: QueryAnalysis,
    evidence_rows: Sequence[Any],
    finding_count: int,
) -> str:
    keys = _dedupe(row.citation_key for row in evidence_rows)[:3]
    cited = ", ".join(f"[{key}]" for key in keys)
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for row in evidence_rows:
        for value in row.supported_terms:
            term = value.strip()
            if term:
                counts[term] += 1
                display.setdefault(term, term)
    themes = sorted(counts, key=lambda value: (-counts[value], value.casefold()))[:4]
    theme_text = ", ".join(display[value] for value in themes) if themes else "the retrieved evidence"
    return (
        f'For the query "{query}", the current {analysis.domain} search evidence '
        f"supports a {analysis.intent} synthesis around {theme_text}. The strongest "
        f"citation-backed candidates are {cited}. {finding_count} finding(s) "
        "were generated only from ranked-paper evidence rows."
    )


def _validate_group_references(
    groups: Sequence[MethodCluster] | Sequence[TimelineItem],
    by_rank: Mapping[int, RankedPaper],
    kind: str,
    issues: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    for index, group in enumerate(groups):
        path = f"{kind}[{index}]"
        ranks = list(group.paper_ranks)
        if len(ranks) != len(set(ranks)) or any(rank not in by_rank for rank in ranks):
            _issue(issues, "invalid_group_reference", path)
        else:
            _provenance(
                provenance,
                kind,
                path,
                [f"returned_papers.R{rank}" for rank in ranks],
            )


def _validate_citation_graph(
    result: SearchRunResultResponse,
    by_rank: Mapping[int, RankedPaper],
    issues: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    expected = {
        node_id: (paper.paper.title, rank)
        for rank, paper in by_rank.items()
        if (node_id := _paper_node_id(paper.paper)) is not None
    }
    actual = {node.id: (node.label, node.rank) for node in result.citation_graph.nodes}
    if actual != expected or result.citation_graph.edges:
        _issue(issues, "citation_graph_mismatch", "citation_graph")
        return
    for node_id, (_label, rank) in sorted(actual.items()):
        _provenance(
            provenance,
            "citation_graph_node",
            f"citation_graph.nodes.{node_id}",
            [f"returned_papers.R{rank}"],
        )


def _compare_models(
    actual: Sequence[Any],
    expected: Sequence[Any],
    code: str,
    path: str,
    issues: list[dict[str, Any]],
) -> None:
    left = [item.model_dump(mode="json") for item in actual]
    right = [item.model_dump(mode="json") for item in expected]
    if left != right:
        _issue(issues, code, path)


def _aggregate_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row["terminal_status"]) for row in rows)
    issues = Counter(
        str(issue["code"]) for row in rows for issue in row.get("issues") or []
    )
    return {
        "case_count": len(rows),
        "terminal_status_counts": dict(sorted(statuses.items())),
        "issue_counts": dict(sorted(issues.items())),
        "visible_candidate_count": sum(int(row.get("visible_candidate_count") or 0) for row in rows),
        "synthesis_evidence_row_count": sum(int(row.get("synthesis_evidence_row_count") or 0) for row in rows),
    }


def _case_row(label: str, case_order: int, case_id: str, terminal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": label,
        "case_order": case_order,
        "case_id": case_id,
        **terminal,
    }


def _terminal(
    status: str,
    issues: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "terminal_status": status,
        "issue_count": len(issues),
        "issues": list(issues),
        "verified_claim_count": sum(item.get("status") == "verified" for item in provenance),
    }


def _issue(issues: list[dict[str, Any]], code: str, path: str, detail: str = "") -> None:
    issues.append({"code": code, "path": path, "detail": detail})


def _provenance(
    rows: list[dict[str, Any]], kind: str, target: str, source_paths: Sequence[str]
) -> None:
    rows.append(
        {
            "kind": kind,
            "target": target,
            "status": "verified",
            "source_paths": list(source_paths),
        }
    )


def _append_unverified_issues(
    issues: Sequence[Mapping[str, Any]], rows: list[dict[str, Any]]
) -> None:
    existing = {
        (str(item["target"]), str(item["status"]))
        for item in rows
    }
    for issue in issues:
        target = str(issue["path"])
        if (target, "unverified") in existing:
            continue
        rows.append(
            {
                "kind": "unverified_output",
                "target": target,
                "status": "unverified",
                "source_paths": [],
                "reason": str(issue["code"]),
            }
        )
        existing.add((target, "unverified"))


def _clip(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _dedupe(values: Sequence[str] | Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = {str(item["case_id"]): item for item in values}
    if len(rows) != len(values):
        raise ValueError(f"duplicate frozen case id:{path}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
