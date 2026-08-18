"""受约束 LLM 查询改写的纯离线因果贡献审计。

本模块只读取已经冻结的 Benchmark Replay 与 Retrieval Snapshot。它不会导入
``SearchService``，不会调用 connector/LLM，也不会把 gold 反馈到查询生成路径。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis
from scholar_agent.evaluation.datasets import load_dataset
from scholar_agent.evaluation.metrics import (
    canonical_paper_id,
    evaluable_gold_count,
    evaluate_ranking,
    matched_paper_ids,
)
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.snapshots import SnapshotStore
from scholar_agent.evaluation.snapshots.store import (
    connector_version,
    retrieval_snapshot_key,
)
from scholar_agent.retrieval.query_adapter import adapt_queries_for_source


AUDIT_SCHEMA_VERSION = "1"
CaseClassification = Literal[
    "rewrite_added_gold",
    "replacement_lost_gold",
    "rewrite_added_and_replacement_lost_gold",
    "candidate_only_change",
    "no_marginal_change",
    "source_terminal_inconsistent",
    "fallback_or_rejected_unattributable",
    "counterfactual_reconstruction_failed",
]


class AuditPair(BaseModel):
    """One frozen baseline/rewrite pair."""

    name: str
    baseline_run: Path
    rewrite_run: Path
    baseline_snapshot: Path
    rewrite_snapshot: Path


class QueryListObservation(BaseModel):
    query: str
    source: str
    request_terminals: list[dict[str, Any]] = Field(default_factory=list)
    raw_candidate_count: int = 0
    unique_candidate_count: int = 0
    duplicate_ratio: float = 0.0
    candidate_ids: list[str] = Field(default_factory=list)
    gold_ids: list[str] = Field(default_factory=list)
    independent_candidate_ids: list[str] = Field(default_factory=list)
    independent_gold_ids: list[str] = Field(default_factory=list)
    first_gold_rank: int | None = None


class _QueryList:
    def __init__(
        self,
        query: str,
        source: str,
        papers: list[Paper],
        raw_count: int,
        terminals: list[dict[str, Any]],
    ) -> None:
        self.query = query
        self.source = source
        self.papers = papers
        self.raw_count = raw_count
        self.terminals = terminals


class _PreparedCounterfactual(NamedTuple):
    candidates: list[Paper]
    diagnostics: list[dict[str, Any]]
    analysis: QueryAnalysis
    baseline_metrics: dict[str, Any]
    baseline_ranked: list[Paper]


def run_causal_audit(pairs: Sequence[AuditPair]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit all pairs without creating a network-capable runtime."""

    case_rows: list[dict[str, Any]] = []
    for pair in pairs:
        case_rows.extend(_audit_pair(pair))
    aggregate = _aggregate(case_rows, pairs)
    return case_rows, aggregate


def write_causal_audit(
    output: str | Path,
    case_rows: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    case_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in case_rows
    )
    _atomic_write_text(root / "case_audit.jsonl", case_payload)
    _atomic_write_json(root / "aggregate.json", aggregate)


def _audit_pair(pair: AuditPair) -> list[dict[str, Any]]:
    baseline_config = _read_json(pair.baseline_run / "config.json")
    rewrite_config = _read_json(pair.rewrite_run / "config.json")
    _validate_pair_configs(baseline_config, rewrite_config)
    baseline_rows = _read_rows(pair.baseline_run / "results.jsonl")
    rewrite_rows = _read_rows(pair.rewrite_run / "results.jsonl")
    if set(baseline_rows) != set(rewrite_rows):
        raise ValueError(f"case set mismatch for pair {pair.name}")

    dataset = load_dataset(str(baseline_config["dataset"]))
    cases = {item.query_id: item for item in dataset}
    selected_case_ids = [str(value) for value in baseline_config["case_ids"]]
    if any(case_id not in cases for case_id in selected_case_ids):
        raise ValueError(f"dataset case missing for pair {pair.name}")

    baseline_store = SnapshotStore(pair.baseline_snapshot)
    rewrite_store = SnapshotStore(pair.rewrite_snapshot)
    result: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        result.append(
            _audit_case(
                pair.name,
                cases[case_id],
                baseline_rows[case_id],
                rewrite_rows[case_id],
                baseline_config,
                rewrite_config,
                baseline_store,
                rewrite_store,
            )
        )
    return result


def _audit_case(
    pair_name: str,
    eval_query: EvalQuery,
    baseline_row: dict[str, Any],
    rewrite_row: dict[str, Any],
    baseline_config: dict[str, Any],
    rewrite_config: dict[str, Any],
    baseline_store: SnapshotStore,
    rewrite_store: SnapshotStore,
) -> dict[str, Any]:
    planning = rewrite_row["stage_diagnostics"]["initial_query_planning"]["planning"]
    accepted_rewrite, fallback_reason = rewrite_acceptance(planning)
    base = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": pair_name,
        "case_id": eval_query.query_id,
        "query": eval_query.query,
        "accepted_rewrite": accepted_rewrite,
        "fallback_used": bool(planning.get("fallback_used")),
        "fallback_reason": fallback_reason,
        "source_rows": [],
        "counterfactual": None,
    }
    if not base["accepted_rewrite"]:
        return {
            **base,
            "classification": "fallback_or_rejected_unattributable",
            "comparable": False,
            "incomparable_reasons": [
                str(planning.get("fallback_reason") or "rewrite_not_accepted")
            ],
        }

    original_query = str(planning["constrained_rewrite_input_summary"]["original_query"])
    replaced_query = str(planning["constrained_rewrite_replaced_query"])
    rewrite_query = str(planning["constrained_rewrite_query"])
    base.update(
        {
            "original_query": original_query,
            "replaced_query": replaced_query,
            "rewrite_query": rewrite_query,
        }
    )
    sources = [str(item) for item in baseline_config["sources"]]
    source_rows: list[dict[str, Any]] = []
    rewrite_lists: list[_QueryList] = []
    for source in sources:
        original_baseline = _query_list(
            baseline_row,
            baseline_config,
            baseline_store,
            original_query,
            source,
        )
        original_rewrite = _query_list(
            rewrite_row,
            rewrite_config,
            rewrite_store,
            original_query,
            source,
        )
        replaced = _query_list(
            baseline_row,
            baseline_config,
            baseline_store,
            replaced_query,
            source,
        )
        rewritten = _query_list(
            rewrite_row,
            rewrite_config,
            rewrite_store,
            rewrite_query,
            source,
        )
        status, reasons = source_comparability(
            original_baseline,
            original_rewrite,
            replaced,
            rewritten,
        )
        observations = query_list_observations(
            [original_baseline, replaced, rewritten],
            eval_query.gold_papers,
        )
        source_rows.append(
            {
                "source": source,
                "status": status,
                "reasons": reasons,
                "original_rewrite_request_terminals": original_rewrite.terminals,
                "lists": {
                    "original": observations[0].model_dump(mode="json"),
                    "replaced": observations[1].model_dump(mode="json"),
                    "rewrite": observations[2].model_dump(mode="json"),
                },
            }
        )
        if status == "comparable":
            rewrite_lists.append(rewritten)

    base["source_rows"] = source_rows
    try:
        prepared = prepare_counterfactual_baseline(
            baseline_row,
            baseline_config,
            baseline_store,
            eval_query,
        )
    except ValueError as exc:
        return {
            **base,
            "classification": "counterfactual_reconstruction_failed",
            "comparable": False,
            "incomparable_reasons": [str(exc)],
        }
    rewritten_by_source = rewrite_lists_by_source(source_rows, rewrite_lists)
    for source_row, rewritten in zip(source_rows, rewritten_by_source):
        if source_row["status"] != "comparable" or rewritten is None:
            source_row["counterfactual"] = None
            continue
        source_row["counterfactual"] = build_source_counterfactual(
            prepared,
            eval_query.gold_papers,
            baseline_config,
            replaced_query=replaced_query,
            source=str(source_row["source"]),
            rewritten=rewritten,
        )
        source_row["counterfactual_classification"] = classify_counterfactual(
            source_row["counterfactual"]
        )
    incomparable = [
        f"{row['source']}:{reason}"
        for row in source_rows
        if row["status"] != "comparable"
        for reason in row["reasons"]
    ]
    if incomparable:
        return {
            **base,
            "classification": "source_terminal_inconsistent",
            "comparable": False,
            "incomparable_reasons": sorted(set(incomparable)),
        }

    counterfactual = build_counterfactual_from_prepared(
        prepared,
        eval_query.gold_papers,
        baseline_config,
        replaced_query=replaced_query,
        rewrite_lists=rewrite_lists,
    )
    classification = classify_counterfactual(counterfactual)
    return {
        **base,
        "classification": classification,
        "comparable": True,
        "incomparable_reasons": [],
        "counterfactual": counterfactual,
    }


def rewrite_acceptance(planning: dict[str, Any]) -> tuple[bool, str | None]:
    """Return whether a generated rewrite was actually accepted for retrieval."""

    accepted = (
        int(planning.get("accepted_query_count") or 0) == 1
        and not bool(planning.get("fallback_used"))
    )
    reason = planning.get("fallback_reason")
    return accepted, str(reason) if reason is not None else None


def _query_list(
    row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
    query: str,
    source: str,
) -> _QueryList:
    planning = row["stage_diagnostics"]["initial_query_planning"]
    selected = planning["planning"]["selected_subqueries"]
    selected_row = next(
        (item for item in selected if str(item.get("query")) == query),
        None,
    )
    if selected_row is None:
        raise ValueError(f"planned query missing:{query}")
    analysis = QueryAnalysis.model_validate(planning["query_analysis"])
    adapter_policy = str(config["query_adapter_policy"])
    planning_policy = str(config["query_planning_policy"])
    planner_version = str(planning["planner_version"])
    observed = set(row["snapshot_cost_report"]["observed_retrieval_keys"])
    limit = int(config["top_k"])
    adapted_queries = adapt_queries_for_source(
        query,
        source,
        constraints=analysis.constraints,
        policy=adapter_policy,
        combination_mode=str(selected_row.get("combination_mode") or "all"),
    )
    terminals: list[dict[str, Any]] = []
    raw_papers: list[Paper] = []
    for adapted in adapted_queries:
        key, _ = retrieval_snapshot_key(
            source=source,
            adapted_query=adapted.query,
            limit=limit,
            adapter_policy=adapter_policy,
            connector_version=connector_version(source),
            query_planning_policy=planning_policy,
            query_planner_version=planner_version,
        )
        if key not in observed:
            terminals.append(
                {
                    "adapted_query": adapted.query,
                    "adaptation_strategy": adapted.strategy,
                    "status": "not_started",
                    "key": key,
                    "paper_count": 0,
                }
            )
            continue
        entry = store.read_retrieval(key)
        if (
            entry.source != source
            or entry.adapted_query != adapted.query
            or entry.limit != limit
            or entry.adapter_policy != adapter_policy
        ):
            raise ValueError(f"snapshot request mismatch:{key}")
        terminals.append(
            {
                "adapted_query": adapted.query,
                "adaptation_strategy": adapted.strategy,
                "status": entry.status,
                "key": key,
                "paper_count": len(entry.papers),
                "error_type": _error_type(entry.error_message),
            }
        )
        if entry.status == "success":
            raw_papers.extend(entry.papers)
    return _QueryList(
        query,
        source,
        deduplicate_papers(raw_papers),
        len(raw_papers),
        terminals,
    )


def source_comparability(
    original_baseline: _QueryList,
    original_rewrite: _QueryList,
    replaced: _QueryList,
    rewritten: _QueryList,
) -> tuple[str, list[str]]:
    """Gate causal comparisons on request terminals and original-list parity."""

    reasons: list[str] = []
    for label, observation in (
        ("original_baseline", original_baseline),
        ("original_rewrite", original_rewrite),
        ("replaced", replaced),
        ("rewrite", rewritten),
    ):
        failures = [
            item for item in observation.terminals if item["status"] == "failed"
        ]
        if failures:
            reasons.append(f"{label}_request_failed")
    baseline_vector = [
        (item["adapted_query"], item["status"])
        for item in original_baseline.terminals
    ]
    rewrite_vector = [
        (item["adapted_query"], item["status"])
        for item in original_rewrite.terminals
    ]
    if baseline_vector != rewrite_vector:
        reasons.append("original_request_terminal_mismatch")
    if not equivalent_paper_sequences(
        original_baseline.papers,
        original_rewrite.papers,
    ):
        reasons.append("original_candidate_list_mismatch")
    return ("comparable", []) if not reasons else (
        "source_terminal_inconsistent",
        sorted(set(reasons)),
    )


def query_list_observations(
    lists: Sequence[_QueryList],
    gold: Sequence[EvalGoldPaper],
) -> list[QueryListObservation]:
    """Compute overlap-aware query-list contribution with shared identity rules."""

    clustered = identity_cluster_labels([item.papers for item in lists])
    gold_sets = [set(matched_paper_ids(item.papers, gold)) for item in lists]
    result: list[QueryListObservation] = []
    for index, item in enumerate(lists):
        candidate_ids = clustered[index]
        other_candidates = set().union(
            *(set(values) for offset, values in enumerate(clustered) if offset != index)
        )
        other_gold = set().union(
            *(values for offset, values in enumerate(gold_sets) if offset != index)
        )
        unique_count = len(candidate_ids)
        result.append(
            QueryListObservation(
                query=item.query,
                source=item.source,
                request_terminals=item.terminals,
                raw_candidate_count=item.raw_count,
                unique_candidate_count=unique_count,
                duplicate_ratio=(
                    max(0, item.raw_count - unique_count) / item.raw_count
                    if item.raw_count
                    else 0.0
                ),
                candidate_ids=candidate_ids,
                gold_ids=sorted(gold_sets[index]),
                independent_candidate_ids=sorted(set(candidate_ids) - other_candidates),
                independent_gold_ids=sorted(gold_sets[index] - other_gold),
                first_gold_rank=first_gold_rank(item.papers, gold),
            )
        )
    return result


def identity_cluster_labels(groups: Sequence[Sequence[Paper]]) -> list[list[str]]:
    """Assign deterministic identity-cluster labels across several candidate lists."""

    profiles = []
    labels: list[str] = []
    result: list[list[str]] = []
    for group in groups:
        group_labels: list[str] = []
        for paper in group:
            profile = build_identity_profile(paper)
            match = next(
                (
                    index
                    for index, existing in enumerate(profiles)
                    if identity_evidence_from_profiles(existing, profile).equivalent
                ),
                None,
            )
            if match is None:
                profiles.append(profile)
                labels.append(canonical_paper_id(paper) or f"identity_cluster:{len(labels)}")
                match = len(labels) - 1
            group_labels.append(labels[match])
        result.append(group_labels)
    return result


def first_gold_rank(papers: Sequence[Paper], gold: Sequence[EvalGoldPaper]) -> int | None:
    for rank in range(1, len(papers) + 1):
        if matched_paper_ids(papers[:rank], gold):
            return rank
    return None


def build_counterfactual(
    baseline_row: dict[str, Any],
    baseline_config: dict[str, Any],
    baseline_store: SnapshotStore,
    eval_query: EvalQuery,
    *,
    replaced_query: str,
    rewrite_lists: Sequence[_QueryList],
) -> dict[str, Any]:
    prepared = prepare_counterfactual_baseline(
        baseline_row,
        baseline_config,
        baseline_store,
        eval_query,
    )
    return build_counterfactual_from_prepared(
        prepared,
        eval_query.gold_papers,
        baseline_config,
        replaced_query=replaced_query,
        rewrite_lists=rewrite_lists,
    )


def prepare_counterfactual_baseline(
    baseline_row: dict[str, Any],
    baseline_config: dict[str, Any],
    baseline_store: SnapshotStore,
    eval_query: EvalQuery,
) -> _PreparedCounterfactual:
    reconstructed = reconstruct_candidate_pool(
        baseline_row, baseline_config, baseline_store
    )
    expected = _stage_candidates(baseline_row, "initial_deduplicated")
    baseline_candidates = align_papers_to_diagnostics(reconstructed, expected)
    analysis = QueryAnalysis.model_validate(
        baseline_row["stage_diagnostics"]["initial_query_planning"]["query_analysis"]
    )
    baseline_metrics, baseline_ranked, baseline_all_ranked = _rank_pool(
        analysis, baseline_candidates, eval_query.gold_papers
    )
    expected_ranked = _stage_candidates(baseline_row, "initial_reranked")
    if not equivalent_paper_sequences(baseline_all_ranked, expected_ranked):
        raise ValueError("baseline_ranking_reconstruction_mismatch")
    return _PreparedCounterfactual(
        baseline_candidates,
        expected,
        analysis,
        baseline_metrics,
        baseline_ranked,
    )


def build_counterfactual_from_prepared(
    prepared: _PreparedCounterfactual,
    gold: Sequence[EvalGoldPaper],
    baseline_config: dict[str, Any],
    *,
    replaced_query: str,
    rewrite_lists: Sequence[_QueryList],
) -> dict[str, Any]:
    return _build_intervention_counterfactual(
        prepared,
        gold,
        baseline_config,
        remove=lambda provenance: bool(provenance)
        and all(str(item.get("origin_subquery")) == replaced_query for item in provenance),
        rewrite_candidates=deduplicate_papers(
            [paper for item in rewrite_lists for paper in item.papers]
        ),
    )


def build_source_counterfactual(
    prepared: _PreparedCounterfactual,
    gold: Sequence[EvalGoldPaper],
    baseline_config: dict[str, Any],
    *,
    replaced_query: str,
    source: str,
    rewritten: _QueryList,
) -> dict[str, Any]:
    return _build_intervention_counterfactual(
        prepared,
        gold,
        baseline_config,
        remove=lambda provenance: bool(provenance)
        and all(
            str(item.get("origin_subquery")) == replaced_query
            and str(item.get("source")) == source
            for item in provenance
        ),
        rewrite_candidates=list(rewritten.papers),
    )


def _build_intervention_counterfactual(
    prepared: _PreparedCounterfactual,
    gold: Sequence[EvalGoldPaper],
    baseline_config: dict[str, Any],
    *,
    remove: Callable[[list[dict[str, Any]]], bool],
    rewrite_candidates: list[Paper],
) -> dict[str, Any]:
    retained: list[Paper] = []
    removed: list[Paper] = []
    for paper, diagnostic in zip(prepared.candidates, prepared.diagnostics):
        provenance = list(diagnostic.get("provenance") or [])
        if remove(provenance):
            removed.append(paper)
        else:
            retained.append(paper)
    added_pool = deduplicate_papers([*retained, *rewrite_candidates])
    candidate_limit = _candidate_limit(baseline_config)
    if len(added_pool) > candidate_limit:
        added_pool = stable_source_coverage_truncate(
            added_pool,
            limit=candidate_limit,
            source_order=[str(item) for item in baseline_config["sources"]],
        )

    removed_metrics, removed_ranked, _ = _rank_pool(
        prepared.analysis, retained, gold
    )
    added_metrics, added_ranked, _ = _rank_pool(
        prepared.analysis, added_pool, gold
    )
    baseline_metrics = prepared.baseline_metrics
    baseline_ranked = prepared.baseline_ranked
    baseline_gold = set(baseline_metrics["returned_gold_ids"])
    removed_gold = set(removed_metrics["returned_gold_ids"])
    added_gold = set(added_metrics["returned_gold_ids"])
    return {
        "baseline": baseline_metrics,
        "remove_replaced": removed_metrics,
        "remove_replaced_add_rewrite": added_metrics,
        "removed_candidate_count": len(removed),
        "rewrite_candidate_count": len(rewrite_candidates),
        "replacement_lost_gold_ids": sorted(baseline_gold - removed_gold),
        "rewrite_added_gold_ids": sorted(added_gold - removed_gold),
        "rewrite_removed_gold_ids": sorted(removed_gold - added_gold),
        "baseline_top20_ids": [canonical_paper_id(item) for item in baseline_ranked],
        "remove_replaced_top20_ids": [canonical_paper_id(item) for item in removed_ranked],
        "add_rewrite_top20_ids": [canonical_paper_id(item) for item in added_ranked],
    }


def rewrite_lists_by_source(
    source_rows: Sequence[dict[str, Any]],
    comparable_lists: Sequence[_QueryList],
) -> list[_QueryList | None]:
    by_source = {item.source: item for item in comparable_lists}
    return [by_source.get(str(row["source"])) for row in source_rows]


def reconstruct_candidate_pool(
    row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
) -> list[Paper]:
    planning = row["stage_diagnostics"]["initial_query_planning"]["planning"]
    sources = [str(item) for item in config["sources"]]
    outputs: list[Paper] = []
    for subquery in planning["selected_subqueries"]:
        raw: list[Paper] = []
        for source in sources:
            observation = _query_list(
                row,
                config,
                store,
                str(subquery["query"]),
                source,
            )
            raw.extend(observation.papers)
        outputs.extend(deduplicate_papers(raw))
    candidates = deduplicate_papers(outputs)
    limit = _candidate_limit(config)
    if len(candidates) > limit:
        candidates = stable_source_coverage_truncate(
            candidates,
            limit=limit,
            source_order=sources,
        )
    return candidates


def stable_source_coverage_truncate(
    papers: Sequence[Paper], *, limit: int, source_order: Sequence[str]
) -> list[Paper]:
    """Mirror the frozen production budget's stable round-robin truncation."""

    ordered_sources = list(
        dict.fromkeys(
            [
                *source_order,
                "openalex",
                "arxiv",
                "semantic_scholar",
                "pubmed",
                "other",
            ]
        )
    )
    buckets: dict[str, list[Paper]] = {source: [] for source in ordered_sources}
    for paper in papers:
        normalized = {
            str(source).strip().casefold().replace("-", "_").replace(" ", "_")
            for source in paper.sources
        }
        bucket = next(
            (source for source in ordered_sources if source in normalized),
            "other",
        )
        buckets[bucket].append(paper)
    selected: list[Paper] = []
    offsets = {source: 0 for source in ordered_sources}
    while len(selected) < limit:
        added = False
        for source in ordered_sources:
            offset = offsets[source]
            if offset >= len(buckets[source]):
                continue
            selected.append(buckets[source][offset])
            offsets[source] += 1
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def classify_counterfactual(counterfactual: dict[str, Any]) -> CaseClassification:
    added = bool(counterfactual["rewrite_added_gold_ids"])
    lost = bool(counterfactual["replacement_lost_gold_ids"])
    if added and lost:
        return "rewrite_added_and_replacement_lost_gold"
    if added:
        return "rewrite_added_gold"
    if lost:
        return "replacement_lost_gold"
    baseline = counterfactual["baseline"]["candidate_ids"]
    final = counterfactual["remove_replaced_add_rewrite"]["candidate_ids"]
    return "candidate_only_change" if baseline != final else "no_marginal_change"


def _rank_pool(
    analysis: QueryAnalysis,
    candidates: list[Paper],
    gold: Sequence[EvalGoldPaper],
) -> tuple[dict[str, Any], list[Paper], list[Paper]]:
    judgements = judge_papers(analysis, candidates, use_llm=False, policy="current_rules")
    ranked = rerank_papers(analysis, judgements, top_k=len(judgements))
    formal = select_ranked_results(
        {"ranked_papers": ranked[:20]},
        policy="highly_and_partial",
    )
    metrics = evaluate_ranking(formal, gold, k_values=[20])
    candidate_gold = matched_paper_ids(candidates, gold)
    returned_gold = matched_paper_ids(formal, gold, k=20)
    denominator = evaluable_gold_count(gold)
    return (
        {
            "candidate_count": len(candidates),
            "candidate_ids": [canonical_paper_id(item) for item in candidates],
            "candidate_gold_ids": candidate_gold,
            "candidate_recall": len(candidate_gold) / denominator if denominator else None,
            "returned_count": len(formal),
            "returned_gold_ids": returned_gold,
            "recall_at_20": metrics.recall_at_k[20],
            "f1_at_20": metrics.f1_at_k[20],
        },
        [item.paper for item in formal],
        [item.paper for item in ranked],
    )


def _stage_candidates(row: dict[str, Any], stage: str) -> list[Any]:
    snapshots = row["stage_diagnostics"]["snapshots"]
    snapshot = next((item for item in snapshots if item["stage"] == stage), None)
    if snapshot is None or snapshot.get("status") != "completed":
        raise ValueError(f"stage snapshot unavailable:{stage}")
    return list(snapshot.get("candidates") or [])


def equivalent_paper_sequences(left: Sequence[Any], right: Sequence[Any]) -> bool:
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right):
        left_profile = build_identity_profile(left_item)
        right_profile = build_identity_profile(right_item)
        if not identity_evidence_from_profiles(left_profile, right_profile).equivalent:
            return False
    return True


def align_papers_to_diagnostics(
    papers: Sequence[Paper], diagnostics: Sequence[Any]
) -> list[Paper]:
    """Restore the frozen pipeline order after concurrent source completion."""

    if len(papers) != len(diagnostics):
        raise ValueError("baseline_candidate_reconstruction_mismatch")
    profiles = [build_identity_profile(paper) for paper in papers]
    used: set[int] = set()
    aligned: list[Paper] = []
    for diagnostic in diagnostics:
        target = build_identity_profile(diagnostic)
        index = next(
            (
                offset
                for offset, profile in enumerate(profiles)
                if offset not in used
                and identity_evidence_from_profiles(profile, target).equivalent
            ),
            None,
        )
        if index is None:
            raise ValueError("baseline_candidate_reconstruction_mismatch")
        used.add(index)
        aligned.append(papers[index])
    return aligned


def _aggregate(case_rows: Sequence[dict[str, Any]], pairs: Sequence[AuditPair]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for pair in pairs:
        rows = [row for row in case_rows if row["dataset"] == pair.name]
        accepted = [row for row in rows if row["accepted_rewrite"]]
        comparable = [row for row in accepted if row["comparable"]]
        source_rows = [source for row in accepted for source in row["source_rows"]]
        comparable_sources = [
            source for source in source_rows if source["status"] == "comparable"
        ]
        source_counterfactuals = [
            source
            for source in comparable_sources
            if source.get("counterfactual") is not None
        ]
        datasets[pair.name] = {
            "case_count": len(rows),
            "accepted_rewrite_case_count": len(accepted),
            "fallback_or_rejected_case_count": len(rows) - len(accepted),
            "fallback_reason_counts": dict(
                sorted(
                    Counter(
                        str(row["fallback_reason"] or "rewrite_not_accepted")
                        for row in rows
                        if not row["accepted_rewrite"]
                    ).items()
                )
            ),
            "comparable_case_count": len(comparable),
            "incomparable_accepted_case_count": len(accepted) - len(comparable),
            "classification_counts": dict(sorted(Counter(row["classification"] for row in rows).items())),
            "accepted_case_classification_counts": dict(sorted(Counter(row["classification"] for row in accepted).items())),
            "source_pair_count": len(source_rows),
            "comparable_source_pair_count": sum(row["status"] == "comparable" for row in source_rows),
            "source_status_counts": dict(sorted(Counter(row["status"] for row in source_rows).items())),
            "source_incomparability_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for row in source_rows
                        if row["status"] != "comparable"
                        for reason in row["reasons"]
                    ).items()
                )
            ),
            "comparable_query_list_contribution": _aggregate_query_lists(
                comparable_sources
            ),
            "source_counterfactual_classification_counts": dict(
                sorted(
                    Counter(
                        str(row["counterfactual_classification"])
                        for row in source_counterfactuals
                    ).items()
                )
            ),
            "rewrite_independent_candidate_count": sum(
                len(source["lists"]["rewrite"]["independent_candidate_ids"])
                for source in source_rows
                if source["status"] == "comparable"
            ),
            "rewrite_independent_gold_count": sum(
                len(source["lists"]["rewrite"]["independent_gold_ids"])
                for source in source_rows
                if source["status"] == "comparable"
            ),
            "replaced_independent_gold_count": sum(
                len(source["lists"]["replaced"]["independent_gold_ids"])
                for source in source_rows
                if source["status"] == "comparable"
            ),
            "source_counterfactual_replacement_lost_gold_count": sum(
                len(row["counterfactual"]["replacement_lost_gold_ids"])
                for row in source_counterfactuals
            ),
            "source_counterfactual_rewrite_added_gold_count": sum(
                len(row["counterfactual"]["rewrite_added_gold_ids"])
                for row in source_counterfactuals
            ),
            "counterfactual": _aggregate_counterfactual(comparable),
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "mode": "offline_frozen_snapshot_replay",
        "network_request_count": 0,
        "snapshot_write_count": 0,
        "inputs": [
            {
                "name": pair.name,
                "baseline_results_sha256": _sha256(pair.baseline_run / "results.jsonl"),
                "rewrite_results_sha256": _sha256(pair.rewrite_run / "results.jsonl"),
                "baseline_manifest_sha256": _sha256(pair.baseline_snapshot / "manifest.json"),
                "rewrite_manifest_sha256": _sha256(pair.rewrite_snapshot / "manifest.json"),
            }
            for pair in pairs
        ],
        "datasets": datasets,
    }


def _aggregate_counterfactual(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    conditions = ("baseline", "remove_replaced", "remove_replaced_add_rewrite")
    result: dict[str, Any] = {"case_count": len(rows)}
    for condition in conditions:
        metrics = [row["counterfactual"][condition] for row in rows]
        result[condition] = {
            "candidate_recall": _average_optional(item["candidate_recall"] for item in metrics),
            "recall_at_20": _average_optional(item["recall_at_20"] for item in metrics),
            "f1_at_20": _average_optional(item["f1_at_20"] for item in metrics),
            "unique_gold_count": len(set().union(*(set(item["returned_gold_ids"]) for item in metrics))) if metrics else 0,
        }
    result["replacement_lost_gold_count"] = sum(
        len(row["counterfactual"]["replacement_lost_gold_ids"]) for row in rows
    )
    result["rewrite_added_gold_count"] = sum(
        len(row["counterfactual"]["rewrite_added_gold_ids"]) for row in rows
    )
    result["rewrite_removed_gold_count"] = sum(
        len(row["counterfactual"]["rewrite_removed_gold_ids"]) for row in rows
    )
    return result


def _aggregate_query_lists(source_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("original", "replaced", "rewrite"):
        lists = [row["lists"][label] for row in source_rows]
        raw_count = sum(int(item["raw_candidate_count"]) for item in lists)
        unique_count = sum(int(item["unique_candidate_count"]) for item in lists)
        first_ranks = [
            int(item["first_gold_rank"])
            for item in lists
            if item["first_gold_rank"] is not None
        ]
        result[label] = {
            "source_pair_count": len(lists),
            "raw_candidate_count": raw_count,
            "unique_candidate_count": unique_count,
            "duplicate_ratio": (
                max(0, raw_count - unique_count) / raw_count if raw_count else 0.0
            ),
            "independent_candidate_count": sum(
                len(item["independent_candidate_ids"]) for item in lists
            ),
            "gold_hit_count": sum(len(item["gold_ids"]) for item in lists),
            "independent_gold_hit_count": sum(
                len(item["independent_gold_ids"]) for item in lists
            ),
            "first_gold_rank_count": len(first_ranks),
            "minimum_first_gold_rank": min(first_ranks) if first_ranks else None,
            "average_first_gold_rank": (
                sum(first_ranks) / len(first_ranks) if first_ranks else None
            ),
        }
    return result


def _validate_pair_configs(baseline: dict[str, Any], rewrite: dict[str, Any]) -> None:
    frozen_fields = (
        "dataset",
        "dataset_sha256",
        "case_ids",
        "sources",
        "top_k",
        "run_profile",
        "query_adapter_policy",
        "ranking_policy",
        "result_policy",
        "judgement_policy",
    )
    mismatched = [field for field in frozen_fields if baseline.get(field) != rewrite.get(field)]
    if mismatched:
        raise ValueError(f"frozen config mismatch:{','.join(mismatched)}")
    if baseline.get("query_planning_policy") != "current_rules":
        raise ValueError("baseline must use current_rules")
    if rewrite.get("query_planning_policy") != "llm_constrained_rewrite":
        raise ValueError("rewrite run must use llm_constrained_rewrite")


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row["case_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate case rows:{path}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object:{path}")
    return value


def _error_type(message: str | None) -> str | None:
    if not message:
        return None
    value = message.casefold()
    for label in ("http_429", "timeout", "cancelled", "source_failure"):
        if label in value:
            return label
    return "failed"


def _average_optional(values: Iterable[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return sum(selected) / len(selected) if selected else None


def _candidate_limit(config: dict[str, Any]) -> int:
    for field in ("budgets", "budget"):
        value = config.get(field)
        if isinstance(value, dict) and value.get("max_candidate_papers") is not None:
            return int(value["max_candidate_papers"])
    return 200


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
