"""Pure-offline paired evaluation for original-query local-BM25 deepening.

The experimental policy exists only in this benchmark module.  It consumes
the frozen retrieval Snapshots for every external and derived-query list and
recomputes the original local list from the immutable corpus.  No connector,
Snapshot writer, or production default is imported or changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scholar_agent.agents.judgement import judge_papers
from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
    normalize_s2orc_corpus_id,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import JudgementResult, QueryAnalysis, RankedPaper
from scholar_agent.evaluation.datasets.beir_scifact import load_beir_scifact_enriched
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    align_papers_to_diagnostics,
    equivalent_paper_sequences,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.local_bm25_budget_audit import (
    OfflineLocalBM25Index,
    RankedDocument,
)
from scholar_agent.evaluation.metrics import (
    canonical_paper_id,
    evaluable_gold_count,
    evaluate_ranking,
    gold_crosswalk_status,
    matched_paper_ids,
)
from scholar_agent.evaluation.selection import select_ranked_results
from scholar_agent.evaluation.snapshots import SnapshotStore


BENCHMARK_VERSION = "beir_scifact_local_bm25_original_deepening_v1"
RETURN_CATEGORIES = {"highly_relevant", "partially_relevant"}
GoldTerminal = Literal[
    "baseline_candidate_preserved",
    "newly_returned",
    "deep_candidate_relevance_filtered",
    "deep_candidate_ranked_outside_top_20",
    "deep_candidate_global_budget_truncated",
    "deep_candidate_local_source_truncated",
    "query_mismatch_top_200",
]


@dataclass(frozen=True)
class OriginalDeepeningPolicy:
    """Benchmark-only policy; disabled unless explicitly constructed enabled."""

    enabled: bool = False
    baseline_depth: int = 20
    original_depth: int = 200
    local_source_candidate_limit: int = 200
    global_candidate_limit: int = 200


@dataclass(frozen=True)
class FrozenCall:
    order: int
    source: str
    adapted_query: str
    adaptation_strategy: str
    purposes: tuple[str, ...]
    origin_subqueries: tuple[str, ...]
    snapshot_key: str
    status: str
    limit: int
    papers: tuple[Paper, ...]


@dataclass(frozen=True)
class CandidateBuild:
    candidates: tuple[Paper, ...]
    pre_global_candidates: tuple[Paper, ...]
    local_candidates: tuple[Paper, ...]
    calls: tuple[FrozenCall, ...]
    original_ranking: tuple[RankedDocument, ...]
    original_call: FrozenCall
    pre_global_count: int
    global_truncated_count: int


@dataclass(frozen=True)
class RankedPool:
    judgements: tuple[JudgementResult, ...]
    all_ranked: tuple[RankedPaper, ...]
    returned: tuple[RankedPaper, ...]


def effective_local_list_depth(
    policy: OriginalDeepeningPolicy = OriginalDeepeningPolicy(),
    *,
    source: str,
    purpose: str,
    adaptation_strategy: str,
    recorded_limit: int,
) -> int:
    """Return the offline list depth without changing default behavior."""

    if (
        policy.enabled
        and source == "local_bm25"
        and purpose == "original_query"
        and adaptation_strategy == "safe_original"
    ):
        return policy.original_depth
    return recorded_limit


def classify_gold_conversion(
    *,
    baseline_candidate: bool,
    experimental_candidate: bool,
    original_rank: int | None,
    local_source_retained: bool,
    pre_global_retained: bool,
    judgement_category: str | None,
    final_rank: int | None,
    returned: bool,
) -> GoldTerminal:
    """Assign one observable terminal after the candidate pools are frozen."""

    if baseline_candidate:
        return "baseline_candidate_preserved"
    if original_rank is None or original_rank > 200:
        return "query_mismatch_top_200"
    if not local_source_retained:
        return "deep_candidate_local_source_truncated"
    if not pre_global_retained or not experimental_candidate:
        return "deep_candidate_global_budget_truncated"
    if judgement_category not in RETURN_CATEGORIES:
        return "deep_candidate_relevance_filtered"
    if final_rank is None or final_rank > 20:
        return "deep_candidate_ranked_outside_top_20"
    if returned:
        return "newly_returned"
    raise ValueError("eligible deep candidate missing from returned Top-20")


def run_original_deepening_benchmark(
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run baseline and enabled policy from frozen inputs with zero I/O side effects."""

    repository = Path(__file__).resolve().parents[3]
    manifest_file = _repo_path(repository, manifest_path)
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest)
    inputs = manifest["inputs"]
    corpus = _repo_path(repository, inputs["corpus"])
    crosswalk = _repo_path(repository, inputs["crosswalk"])
    run_root = _repo_path(repository, inputs["frozen_run"])
    snapshot_root = _repo_path(repository, inputs["snapshot_dir"])
    _require_sha(corpus, inputs["corpus_sha256"], "corpus")
    _require_sha(crosswalk, inputs["crosswalk_sha256"], "crosswalk")
    if _directory_content_sha256(snapshot_root / "retrieval") != inputs[
        "retrieval_directory_content_sha256"
    ]:
        raise ValueError("frozen retrieval Snapshot directory drift")
    snapshot_tree_before = _tree_sha256(snapshot_root)

    config = _read_json(run_root / "config.json")
    _validate_config(config, manifest, corpus)
    result_rows = _read_jsonl_by_id(run_root / "results.jsonl")
    case_ids = [str(value) for value in config["case_ids"]]
    if set(result_rows) != set(case_ids):
        raise ValueError("frozen result case set mismatch")
    queries = load_beir_scifact_enriched(
        str(config["dataset_source_path"]), crosswalk_path=crosswalk
    )
    by_case = {query.query_id: query for query in queries}
    if list(by_case) != case_ids:
        raise ValueError("frozen query order mismatch")

    strategy = manifest["strategy"]
    baseline_policy = OriginalDeepeningPolicy(
        enabled=False,
        baseline_depth=int(strategy["baseline_list_depth"]),
        original_depth=int(strategy["experimental_list_depth"]),
        local_source_candidate_limit=int(strategy["local_source_candidate_limit"]),
        global_candidate_limit=int(strategy["global_candidate_limit"]),
    )
    experiment_policy = OriginalDeepeningPolicy(
        enabled=True,
        baseline_depth=baseline_policy.baseline_depth,
        original_depth=baseline_policy.original_depth,
        local_source_candidate_limit=baseline_policy.local_source_candidate_limit,
        global_candidate_limit=baseline_policy.global_candidate_limit,
    )
    index = OfflineLocalBM25Index(corpus)
    store = SnapshotStore(snapshot_root)
    case_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for case_order, case_id in enumerate(case_ids):
        case, candidates, gold = _run_case(
            case_order=case_order,
            eval_query=by_case[case_id],
            result_row=result_rows[case_id],
            config=config,
            store=store,
            index=index,
            baseline_policy=baseline_policy,
            experiment_policy=experiment_policy,
        )
        case_rows.append(case)
        candidate_rows.extend(candidates)
        gold_rows.extend(gold)

    if len(gold_rows) != int(manifest["dataset"]["evaluable_gold_relation_count"]):
        raise ValueError("gold conversion chain count is not closed")
    if len({(row["case_id"], row["gold_index"]) for row in gold_rows}) != len(
        gold_rows
    ):
        raise ValueError("duplicate gold conversion chain")
    aggregate = _aggregate(
        case_rows=case_rows,
        candidate_rows=candidate_rows,
        gold_rows=gold_rows,
        manifest=manifest,
        inputs={
            "manifest_sha256": _sha256(manifest_file),
            "run_config_sha256": _sha256(run_root / "config.json"),
            "run_results_sha256": _sha256(run_root / "results.jsonl"),
            "corpus_sha256": _sha256(corpus),
            "crosswalk_sha256": _sha256(crosswalk),
            "snapshot_tree_sha256_before": snapshot_tree_before,
        },
    )
    snapshot_tree_after = _tree_sha256(snapshot_root)
    if snapshot_tree_after != snapshot_tree_before:
        raise ValueError("Snapshot tree changed during offline benchmark")
    aggregate["execution"]["snapshot_tree_sha256_after"] = snapshot_tree_after
    return case_rows, candidate_rows, gold_rows, aggregate


def _run_case(
    *,
    case_order: int,
    eval_query: EvalQuery,
    result_row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
    index: OfflineLocalBM25Index,
    baseline_policy: OriginalDeepeningPolicy,
    experiment_policy: OriginalDeepeningPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if result_row.get("status") != "succeeded":
        raise ValueError(f"frozen case is not succeeded:{eval_query.query_id}")
    stages = {
        str(stage["stage"]): stage
        for stage in result_row["stage_diagnostics"]["snapshots"]
    }
    required = {
        "initial_retrieval",
        "initial_deduplicated",
        "initial_reranked",
        "final_returned",
    }
    if not required.issubset(stages) or any(
        stages[name].get("status") != "completed" for name in required
    ):
        raise ValueError(f"frozen stage incomplete:{eval_query.query_id}")
    initial = stages["initial_retrieval"]
    analysis = QueryAnalysis.model_validate(
        result_row["stage_diagnostics"]["initial_query_planning"]["query_analysis"]
    )
    selected = list(
        result_row["stage_diagnostics"]["initial_query_planning"]["planning"][
            "selected_subqueries"
        ]
    )
    calls = _load_calls(initial, store)
    baseline = build_candidate_pool(
        calls=calls,
        selected_subqueries=selected,
        source_order=[str(value) for value in config["sources"]],
        index=index,
        policy=baseline_policy,
    )
    experiment = build_candidate_pool(
        calls=calls,
        selected_subqueries=selected,
        source_order=[str(value) for value in config["sources"]],
        index=index,
        policy=experiment_policy,
    )
    frozen_dedup = list(stages["initial_deduplicated"]["candidates"])
    aligned_baseline = align_papers_to_diagnostics(baseline.candidates, frozen_dedup)
    if not equivalent_paper_sequences(aligned_baseline, frozen_dedup):
        raise ValueError(f"baseline candidate reconstruction mismatch:{eval_query.query_id}")
    baseline_ranked = _rank_pool(analysis, aligned_baseline, config)
    _validate_frozen_ranking(
        baseline_ranked,
        stages["initial_reranked"],
        stages["final_returned"],
        eval_query.query_id,
    )
    experiment_ranked = _rank_pool(analysis, list(experiment.candidates), config)

    baseline_metrics = _case_metrics(
        aligned_baseline, baseline_ranked.returned, eval_query.gold_papers
    )
    experiment_metrics = _case_metrics(
        experiment.candidates, experiment_ranked.returned, eval_query.gold_papers
    )
    external_signature = _external_call_signature(calls)
    candidate_diagnostics = _deep_candidate_diagnostics(
        case_order=case_order,
        case_id=eval_query.query_id,
        baseline=baseline,
        experiment=experiment,
        experiment_ranked=experiment_ranked,
        gold=eval_query.gold_papers,
    )
    gold_diagnostics = _gold_diagnostics(
        case_order=case_order,
        eval_query=eval_query,
        baseline=baseline,
        experiment=experiment,
        baseline_ranked=baseline_ranked,
        experiment_ranked=experiment_ranked,
    )
    return (
        {
            "schema_version": "1",
            "case_order": case_order,
            "case_id": eval_query.query_id,
            "evaluable_gold_count": evaluable_gold_count(eval_query.gold_papers),
            "external_call_signature": external_signature,
            "external_candidate_parity": True,
            "derived_local_candidate_parity": _derived_local_signature(
                baseline.calls
            )
            == _derived_local_signature(experiment.calls),
            "baseline": {
                **baseline_metrics,
                "pre_global_candidate_count": baseline.pre_global_count,
                "global_truncated_count": baseline.global_truncated_count,
                "local_source_candidate_count": len(baseline.local_candidates),
            },
            "experiment": {
                **experiment_metrics,
                "pre_global_candidate_count": experiment.pre_global_count,
                "global_truncated_count": experiment.global_truncated_count,
                "local_source_candidate_count": len(experiment.local_candidates),
            },
            "deep_tail_candidate_count": len(candidate_diagnostics),
            "new_global_candidate_count": sum(
                bool(row["new_vs_baseline_global_pool"])
                for row in candidate_diagnostics
            ),
        },
        candidate_diagnostics,
        gold_diagnostics,
    )


def build_candidate_pool(
    *,
    calls: Sequence[FrozenCall],
    selected_subqueries: Sequence[Mapping[str, Any]],
    source_order: Sequence[str],
    index: OfflineLocalBM25Index,
    policy: OriginalDeepeningPolicy = OriginalDeepeningPolicy(),
) -> CandidateBuild:
    """Build one candidate pool while changing only the eligible local list."""

    original_calls = [
        call
        for call in calls
        if call.source == "local_bm25"
        and "original_query" in call.purposes
        and call.adaptation_strategy == "safe_original"
        and call.status == "success"
    ]
    if len(original_calls) != 1:
        raise ValueError("expected one successful original local BM25 call")
    original_call = original_calls[0]
    original_ranking = index.rank(
        original_call.adapted_query, limit=policy.original_depth
    )
    observed = [_paper_id(paper) for paper in original_call.papers]
    expected = [item.corpus_id for item in original_ranking[: original_call.limit]]
    if observed != expected:
        raise ValueError("old local Snapshot is not the BM25 ranking prefix")

    effective: dict[int, tuple[Paper, ...]] = {}
    for call in calls:
        depth = effective_local_list_depth(
            policy,
            source=call.source,
            purpose=(call.purposes[0] if len(call.purposes) == 1 else ""),
            adaptation_strategy=call.adaptation_strategy,
            recorded_limit=call.limit,
        )
        if call is original_call and depth > call.limit:
            effective[call.order] = tuple(
                item.paper.model_copy(deep=True) for item in original_ranking[:depth]
            )
        else:
            effective[call.order] = tuple(
                paper.model_copy(deep=True) for paper in call.papers
            )

    local_raw = _ordered_source_papers(
        calls,
        effective,
        selected_subqueries,
        sources=("local_bm25",),
    )
    local_deduplicated = deduplicate_papers(local_raw)
    local_candidates = local_deduplicated[: policy.local_source_candidate_limit]
    allowed_local_ids = {_paper_id(paper) for paper in local_candidates}

    outputs: list[Paper] = []
    for selected in selected_subqueries:
        query = str(selected.get("query") or "")
        raw: list[Paper] = []
        for source in source_order:
            for call in calls:
                if call.source != source or query not in call.origin_subqueries:
                    continue
                papers = effective[call.order]
                if source == "local_bm25":
                    papers = tuple(
                        paper
                        for paper in papers
                        if _paper_id(paper) in allowed_local_ids
                    )
                raw.extend(paper.model_copy(deep=True) for paper in papers)
        outputs.extend(deduplicate_papers(raw))
    pre_global = deduplicate_papers(outputs)
    candidates = list(pre_global)
    if len(candidates) > policy.global_candidate_limit:
        candidates = stable_source_coverage_truncate(
            candidates,
            limit=policy.global_candidate_limit,
            source_order=source_order,
        )
    return CandidateBuild(
        candidates=tuple(candidates),
        pre_global_candidates=tuple(pre_global),
        local_candidates=tuple(local_candidates),
        calls=tuple(calls),
        original_ranking=tuple(original_ranking),
        original_call=original_call,
        pre_global_count=len(pre_global),
        global_truncated_count=max(0, len(pre_global) - len(candidates)),
    )


def _ordered_source_papers(
    calls: Sequence[FrozenCall],
    effective: Mapping[int, Sequence[Paper]],
    selected_subqueries: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[str],
) -> list[Paper]:
    papers: list[Paper] = []
    for selected in selected_subqueries:
        query = str(selected.get("query") or "")
        for source in sources:
            for call in calls:
                if call.source == source and query in call.origin_subqueries:
                    papers.extend(
                        paper.model_copy(deep=True) for paper in effective[call.order]
                    )
    return papers


def _load_calls(initial: dict[str, Any], store: SnapshotStore) -> list[FrozenCall]:
    calls: list[FrozenCall] = []
    for call in initial.get("retrieval_calls") or []:
        if not call.get("logical_call_executed"):
            continue
        key = str(call.get("snapshot_key") or "")
        if not key:
            raise ValueError("executed frozen call lacks Snapshot key")
        entry = store.read_retrieval(key)
        if (
            entry.source != call.get("source")
            or entry.adapted_query != call.get("adapted_query")
            or entry.status != call.get("terminal_status")
        ):
            raise ValueError(f"frozen Snapshot request mismatch:{key}")
        provenance = [
            item
            for item in call.get("query_provenance") or []
            if isinstance(item, dict)
        ]
        origins = tuple(
            dict.fromkeys(
                str(item.get("origin_subquery") or call.get("origin_subquery") or "")
                for item in provenance or [{}]
            )
        )
        purposes = tuple(
            dict.fromkeys(str(item.get("purpose") or "unknown") for item in provenance)
        ) or ("unknown",)
        calls.append(
            FrozenCall(
                order=len(calls),
                source=entry.source,
                adapted_query=entry.adapted_query,
                adaptation_strategy=str(call.get("adaptation_strategy") or ""),
                purposes=purposes,
                origin_subqueries=origins,
                snapshot_key=key,
                status=entry.status,
                limit=entry.limit,
                papers=tuple(
                    paper.model_copy(deep=True)
                    for paper in (entry.papers if entry.status == "success" else [])
                ),
            )
        )
    return calls


def _rank_pool(
    analysis: QueryAnalysis, candidates: Sequence[Paper], config: Mapping[str, Any]
) -> RankedPool:
    judgements = judge_papers(
        analysis,
        list(candidates),
        use_llm=False,
        policy="current_rules",
    )
    ranked = rerank_papers(analysis, judgements, top_k=len(judgements))
    top = ranked[: int(config["top_k"])]
    returned = select_ranked_results(
        {"ranked_papers": top}, policy=str(config["result_policy"])
    )
    return RankedPool(
        judgements=tuple(judgements),
        all_ranked=tuple(ranked),
        returned=tuple(returned),
    )


def _validate_frozen_ranking(
    reconstructed: RankedPool,
    frozen_reranked: dict[str, Any],
    frozen_returned: dict[str, Any],
    case_id: str,
) -> None:
    expected = list(frozen_reranked.get("candidates") or [])
    if len(reconstructed.all_ranked) != len(expected):
        raise ValueError(f"baseline rerank count mismatch:{case_id}")
    for actual, diagnostic in zip(reconstructed.all_ranked, expected):
        if (
            not _equivalent(actual.paper, diagnostic)
            or actual.rank != int(diagnostic["rank"])
            or actual.category != diagnostic["category"]
            or actual.final_score != float(diagnostic["final_score"])
        ):
            raise ValueError(f"baseline rerank reconstruction mismatch:{case_id}")
    if not equivalent_paper_sequences(
        [item.paper for item in reconstructed.returned],
        list(frozen_returned.get("candidates") or []),
    ):
        raise ValueError(f"baseline returned reconstruction mismatch:{case_id}")


def _case_metrics(
    candidates: Sequence[Paper],
    returned: Sequence[RankedPaper],
    gold: Sequence[EvalGoldPaper],
) -> dict[str, Any]:
    denominator = evaluable_gold_count(gold)
    candidate_gold = matched_paper_ids(candidates, gold)
    returned_gold = matched_paper_ids(returned, gold, k=20)
    metric = evaluate_ranking(returned, gold, k_values=[20])
    return {
        "candidate_count": len(candidates),
        "candidate_recall": (
            len(candidate_gold) / denominator if denominator else None
        ),
        "candidate_gold_ids": candidate_gold,
        "returned_count": len(returned),
        "returned_gold_ids": returned_gold,
        "recall_at_20": metric.recall_at_k[20] if denominator else None,
        "f1_at_20": metric.f1_at_k[20] if denominator else None,
    }


def _deep_candidate_diagnostics(
    *,
    case_order: int,
    case_id: str,
    baseline: CandidateBuild,
    experiment: CandidateBuild,
    experiment_ranked: RankedPool,
    gold: Sequence[EvalGoldPaper],
) -> list[dict[str, Any]]:
    baseline_ids = set(_s2_positions(baseline.candidates))
    local_positions = _s2_positions(experiment.local_candidates)
    pre_global_positions = _s2_positions(experiment.pre_global_candidates)
    global_positions = _s2_positions(experiment.candidates)
    judgement_by_id = {
        _paper_id(item.paper): item
        for item in experiment_ranked.judgements
        if _optional_paper_id(item.paper) is not None
    }
    ranked_by_id = {
        _paper_id(item.paper): item
        for item in experiment_ranked.all_ranked
        if _optional_paper_id(item.paper) is not None
    }
    returned_ids = {
        _paper_id(item.paper)
        for item in experiment_ranked.returned
        if _optional_paper_id(item.paper) is not None
    }
    gold_ids = {
        value
        for value in (_optional_paper_id(item) for item in gold)
        if value is not None
    }
    rows: list[dict[str, Any]] = []
    for ranked in experiment.original_ranking:
        if ranked.rank <= baseline.original_call.limit:
            continue
        paper = ranked.paper
        corpus_id = _paper_id(paper)
        key = _candidate_key(paper)
        judgement = judgement_by_id.get(corpus_id)
        ranked_model = ranked_by_id.get(corpus_id)
        local_position = local_positions.get(corpus_id)
        pre_global_position = pre_global_positions.get(corpus_id)
        global_position = global_positions.get(corpus_id)
        rows.append(
            {
                "schema_version": "1",
                "case_order": case_order,
                "case_id": case_id,
                "candidate_id": key,
                "original_list_rank": ranked.rank,
                "bm25_score": ranked.score,
                "duplicate_with_baseline_candidate": corpus_id in baseline_ids,
                "local_source_pool_position": local_position,
                "pre_global_position": pre_global_position,
                "global_candidate_position": global_position,
                "budget_terminal": (
                    "local_source_truncated"
                    if local_position is None
                    else "global_budget_truncated"
                    if global_position is None
                    else "global_candidate_retained"
                ),
                "new_vs_baseline_global_pool": corpus_id not in baseline_ids,
                "judgement_score": (
                    judgement.score if judgement is not None else None
                ),
                "judgement_category": (
                    judgement.category if judgement is not None else None
                ),
                "judgement_features": (
                    judgement.feature_vector.model_dump(mode="json")
                    if judgement is not None
                    and judgement.feature_vector is not None
                    else None
                ),
                "final_rank": ranked_model.rank if ranked_model is not None else None,
                "returned": corpus_id in returned_ids,
                "matches_gold": corpus_id in gold_ids,
            }
        )
    return rows


def _gold_diagnostics(
    *,
    case_order: int,
    eval_query: EvalQuery,
    baseline: CandidateBuild,
    experiment: CandidateBuild,
    baseline_ranked: RankedPool,
    experiment_ranked: RankedPool,
) -> list[dict[str, Any]]:
    baseline_positions = _s2_positions(baseline.candidates)
    experimental_positions = _s2_positions(experiment.candidates)
    local_positions = _s2_positions(experiment.local_candidates)
    pre_global_positions = _s2_positions(experiment.pre_global_candidates)
    experiment_by_id = {
        _paper_id(item.paper): item
        for item in experiment_ranked.all_ranked
        if _optional_paper_id(item.paper) is not None
    }
    judgement_by_id = {
        _paper_id(item.paper): item
        for item in experiment_ranked.judgements
        if _optional_paper_id(item.paper) is not None
    }
    baseline_returned_ids = {
        value
        for value in (
            _optional_paper_id(item.paper) for item in baseline_ranked.returned
        )
        if value is not None
    }
    returned_ids = {
        value
        for value in (
            _optional_paper_id(item.paper) for item in experiment_ranked.returned
        )
        if value is not None
    }
    rows: list[dict[str, Any]] = []
    for gold_index, gold in enumerate(eval_query.gold_papers):
        if not _gold_evaluable(gold):
            continue
        corpus_id = _paper_id(gold)
        original_rank = next(
            (
                item.rank
                for item in experiment.original_ranking
                if item.corpus_id == corpus_id
            ),
            None,
        )
        baseline_position = baseline_positions.get(corpus_id)
        experiment_position = experimental_positions.get(corpus_id)
        ranked_match = experiment_by_id.get(corpus_id)
        judgement_match = judgement_by_id.get(corpus_id)
        returned_match = corpus_id in returned_ids
        terminal = classify_gold_conversion(
            baseline_candidate=baseline_position is not None,
            experimental_candidate=experiment_position is not None,
            original_rank=original_rank,
            local_source_retained=corpus_id in local_positions,
            pre_global_retained=corpus_id in pre_global_positions,
            judgement_category=(ranked_match.category if ranked_match else None),
            final_rank=(ranked_match.rank if ranked_match else None),
            returned=returned_match,
        )
        rows.append(
            {
                "schema_version": "1",
                "case_order": case_order,
                "case_id": eval_query.query_id,
                "gold_index": gold_index,
                "gold_id": canonical_paper_id(gold),
                "original_list_rank": original_rank,
                "baseline_candidate_position": baseline_position,
                "baseline_returned": corpus_id in baseline_returned_ids,
                "local_source_pool_position": local_positions.get(corpus_id),
                "pre_global_position": pre_global_positions.get(corpus_id),
                "global_candidate_position": experiment_position,
                "judgement_score": (
                    judgement_match.score if judgement_match else None
                ),
                "judgement_category": (
                    judgement_match.category if judgement_match else None
                ),
                "final_rank": ranked_match.rank if ranked_match else None,
                "returned": returned_match,
                "terminal_class": terminal,
                "deep_gap_candidate": bool(
                    baseline_position is None
                    and original_rank is not None
                    and 20 < original_rank <= 200
                ),
            }
        )
    return rows


def _aggregate(
    *,
    case_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    gold_rows: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    evaluable = [row for row in case_rows if row["evaluable_gold_count"] > 0]
    baseline = _aggregate_variant(evaluable, "baseline")
    experiment = _aggregate_variant(evaluable, "experiment")
    terminals = dict(
        sorted(Counter(str(row["terminal_class"]) for row in gold_rows).items())
    )
    if sum(terminals.values()) != len(gold_rows):
        raise ValueError("gold terminal classification is not closed")
    deep_gap = [row for row in gold_rows if row["deep_gap_candidate"]]
    return {
        "schema_version": "1",
        "benchmark": BENCHMARK_VERSION,
        "scope": manifest["scope"],
        "inputs": dict(inputs),
        "case_count": len(case_rows),
        "evaluable_query_count": len(evaluable),
        "evaluable_gold_relation_count": len(gold_rows),
        "candidate_diagnostic_count": len(candidate_rows),
        "external_candidate_parity_case_count": sum(
            bool(row["external_candidate_parity"]) for row in case_rows
        ),
        "derived_local_candidate_parity_case_count": sum(
            bool(row["derived_local_candidate_parity"]) for row in case_rows
        ),
        "variants": {
            "baseline": baseline,
            "local_bm25_original_deepening": experiment,
            "delta": {
                "candidate_recall": experiment["candidate_recall"]
                - baseline["candidate_recall"],
                "recall_at_20": experiment["recall_at_20"]
                - baseline["recall_at_20"],
                "f1_at_20": experiment["f1_at_20"] - baseline["f1_at_20"],
                "candidate_gold_relations": experiment[
                    "candidate_gold_relation_count"
                ]
                - baseline["candidate_gold_relation_count"],
                "returned_gold_relations": experiment[
                    "returned_gold_relation_count"
                ]
                - baseline["returned_gold_relation_count"],
            },
        },
        "paired_query_outcomes": {
            metric: _paired_outcomes(evaluable, metric)
            for metric in ("candidate_recall", "recall_at_20", "f1_at_20")
        },
        "terminal_classification": terminals,
        "deep_gap_gold": {
            "count": len(deep_gap),
            "terminal_counts": dict(
                sorted(Counter(row["terminal_class"] for row in deep_gap).items())
            ),
            "records": [
                {
                    "case_id": row["case_id"],
                    "gold_index": row["gold_index"],
                    "original_list_rank": row["original_list_rank"],
                    "local_source_pool_position": row["local_source_pool_position"],
                    "pre_global_position": row["pre_global_position"],
                    "global_candidate_position": row["global_candidate_position"],
                    "judgement_category": row["judgement_category"],
                    "final_rank": row["final_rank"],
                    "returned": row["returned"],
                    "terminal_class": row["terminal_class"],
                }
                for row in deep_gap
            ],
        },
        "candidate_budget": {
            "deep_tail_candidate_count": len(candidate_rows),
            "local_source_retained_count": sum(
                row["local_source_pool_position"] is not None
                for row in candidate_rows
            ),
            "pre_global_retained_count": sum(
                row["pre_global_position"] is not None for row in candidate_rows
            ),
            "global_retained_count": sum(
                row["global_candidate_position"] is not None
                for row in candidate_rows
            ),
            "new_global_candidate_count": sum(
                bool(row["new_vs_baseline_global_pool"])
                and row["global_candidate_position"] is not None
                for row in candidate_rows
            ),
            "returned_count": sum(bool(row["returned"]) for row in candidate_rows),
        },
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "snapshot_tree_sha256_after": None,
            "production_connector_modified": False,
            "production_adapter_modified": False,
            "strategy_default_enabled": False,
            "gold_used_after_candidate_ranking": True,
        },
    }


def _aggregate_variant(
    rows: Sequence[dict[str, Any]], variant: str
) -> dict[str, Any]:
    candidate_values = [float(row[variant]["candidate_recall"]) for row in rows]
    recall_values = [float(row[variant]["recall_at_20"]) for row in rows]
    f1_values = [float(row[variant]["f1_at_20"]) for row in rows]
    candidate_gold = [
        value for row in rows for value in row[variant]["candidate_gold_ids"]
    ]
    returned_gold = [
        value for row in rows for value in row[variant]["returned_gold_ids"]
    ]
    return {
        "candidate_recall": sum(candidate_values) / len(candidate_values),
        "recall_at_20": sum(recall_values) / len(recall_values),
        "f1_at_20": sum(f1_values) / len(f1_values),
        "candidate_gold_relation_count": len(candidate_gold),
        "candidate_unique_gold_count": len(set(candidate_gold)),
        "returned_gold_relation_count": len(returned_gold),
        "returned_unique_gold_count": len(set(returned_gold)),
        "average_candidate_count": sum(
            int(row[variant]["candidate_count"]) for row in rows
        )
        / len(rows),
        "average_returned_count": sum(
            int(row[variant]["returned_count"]) for row in rows
        )
        / len(rows),
    }


def _paired_outcomes(
    rows: Sequence[dict[str, Any]], metric: str
) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        baseline = float(row["baseline"][metric])
        experiment = float(row["experiment"][metric])
        counts[
            "improved"
            if experiment > baseline
            else "degraded"
            if experiment < baseline
            else "tied"
        ] += 1
    return {
        name: int(counts.get(name, 0))
        for name in ("improved", "tied", "degraded")
    }


def write_original_deepening_artifacts(
    output_dir: str | Path,
    case_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "case_comparison.jsonl": root / "case_comparison.jsonl",
        "deep_candidates.jsonl": root / "deep_candidates.jsonl",
        "gold_conversion.jsonl": root / "gold_conversion.jsonl",
        "aggregate.json": root / "aggregate.json",
    }
    _write_jsonl(paths["case_comparison.jsonl"], case_rows)
    _write_jsonl(paths["deep_candidates.jsonl"], candidate_rows)
    _write_jsonl(paths["gold_conversion.jsonl"], gold_rows)
    _write_json(paths["aggregate.json"], aggregate)
    return {name: _sha256(path) for name, path in paths.items()}


def _external_call_signature(calls: Sequence[FrozenCall]) -> str:
    payload = [
        {
            "source": call.source,
            "adapted_query": call.adapted_query,
            "key": call.snapshot_key,
            "status": call.status,
            "limit": call.limit,
            "paper_ids": [_candidate_key(paper) for paper in call.papers],
        }
        for call in calls
        if call.source != "local_bm25"
    ]
    return _json_hash(payload)


def _derived_local_signature(calls: Sequence[FrozenCall]) -> str:
    payload = [
        {
            "adapted_query": call.adapted_query,
            "key": call.snapshot_key,
            "status": call.status,
            "paper_ids": [_paper_id(paper) for paper in call.papers],
        }
        for call in calls
        if call.source == "local_bm25" and "original_query" not in call.purposes
    ]
    return _json_hash(payload)


def _optional_paper_id(paper: Any) -> str | None:
    identifiers = (
        paper.get("identifiers")
        if isinstance(paper, Mapping)
        else getattr(paper, "identifiers", None)
    )
    value = (
        identifiers.get("s2orc_corpus_id")
        if isinstance(identifiers, Mapping)
        else getattr(identifiers, "s2orc_corpus_id", None)
    )
    if value is None:
        value = (
            paper.get("s2orc_corpus_id")
            if isinstance(paper, Mapping)
            else getattr(paper, "s2orc_corpus_id", None)
        )
    return normalize_s2orc_corpus_id(value)


def _paper_id(paper: Any) -> str:
    value = _optional_paper_id(paper)
    if value is None:
        raise ValueError("local BM25 candidate lacks S2ORC Corpus ID")
    return value


def _s2_positions(candidates: Sequence[Any]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for position, candidate in enumerate(candidates, start=1):
        value = _optional_paper_id(candidate)
        if value is not None:
            positions.setdefault(value, position)
    return positions


def _candidate_key(candidate: Any) -> str:
    value = canonical_paper_id(candidate)
    if value is None:
        raise ValueError("candidate lacks stable identity")
    return value


def _equivalent(left: Any, right: Any) -> bool:
    return identity_evidence_from_profiles(
        build_identity_profile(left), build_identity_profile(right)
    ).equivalent


def _gold_evaluable(gold: EvalGoldPaper) -> bool:
    return bool(
        gold.relevance_grade > 0
        and gold_crosswalk_status(gold) == "success"
        and normalize_s2orc_corpus_id(gold.s2orc_corpus_id) is not None
    )


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("benchmark") != BENCHMARK_VERSION:
        raise ValueError("unsupported original-deepening manifest")
    strategy = manifest.get("strategy") or {}
    expected = {
        "default_enabled": False,
        "source": "local_bm25",
        "eligible_purpose": "original_query",
        "eligible_adaptation_strategy": "safe_original",
        "baseline_list_depth": 20,
        "experimental_list_depth": 200,
        "derived_query_list_depth": 20,
        "local_source_candidate_limit": 200,
        "global_candidate_limit": 200,
        "top_k": 20,
    }
    for field, value in expected.items():
        if strategy.get(field) != value:
            raise ValueError(f"original-deepening policy drift:{field}")
    invariants = manifest.get("invariants") or {}
    if any(
        int(invariants.get(field, -1)) != 0
        for field in (
            "network_request_count",
            "llm_request_count",
            "snapshot_write_count",
        )
    ):
        raise ValueError("offline execution invariant drift")


def _validate_config(
    config: Mapping[str, Any], manifest: Mapping[str, Any], corpus: Path
) -> None:
    frozen = manifest["frozen_pipeline"]
    for field in (
        "sources",
        "query_planning_policy",
        "query_adapter_policy",
        "judgement_policy",
        "ranking_policy",
        "result_policy",
    ):
        if config.get(field) != frozen[field]:
            raise ValueError(f"frozen config drift:{field}")
    if config.get("dataset") != "beir_scifact" or int(config.get("top_k") or 0) != 20:
        raise ValueError("benchmark requires frozen SciFact Top-20")
    if int(config.get("budgets", {}).get("max_candidate_papers") or 0) != 200:
        raise ValueError("global candidate budget drift")
    if (config.get("local_bm25") or {}).get("corpus_sha256") != _sha256(corpus):
        raise ValueError("local BM25 corpus config mismatch")
    if (config.get("local_bm25") or {}).get("parameters") != frozen["bm25"]:
        raise ValueError("local BM25 parameter drift")
    if any(
        bool(config.get(field))
        for field in (
            "enable_query_evolution",
            "enable_refchain",
            "enable_semantic_seed_expansion",
        )
    ):
        raise ValueError("benchmark requires expansion modules disabled")
    llm = config.get("llm") or {}
    if any(
        bool(llm.get(field))
        for field in (
            "llm_enabled",
            "requested",
            "query_understanding",
            "judgement",
            "semantic_query_planning",
            "constrained_query_rewrite",
        )
    ):
        raise ValueError("benchmark requires all LLM capabilities disabled")


def _repo_path(repository: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object:{path}")
    return payload


def _read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row.get("case_id") or ""): row for row in rows}
    if len(result) != len(rows) or "" in result:
        raise ValueError(f"invalid or duplicate case IDs:{path}")
    return result


def _require_sha(path: Path, expected: str, label: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"{label} checksum drift")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _directory_content_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
    )


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
