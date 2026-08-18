"""Pure-offline local BM25 candidate-to-Top-20 conversion audit.

The audit reads one frozen Benchmark Replay and its retrieval Snapshots.  Gold
is introduced only after the candidate, judgement, and ranking stages have
been reconstructed.  It never imports SearchService or invokes a connector.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rank_bm25 import BM25Okapi

from scholar_agent.agents.reranker import rerank_papers
from scholar_agent.connectors.local_bm25 import tokenize_local_bm25
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
    normalize_s2orc_corpus_id,
    normalize_title,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    JudgementFeatureVector,
    JudgementResult,
    QueryAnalysis,
    RankedPaper,
)
from scholar_agent.evaluation.datasets.beir_scifact import (
    DEFAULT_CROSSWALK_PATH,
    load_beir_scifact_enriched,
)
from scholar_agent.evaluation.llm_rewrite_causal_audit import (
    align_papers_to_diagnostics,
    stable_source_coverage_truncate,
)
from scholar_agent.evaluation.metrics import (
    average_metric_sets,
    canonical_paper_id,
    evaluable_gold_count,
    evaluate_ranking,
    gold_crosswalk_status,
    matched_paper_ids,
)
from scholar_agent.evaluation.snapshots import SnapshotStore


AUDIT_SCHEMA_VERSION = "1"
RETURN_CATEGORIES = {"highly_relevant", "partially_relevant"}
LossClass = Literal[
    "identity_merge_loss",
    "candidate_budget_truncation",
    "weak_or_irrelevant_filter",
    "ranking_outside_top_20",
    "successfully_returned",
]


@dataclass(frozen=True)
class LocalListHit:
    query: str
    query_order: int
    source_rank: int
    bm25_score: float
    snapshot_key: str | None


class FrozenLocalBM25Scorer:
    """In-memory scorer matching ``local_bm25-v1`` without cache writes."""

    def __init__(self, corpus_path: str | Path) -> None:
        path = Path(corpus_path).expanduser().resolve()
        rows: dict[str, tuple[str, str]] = {}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"invalid local BM25 corpus row:{line_number}")
            document_id = str(payload.get("_id") or "").strip()
            title = str(payload.get("title") or "").strip()
            abstract = str(payload.get("text") or "").strip()
            if not document_id or not title:
                raise ValueError(f"incomplete local BM25 corpus row:{line_number}")
            signature = (title, abstract)
            prior = rows.get(document_id)
            if prior is not None and prior != signature:
                raise ValueError(f"conflicting local BM25 document:{document_id}")
            rows[document_id] = signature
        if not rows:
            raise ValueError("empty local BM25 corpus")
        self.document_ids = sorted(rows)
        tokenized = [
            tokenize_local_bm25(f"{rows[document_id][0]} {rows[document_id][1]}")
            for document_id in self.document_ids
        ]
        self.engine = BM25Okapi(tokenized, k1=1.5, b=0.75, epsilon=0.25)
        self._score_cache: dict[str, dict[str, float]] = {}

    def score(self, query: str, document_id: str | int | None) -> float:
        normalized_id = normalize_s2orc_corpus_id(document_id)
        if normalized_id is None:
            raise ValueError("local candidate lacks S2ORC Corpus ID")
        if query not in self._score_cache:
            values = self.engine.get_scores(tokenize_local_bm25(query))
            self._score_cache[query] = {
                document_id: float(values[index])
                for index, document_id in enumerate(self.document_ids)
            }
        try:
            return self._score_cache[query][normalized_id]
        except KeyError as exc:
            raise ValueError(f"local candidate absent from corpus:{normalized_id}") from exc


def classify_gold_terminal(
    *,
    retrieval_match: bool,
    deduplicated_match: bool,
    identity_merge_evidence: bool,
    budget_truncated: bool,
    category: str | None,
    current_rank: int | None,
    final_returned: bool,
    top_k: int = 20,
) -> LossClass:
    """Assign exactly one observable terminal to a retrieved gold relation."""

    if not retrieval_match:
        raise ValueError("candidate gold was not retrieved")
    if not deduplicated_match:
        if identity_merge_evidence:
            return "identity_merge_loss"
        if budget_truncated:
            return "candidate_budget_truncation"
        raise ValueError("candidate disappeared without merge or budget evidence")
    if category not in RETURN_CATEGORIES:
        return "weak_or_irrelevant_filter"
    if current_rank is None or current_rank > top_k:
        return "ranking_outside_top_20"
    if final_returned:
        return "successfully_returned"
    raise ValueError("eligible Top-20 gold missing from final returned stage")


def rank_by_local_best(
    candidates: Sequence[Any],
    *,
    query_order: dict[str, int],
    top_k: int = 20,
) -> list[Any]:
    """Rank local candidates by best source rank with deterministic ties."""

    rows: list[tuple[tuple[int, int, str], Any]] = []
    for candidate in candidates:
        local = [
            item
            for item in _value(candidate, "provenance", [])
            if str(item.get("source") or "") == "local_bm25"
            and _positive_int(item.get("source_rank")) is not None
        ]
        if not local:
            continue
        best = min(
            local,
            key=lambda item: (
                int(item["source_rank"]),
                query_order.get(str(item.get("adapted_query") or ""), 10**9),
                str(item.get("adapted_query") or ""),
            ),
        )
        stable_id = canonical_paper_id(candidate) or (
            "title:" + normalize_title(str(_value(candidate, "title", "")))
        )
        rows.append(
            (
                (
                    int(best["source_rank"]),
                    query_order.get(str(best.get("adapted_query") or ""), 10**9),
                    stable_id,
                ),
                candidate,
            )
        )
    rows.sort(key=lambda item: item[0])
    return [item[1] for item in rows[:top_k]]


def gold_first_oracle(
    candidates: Sequence[Any],
    gold: Sequence[EvalGoldPaper],
    *,
    top_k: int = 20,
) -> list[Any]:
    """Gold-prioritized candidate-pool upper bound; evaluator-only by design."""

    gold_profiles = [build_identity_profile(item) for item in gold]

    def key(candidate: Any) -> tuple[int, int, str]:
        profile = build_identity_profile(candidate)
        is_gold = any(
            identity_evidence_from_profiles(profile, target).equivalent
            for target in gold_profiles
        )
        rank = _positive_int(_value(candidate, "rank")) or 10**9
        stable_id = canonical_paper_id(candidate) or (
            "title:" + normalize_title(str(_value(candidate, "title", "")))
        )
        return (0 if is_gold else 1, rank, stable_id)

    return sorted(candidates, key=key)[:top_k]


def run_conversion_audit(
    *,
    baseline_run_dir: str | Path,
    run_dir: str | Path,
    snapshot_dir: str | Path,
    corpus_path: str | Path,
    crosswalk_path: str | Path = DEFAULT_CROSSWALK_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct the frozen run and return case rows, gold chains, aggregate."""

    baseline_root = Path(baseline_run_dir).expanduser().resolve()
    run_root = Path(run_dir).expanduser().resolve()
    snapshot_root = Path(snapshot_dir).expanduser().resolve()
    corpus = Path(corpus_path).expanduser().resolve()
    crosswalk = Path(crosswalk_path).expanduser().resolve()
    config = _read_json(run_root / "config.json")
    _validate_config(config, corpus)
    results = _read_rows(run_root / "results.jsonl")
    baseline_config = _read_json(baseline_root / "config.json")
    baseline_results = _read_rows(baseline_root / "results.jsonl")
    _validate_pair(baseline_config, config, baseline_results, results)
    queries = load_beir_scifact_enriched(
        str(config["dataset_source_path"]), crosswalk_path=crosswalk
    )
    by_case = {item.query_id: item for item in queries}
    case_ids = [str(item) for item in config["case_ids"]]
    if set(results) != set(case_ids) or any(case_id not in by_case for case_id in case_ids):
        raise ValueError("frozen SciFact case set mismatch")
    scorer = FrozenLocalBM25Scorer(corpus)
    store = SnapshotStore(snapshot_root)
    case_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for case_order, case_id in enumerate(case_ids):
        case_row, chains = _audit_case(
            case_order=case_order,
            eval_query=by_case[case_id],
            row=results[case_id],
            config=config,
            store=store,
            scorer=scorer,
        )
        case_rows.append(case_row)
        gold_rows.extend(chains)
    candidate_chains = [row for row in gold_rows if row["candidate_gold"]]
    if len(candidate_chains) != 32:
        raise ValueError(f"expected 32 candidate gold relations, got {len(candidate_chains)}")
    if any(not row["local_list_hits"] for row in candidate_chains):
        raise ValueError("candidate gold lacks local_bm25 provenance")
    aggregate = _aggregate(
        case_rows,
        candidate_chains,
        inputs={
            "run_config_sha256": _sha256(run_root / "config.json"),
            "run_results_sha256": _sha256(run_root / "results.jsonl"),
            "baseline_config_sha256": _sha256(baseline_root / "config.json"),
            "baseline_results_sha256": _sha256(baseline_root / "results.jsonl"),
            "snapshot_manifest_sha256": _sha256(snapshot_root / "manifest.json"),
            "corpus_sha256": _sha256(corpus),
            "crosswalk_sha256": _sha256(crosswalk),
        },
    )
    return case_rows, candidate_chains, aggregate


def write_conversion_audit(
    output: str | Path,
    case_rows: Sequence[dict[str, Any]],
    gold_rows: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "case_audit.jsonl", case_rows)
    _write_jsonl(root / "gold_chains.jsonl", gold_rows)
    _atomic_write_json(root / "aggregate.json", aggregate)


def _audit_case(
    *,
    case_order: int,
    eval_query: EvalQuery,
    row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
    scorer: FrozenLocalBM25Scorer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if row.get("status") != "succeeded":
        raise ValueError(f"frozen case is not succeeded:{eval_query.query_id}")
    snapshots = {
        str(item["stage"]): item for item in row["stage_diagnostics"]["snapshots"]
    }
    required = {
        "initial_retrieval",
        "initial_deduplicated",
        "initial_judged",
        "initial_reranked",
        "final_ranked",
        "final_returned",
    }
    if not required.issubset(snapshots):
        raise ValueError(f"missing frozen stage:{eval_query.query_id}")
    initial = snapshots["initial_retrieval"]
    dedup = snapshots["initial_deduplicated"]
    judged = snapshots["initial_judged"]
    reranked = snapshots["initial_reranked"]
    final_ranked = snapshots["final_ranked"]
    final_returned = snapshots["final_returned"]
    if any(snapshots[name].get("status") != "completed" for name in required):
        raise ValueError(f"incomplete frozen stage:{eval_query.query_id}")

    query_order, snapshot_keys = _local_query_order(initial)
    full_candidates = _reconstruct_candidates(initial, dedup, config, store)
    ranked_models = _reconstruct_ranking(full_candidates, judged, reranked, row)
    ranked_by_id = {
        _candidate_key(item.paper): item for item in ranked_models
    }
    reranked_candidates = list(reranked.get("candidates") or [])
    current = list(final_returned.get("candidates") or [])
    no_filter = reranked_candidates[: int(config["top_k"])]
    local_rank = rank_by_local_best(
        reranked_candidates, query_order=query_order, top_k=int(config["top_k"])
    )
    oracle = gold_first_oracle(
        reranked_candidates, eval_query.gold_papers, top_k=int(config["top_k"])
    )
    variants = {
        "current": current,
        "current_ranking_without_relevance_filter": no_filter,
        "local_bm25_best_rank_top20": local_rank,
        "gold_first_candidate_pool_oracle": oracle,
    }
    metric_rows = {
        name: _case_metrics(values, eval_query.gold_papers)
        for name, values in variants.items()
    }
    candidate_metrics = _candidate_metrics(
        reranked_candidates, eval_query.gold_papers
    )
    budget_truncated = bool(
        row["result"]["budget_status"].get("candidate_truncations")
    )
    chains: list[dict[str, Any]] = []
    for gold_index, gold in enumerate(eval_query.gold_papers):
        if gold.relevance_grade <= 0 or gold_crosswalk_status(gold) != "success":
            continue
        retrieval_matches = _matching_candidates(initial["candidates"], gold)
        local_retrieval = [
            item
            for item in retrieval_matches
            if any(
                str(provenance.get("source") or "") == "local_bm25"
                for provenance in item.get("provenance") or []
            )
        ]
        dedup_matches = _matching_candidates(dedup["candidates"], gold)
        judged_matches = _matching_candidates(judged["candidates"], gold)
        reranked_matches = _matching_candidates(reranked_candidates, gold)
        final_ranked_matches = _matching_candidates(final_ranked["candidates"], gold)
        returned_matches = _matching_candidates(final_returned["candidates"], gold)
        candidate_gold = bool(local_retrieval)
        if not candidate_gold:
            continue
        local_hits = _local_hits(
            local_retrieval,
            query_order=query_order,
            snapshot_keys=snapshot_keys,
            scorer=scorer,
        )
        dedup_match = dedup_matches[0] if dedup_matches else None
        judged_match = judged_matches[0] if judged_matches else None
        reranked_match = reranked_matches[0] if reranked_matches else None
        final_ranked_match = final_ranked_matches[0] if final_ranked_matches else None
        merge_evidence = _identity_merge_evidence(local_retrieval, dedup)
        current_rank = _positive_int(_value(reranked_match, "rank"))
        category = _optional_string(_value(judged_match, "category"))
        loss = classify_gold_terminal(
            retrieval_match=True,
            deduplicated_match=dedup_match is not None,
            identity_merge_evidence=bool(merge_evidence),
            budget_truncated=budget_truncated,
            category=category,
            current_rank=current_rank,
            final_returned=bool(returned_matches),
            top_k=int(config["top_k"]),
        )
        ranked_model = (
            ranked_by_id.get(_candidate_key(reranked_match))
            if reranked_match is not None
            else None
        )
        chains.append(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "case_order": case_order,
                "case_id": eval_query.query_id,
                "gold_index": gold_index,
                "gold_id": canonical_paper_id(gold),
                "gold_title": gold.title,
                "candidate_gold": True,
                "local_list_hits": [
                    {
                        "query": item.query,
                        "query_order": item.query_order,
                        "source_rank": item.source_rank,
                        "bm25_score": item.bm25_score,
                        "snapshot_key": item.snapshot_key,
                    }
                    for item in local_hits
                ],
                "best_local_rank": local_hits[0].source_rank,
                "best_local_bm25_score": local_hits[0].bm25_score,
                "deduplicated_pool_position": (
                    _candidate_position(dedup["candidates"], dedup_match)
                    if dedup_match is not None
                    else None
                ),
                "identity_merge_evidence": merge_evidence,
                "judgement_score": _value(judged_match, "judgement_score"),
                "judgement_category": category,
                "judgement_score_components": _value(
                    _value(judged_match, "judgement_features", {}),
                    "score_components",
                    {},
                ),
                "rerank_score_breakdown": (
                    ranked_model.score_breakdown.model_dump(mode="json")
                    if ranked_model is not None
                    else None
                ),
                "current_rank": current_rank,
                "final_rank": _positive_int(_value(final_ranked_match, "rank")),
                "returned_rank": (
                    _positive_int(_value(returned_matches[0], "rank"))
                    if returned_matches
                    else None
                ),
                "loss_class": loss,
                "recovered_by": sorted(
                    name
                    for name, values in variants.items()
                    if name != "current" and _matches_any(values, gold)
                ),
            }
        )

    return (
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "case_order": case_order,
            "case_id": eval_query.query_id,
            "evaluable_gold_count": evaluable_gold_count(eval_query.gold_papers),
            "candidate": candidate_metrics,
            "variants": metric_rows,
            "local_query_count": len(query_order),
            "candidate_count": len(reranked_candidates),
            "budget_truncated": budget_truncated,
        },
        chains,
    )


def _reconstruct_candidates(
    initial: dict[str, Any],
    dedup: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
) -> list[Paper]:
    raw: list[Paper] = []
    seen_keys: set[str] = set()
    for call in initial.get("retrieval_calls") or []:
        key = _optional_string(call.get("snapshot_key"))
        if not call.get("logical_call_executed") or not key or key in seen_keys:
            continue
        seen_keys.add(key)
        entry = store.read_retrieval(key)
        if (
            entry.source != call.get("source")
            or entry.adapted_query != call.get("adapted_query")
            or entry.status != call.get("terminal_status")
        ):
            raise ValueError(f"frozen Snapshot request mismatch:{key}")
        if entry.status == "success":
            raw.extend(item.model_copy(deep=True) for item in entry.papers)
    candidates = deduplicate_papers(raw)
    limit = int(config["budgets"]["max_candidate_papers"])
    if len(candidates) > limit:
        candidates = stable_source_coverage_truncate(
            candidates,
            limit=limit,
            source_order=[str(item) for item in config["sources"]],
        )
    return align_papers_to_diagnostics(candidates, dedup["candidates"])


def _reconstruct_ranking(
    candidates: Sequence[Paper],
    judged: dict[str, Any],
    reranked: dict[str, Any],
    row: dict[str, Any],
) -> list[RankedPaper]:
    judged_candidates = list(judged.get("candidates") or [])
    if len(candidates) != len(judged_candidates):
        raise ValueError("frozen judgement candidate count mismatch")
    judgements: list[JudgementResult] = []
    for paper, diagnostic in zip(candidates, judged_candidates):
        if not _equivalent(paper, diagnostic):
            raise ValueError("frozen judgement candidate order mismatch")
        feature = diagnostic.get("judgement_features")
        judgements.append(
            JudgementResult(
                paper=paper,
                score=float(diagnostic["judgement_score"]),
                category=str(diagnostic["category"]),
                reasoning="frozen_stage_diagnostic",
                matched_terms=list(diagnostic.get("matched_terms") or []),
                warnings=list(diagnostic.get("warnings") or []),
                feature_vector=(
                    JudgementFeatureVector.model_validate(feature) if feature else None
                ),
            )
        )
    analysis = QueryAnalysis.model_validate(
        row["stage_diagnostics"]["initial_query_planning"]["query_analysis"]
    )
    ranked = rerank_papers(analysis, judgements, top_k=len(judgements))
    frozen = list(reranked.get("candidates") or [])
    if len(ranked) != len(frozen):
        raise ValueError("frozen rerank count mismatch")
    for live, expected in zip(ranked, frozen):
        if (
            not _equivalent(live, expected)
            or live.rank != int(expected["rank"])
            or live.category != expected["category"]
            or live.final_score != float(expected["final_score"])
        ):
            raise ValueError("frozen rerank reconstruction mismatch")
    return ranked


def _local_query_order(
    initial: dict[str, Any],
) -> tuple[dict[str, int], dict[str, str]]:
    order: dict[str, int] = {}
    keys: dict[str, str] = {}
    for call in initial.get("retrieval_calls") or []:
        if (
            str(call.get("source") or "") != "local_bm25"
            or not call.get("logical_call_executed")
            or call.get("terminal_status") != "success"
        ):
            continue
        query = str(call.get("adapted_query") or "")
        if query not in order:
            order[query] = len(order)
        if call.get("snapshot_key"):
            keys[query] = str(call["snapshot_key"])
    return order, keys


def _local_hits(
    retrieval_matches: Sequence[dict[str, Any]],
    *,
    query_order: dict[str, int],
    snapshot_keys: dict[str, str],
    scorer: FrozenLocalBM25Scorer,
) -> list[LocalListHit]:
    by_key: dict[tuple[str, int], LocalListHit] = {}
    for candidate in retrieval_matches:
        corpus_id = _value(_value(candidate, "identifiers", {}), "s2orc_corpus_id")
        for provenance in candidate.get("provenance") or []:
            if str(provenance.get("source") or "") != "local_bm25":
                continue
            rank = _positive_int(provenance.get("source_rank"))
            query = str(provenance.get("adapted_query") or "")
            if rank is None or query not in query_order:
                continue
            by_key[(query, rank)] = LocalListHit(
                query=query,
                query_order=query_order[query],
                source_rank=rank,
                bm25_score=scorer.score(query, corpus_id),
                snapshot_key=snapshot_keys.get(query),
            )
    result = sorted(
        by_key.values(),
        key=lambda item: (item.source_rank, item.query_order, item.query),
    )
    if not result:
        raise ValueError("local gold match lacks an executed local list")
    return result


def _identity_merge_evidence(
    retrieval_matches: Sequence[dict[str, Any]], dedup: dict[str, Any]
) -> list[dict[str, Any]]:
    titles = {
        normalize_title(str(item.get("title") or "")) for item in retrieval_matches
    }
    return [
        item
        for item in dedup.get("identity_audit") or []
        if normalize_title(str(item.get("incoming_title") or "")) in titles
    ]


def _aggregate(
    case_rows: Sequence[dict[str, Any]],
    gold_rows: Sequence[dict[str, Any]],
    *,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    evaluable = [row for row in case_rows if row["evaluable_gold_count"] > 0]
    variant_names = list(evaluable[0]["variants"]) if evaluable else []
    variants = {
        name: _aggregate_variant(evaluable, name) for name in variant_names
    }
    current = variants["current"]
    for name, values in variants.items():
        values["delta_vs_current"] = {
            "recall_at_20": values["recall_at_20"] - current["recall_at_20"],
            "f1_at_20": values["f1_at_20"] - current["f1_at_20"],
            "gold_relations": (
                values["returned_gold_relation_count"]
                - current["returned_gold_relation_count"]
            ),
        }
    no_filter = variants["current_ranking_without_relevance_filter"]
    oracle = variants["gold_first_candidate_pool_oracle"]
    classification = dict(
        sorted(Counter(str(row["loss_class"]) for row in gold_rows).items())
    )
    if sum(classification.values()) != len(gold_rows):
        raise ValueError("gold terminal classification is not closed")
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit": "local_bm25_candidate_conversion",
        "inputs": inputs,
        "case_count": len(case_rows),
        "evaluable_query_count": len(evaluable),
        "evaluable_gold_relation_count": sum(
            int(row["evaluable_gold_count"]) for row in evaluable
        ),
        "candidate_gold_relation_count": len(gold_rows),
        "candidate_gold_unique_paper_count": len(
            {str(row["gold_id"]) for row in gold_rows}
        ),
        "loss_classification": classification,
        "filtered_gold_category_counts": dict(
            sorted(
                Counter(
                    str(row["judgement_category"])
                    for row in gold_rows
                    if row["loss_class"] == "weak_or_irrelevant_filter"
                ).items()
            )
        ),
        "candidate": _aggregate_candidate(evaluable),
        "variants": variants,
        "recoverable_bounds": {
            "immediate_filter_recovery_gold_relations": (
                no_filter["returned_gold_relation_count"]
                - current["returned_gold_relation_count"]
            ),
            "residual_ranking_recovery_after_filter_skip_gold_relations": (
                oracle["returned_gold_relation_count"]
                - no_filter["returned_gold_relation_count"]
            ),
            "combined_candidate_pool_upper_bound_gold_relations": (
                oracle["returned_gold_relation_count"]
                - current["returned_gold_relation_count"]
            ),
            "oracle_is_not_an_achievable_score": True,
        },
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_used_only_after_candidate_reconstruction": True,
            "bm25_score_recomputed_from_frozen_corpus": True,
            "reranker_components_recomputed_from_frozen_judgements": True,
            "paired_baseline_external_terminal_parity": True,
        },
        "diagnostic_conclusion": {
            "primary_observed_loss": "relevance_filtering",
            "pure_ranking_outside_top_20_gold_relations": int(
                classification.get("ranking_outside_top_20", 0)
            ),
            "filtered_gold_relations": int(
                classification.get("weak_or_irrelevant_filter", 0)
            ),
            "recommended_next_audit_priority": "relevance_filter_calibration",
            "production_change_recommended": False,
        },
    }


def _aggregate_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_recall": _average(row["candidate"]["recall"] for row in rows),
        "matched_gold_relation_count": sum(
            int(row["candidate"]["matched_gold_count"]) for row in rows
        ),
        "matched_unique_paper_count": len(
            {
                value
                for row in rows
                for value in row["candidate"]["matched_gold_ids"]
            }
        ),
    }


def _aggregate_variant(
    rows: Sequence[dict[str, Any]], name: str
) -> dict[str, Any]:
    metrics = average_metric_sets(
        [
            evaluate_ranking(
                row["variants"][name]["ranked"],
                row["variants"][name]["gold"],
                k_values=[20],
            )
            for row in rows
        ]
    )
    return {
        "recall_at_20": metrics.recall_at_k[20],
        "f1_at_20": metrics.f1_at_k[20],
        "returned_gold_relation_count": sum(
            int(row["variants"][name]["matched_gold_count"]) for row in rows
        ),
        "returned_unique_paper_count": len(
            {
                value
                for row in rows
                for value in row["variants"][name]["matched_gold_ids"]
            }
        ),
    }


def _case_metrics(
    ranked: Sequence[Any], gold: Sequence[EvalGoldPaper]
) -> dict[str, Any]:
    metric = evaluate_ranking(ranked, gold, k_values=[20])
    matched = matched_paper_ids(ranked, gold, k=20)
    return {
        "recall_at_20": metric.recall_at_k[20],
        "f1_at_20": metric.f1_at_k[20],
        "matched_gold_count": len(matched),
        "matched_gold_ids": matched,
        "ranked_ids": [canonical_paper_id(item) for item in ranked],
        # Retained in-memory-compatible form for exact aggregate recomputation.
        "ranked": list(ranked),
        "gold": list(gold),
    }


def _candidate_metrics(
    candidates: Sequence[Any], gold: Sequence[EvalGoldPaper]
) -> dict[str, Any]:
    matched = matched_paper_ids(candidates, gold)
    denominator = evaluable_gold_count(gold)
    return {
        "recall": len(matched) / denominator if denominator else None,
        "matched_gold_count": len(matched),
        "matched_gold_ids": matched,
    }


def _serializable_case(row: dict[str, Any]) -> dict[str, Any]:
    copy = json.loads(json.dumps(row, default=_json_default))
    for variant in copy.get("variants", {}).values():
        variant.pop("ranked", None)
        variant.pop("gold", None)
    return copy


def _matching_candidates(
    candidates: Sequence[Any], gold: EvalGoldPaper
) -> list[Any]:
    target = build_identity_profile(gold)
    return [
        item
        for item in candidates
        if identity_evidence_from_profiles(build_identity_profile(item), target).equivalent
    ]


def _matches_any(candidates: Sequence[Any], gold: EvalGoldPaper) -> bool:
    return bool(_matching_candidates(candidates, gold))


def _candidate_position(candidates: Sequence[Any], target: Any) -> int | None:
    for index, candidate in enumerate(candidates, 1):
        if _equivalent(candidate, target):
            return index
    return None


def _candidate_key(candidate: Any) -> str:
    value = canonical_paper_id(candidate)
    if value:
        return value
    return "title:" + normalize_title(str(_value(candidate, "title", "")))


def _equivalent(left: Any, right: Any) -> bool:
    return identity_evidence_from_profiles(
        build_identity_profile(left), build_identity_profile(right)
    ).equivalent


def _validate_config(config: dict[str, Any], corpus: Path) -> None:
    expected_sources = [
        "openalex",
        "arxiv",
        "semantic_scholar",
        "pubmed",
        "local_bm25",
    ]
    if config.get("dataset") != "beir_scifact":
        raise ValueError("audit requires beir_scifact")
    if config.get("sources") != expected_sources:
        raise ValueError("audit requires frozen five-source order")
    if config.get("query_planning_policy") != "current_rules":
        raise ValueError("audit requires current_rules")
    if config.get("ranking_policy") != "current_rules":
        raise ValueError("audit requires current_rules ranking")
    if int(config.get("top_k") or 0) != 20:
        raise ValueError("audit requires Top-20")
    if any(
        bool(config.get(field))
        for field in (
            "enable_query_evolution",
            "enable_refchain",
            "enable_semantic_seed_expansion",
        )
    ):
        raise ValueError("audit requires all expansion modules disabled")
    local = config.get("local_bm25") or {}
    if local.get("corpus_sha256") != _sha256(corpus):
        raise ValueError("frozen local BM25 corpus checksum mismatch")
    if local.get("parameters") != {"k1": 1.5, "b": 0.75, "epsilon": 0.25}:
        raise ValueError("frozen local BM25 parameters mismatch")


def _validate_pair(
    baseline_config: dict[str, Any],
    experiment_config: dict[str, Any],
    baseline_rows: dict[str, dict[str, Any]],
    experiment_rows: dict[str, dict[str, Any]],
) -> None:
    if baseline_config.get("sources") != [
        "openalex",
        "arxiv",
        "semantic_scholar",
        "pubmed",
    ]:
        raise ValueError("paired baseline requires frozen four-source order")
    for field in (
        "dataset",
        "dataset_sha256",
        "case_ids",
        "top_k",
        "query_planning_policy",
        "ranking_policy",
        "judgement_policy",
        "judgement_config_hash",
        "query_adapter_policy",
        "budgets",
        "result_policy",
    ):
        if baseline_config.get(field) != experiment_config.get(field):
            raise ValueError(f"paired config mismatch:{field}")
    if set(baseline_rows) != set(experiment_rows):
        raise ValueError("paired result case mismatch")
    for case_id in baseline_config["case_ids"]:
        baseline = baseline_rows[str(case_id)]
        experiment = experiment_rows[str(case_id)]
        left_plan = baseline["stage_diagnostics"]["initial_query_planning"][
            "planning"
        ]["selected_subqueries"]
        right_plan = experiment["stage_diagnostics"]["initial_query_planning"][
            "planning"
        ]["selected_subqueries"]
        if left_plan != right_plan:
            raise ValueError(f"paired query planning mismatch:{case_id}")
        if _external_terminal_rows(baseline) != _external_terminal_rows(experiment):
            raise ValueError(f"paired external retrieval mismatch:{case_id}")


def _external_terminal_rows(row: dict[str, Any]) -> list[tuple[Any, ...]]:
    initial = next(
        item
        for item in row["stage_diagnostics"]["snapshots"]
        if item["stage"] == "initial_retrieval"
    )
    return sorted(
        (
            str(call.get("source") or ""),
            str(call.get("adapted_query") or ""),
            bool(call.get("logical_call_executed")),
            str(call.get("snapshot_key") or ""),
            str(call.get("terminal_status") or ""),
        )
        for call in initial.get("retrieval_calls") or []
        if str(call.get("source") or "") != "local_bm25"
    )


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


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(
            _serializable_case(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    _atomic_write_text(path, payload)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _average(values: Iterable[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return sum(selected) / len(selected) if selected else None


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _optional_string(value: Any) -> str | None:
    result = str(value).strip() if value is not None else ""
    return result or None


def _value(item: Any, field: str, default: Any = None) -> Any:
    if item is None:
        return default
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"not JSON serializable:{type(value).__name__}")
