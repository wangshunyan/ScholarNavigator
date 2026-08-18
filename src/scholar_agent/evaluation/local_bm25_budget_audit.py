"""Pure-offline query-budget audit for the frozen SciFact local BM25 run.

This module rebuilds BM25 rankings directly from the immutable local corpus.
It reads frozen Benchmark diagnostics and Snapshots only to validate the exact
query/list contract.  It never imports SearchService or invokes a connector.
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

from rank_bm25 import BM25Okapi

from scholar_agent.connectors.local_bm25 import tokenize_local_bm25
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import (
    build_identity_profile,
    identity_evidence_from_profiles,
    normalize_s2orc_corpus_id,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.datasets.beir_scifact import (
    load_beir_scifact_enriched,
)
from scholar_agent.evaluation.metrics import (
    canonical_paper_id,
    gold_crosswalk_status,
)
from scholar_agent.evaluation.snapshots import SnapshotStore


AUDIT_VERSION = "beir_scifact_local_bm25_query_budget_v1"
DEPTHS = (20, 50, 100, 200)
SCOPES = (
    "current_adapter_budget",
    "original_query_top_200",
    "all_subqueries_merge_top_200",
)
GapClass = Literal[
    "current_connector_candidate",
    "per_query_adapter_quota",
    "cross_query_identity_deduplication",
    "local_source_pool_truncation",
    "global_candidate_budget",
    "identity_matching_gap",
    "query_mismatch_top_200",
]


@dataclass(frozen=True)
class RankedDocument:
    paper: Paper
    corpus_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class QueryList:
    order: int
    query: str
    purpose: str
    adaptation_strategy: str
    snapshot_key: str
    adapter_limit: int
    ranking: tuple[RankedDocument, ...]


@dataclass(frozen=True)
class MergedPool:
    papers: tuple[Paper, ...]
    raw_count: int
    deduplicated_count: int
    duplicate_count: int
    truncated_count: int


class OfflineLocalBM25Index:
    """In-memory index matching local-bm25-v1 without cache or connector state."""

    def __init__(self, corpus_path: str | Path) -> None:
        path = Path(corpus_path).expanduser().resolve()
        rows: dict[str, tuple[str, str]] = {}
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"invalid local BM25 corpus row:{line_number}")
            corpus_id = normalize_s2orc_corpus_id(payload.get("_id"))
            title = str(payload.get("title") or "").strip()
            abstract = str(payload.get("text") or "").strip()
            if corpus_id is None or not title:
                raise ValueError(f"incomplete local BM25 corpus row:{line_number}")
            signature = (title, abstract)
            if corpus_id in rows and rows[corpus_id] != signature:
                raise ValueError(f"conflicting local BM25 document:{corpus_id}")
            rows[corpus_id] = signature
        if not rows:
            raise ValueError("empty local BM25 corpus")
        self.document_ids = sorted(rows)
        self.papers = [
            Paper(
                title=rows[corpus_id][0],
                abstract=rows[corpus_id][1],
                sources=["local_bm25"],
                identifiers=PaperIdentifiers(s2orc_corpus_id=corpus_id),
            )
            for corpus_id in self.document_ids
        ]
        tokenized = [
            tokenize_local_bm25(f"{paper.title} {paper.abstract}")
            for paper in self.papers
        ]
        self.engine = BM25Okapi(
            tokenized,
            k1=1.5,
            b=0.75,
            epsilon=0.25,
        )
        self._cache: dict[str, tuple[RankedDocument, ...]] = {}

    def rank(self, query: str, *, limit: int = 200) -> tuple[RankedDocument, ...]:
        normalized_query = str(query).strip()
        if not normalized_query:
            raise ValueError("empty BM25 audit query")
        if normalized_query not in self._cache:
            scores = self.engine.get_scores(tokenize_local_bm25(normalized_query))
            offsets = sorted(
                range(len(self.papers)),
                key=lambda offset: (
                    -float(scores[offset]),
                    self.document_ids[offset],
                ),
            )
            self._cache[normalized_query] = tuple(
                RankedDocument(
                    paper=self.papers[offset].model_copy(deep=True),
                    corpus_id=self.document_ids[offset],
                    rank=rank,
                    score=float(scores[offset]),
                )
                for rank, offset in enumerate(offsets, start=1)
            )
        return self._cache[normalized_query][:limit]


def merge_ranked_lists(
    rankings: Sequence[Sequence[RankedDocument]],
    *,
    per_list_depth: int,
    candidate_limit: int,
) -> MergedPool:
    """Stable list-order merge using the production unified identity deduper."""

    raw = [
        item.paper.model_copy(deep=True)
        for ranking in rankings
        for item in ranking[:per_list_depth]
    ]
    deduplicated = deduplicate_papers(raw)
    selected = deduplicated[:candidate_limit]
    return MergedPool(
        papers=tuple(selected),
        raw_count=len(raw),
        deduplicated_count=len(deduplicated),
        duplicate_count=len(raw) - len(deduplicated),
        truncated_count=max(0, len(deduplicated) - len(selected)),
    )


def scope_depth_curve(
    pools: Mapping[str, Sequence[Paper]],
    queries: Mapping[str, EvalQuery],
    *,
    depths: Sequence[int] = DEPTHS,
) -> dict[str, dict[str, float | int]]:
    """Compute relation and macro-query recall from fixed candidate prefixes."""

    evaluable_queries = [
        query
        for query in queries.values()
        if any(_gold_evaluable(gold) for gold in query.gold_papers)
    ]
    result: dict[str, dict[str, float | int]] = {}
    for depth in depths:
        matched_relations = 0
        matched_unique: set[str] = set()
        macro: list[float] = []
        denominator = 0
        for query in evaluable_queries:
            gold = [item for item in query.gold_papers if _gold_evaluable(item)]
            candidates = list(pools[query.query_id])[:depth]
            matched = [item for item in gold if _matches_any(candidates, item)]
            denominator += len(gold)
            matched_relations += len(matched)
            matched_unique.update(
                canonical_paper_id(item) or f"gold:{query.query_id}:{index}"
                for index, item in enumerate(matched)
            )
            macro.append(len(matched) / len(gold))
        result[str(depth)] = {
            "matched_gold_relation_count": matched_relations,
            "matched_unique_gold_count": len(matched_unique),
            "evaluable_gold_relation_count": denominator,
            "micro_candidate_recall": (
                matched_relations / denominator if denominator else 0.0
            ),
            "macro_candidate_recall": sum(macro) / len(macro) if macro else 0.0,
        }
    return result


def classify_gold_budget_gap(
    *,
    current_scope_hit: bool,
    formal_candidate_hit: bool,
    any_rank_within_adapter_limit: bool,
    current_merged_position: int | None,
    local_source_limit: int,
    formal_retrieval_hit: bool,
    formal_deduplicated_hit: bool,
    original_top_200_hit: bool,
    all_subqueries_top_200_hit: bool,
    exact_corpus_identity_available: bool,
) -> GapClass:
    """Assign one conservative terminal without treating an upper bound as policy."""

    if formal_candidate_hit:
        if not current_scope_hit:
            return "identity_matching_gap"
        return "current_connector_candidate"
    if current_scope_hit:
        if current_merged_position is not None and current_merged_position > local_source_limit:
            return "local_source_pool_truncation"
        if formal_retrieval_hit and not formal_deduplicated_hit:
            return "cross_query_identity_deduplication"
        if formal_deduplicated_hit:
            return "global_candidate_budget"
        return "identity_matching_gap"
    if any_rank_within_adapter_limit:
        return "cross_query_identity_deduplication"
    if original_top_200_hit or all_subqueries_top_200_hit:
        return "per_query_adapter_quota"
    if not exact_corpus_identity_available:
        return "identity_matching_gap"
    return "query_mismatch_top_200"


def run_budget_audit(
    manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    repository = Path(__file__).resolve().parents[3]
    manifest_file = _repo_path(repository, manifest_path)
    manifest = _read_json(manifest_file)
    _validate_manifest(manifest)
    inputs = manifest["inputs"]
    corpus_path = _repo_path(repository, inputs["corpus"])
    crosswalk_path = _repo_path(repository, inputs["crosswalk"])
    run_root = _repo_path(repository, inputs["frozen_run"])
    snapshot_root = _repo_path(repository, inputs["snapshot_dir"])
    _require_sha(corpus_path, inputs["corpus_sha256"], "corpus")
    _require_sha(crosswalk_path, inputs["crosswalk_sha256"], "crosswalk")
    retrieval_hash = _directory_content_sha256(snapshot_root / "retrieval")
    if retrieval_hash != inputs["retrieval_directory_content_sha256"]:
        raise ValueError("frozen retrieval Snapshot directory drift")
    snapshot_tree_before = _tree_sha256(snapshot_root)

    config = _read_json(run_root / "config.json")
    _validate_frozen_config(config, manifest, corpus_path)
    result_rows = _read_jsonl_by_id(run_root / "results.jsonl")
    case_ids = [str(item) for item in config["case_ids"]]
    if set(result_rows) != set(case_ids):
        raise ValueError("frozen result case set mismatch")
    queries = load_beir_scifact_enriched(
        str(config["dataset_source_path"]),
        crosswalk_path=crosswalk_path,
    )
    by_case = {query.query_id: query for query in queries}
    if list(by_case) != case_ids:
        raise ValueError("frozen query order mismatch")
    evaluable_gold = [
        (query.query_id, index, gold)
        for query in queries
        for index, gold in enumerate(query.gold_papers)
        if _gold_evaluable(gold)
    ]
    if len(evaluable_gold) != int(
        manifest["dataset"]["evaluable_gold_relation_count"]
    ):
        raise ValueError("evaluable gold count drift")

    index = OfflineLocalBM25Index(corpus_path)
    store = SnapshotStore(snapshot_root)
    case_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    pools_by_scope: dict[str, dict[str, Sequence[Paper]]] = {
        scope: {} for scope in SCOPES
    }
    for case_order, case_id in enumerate(case_ids):
        case_row, chains, pools = _audit_case(
            case_order=case_order,
            eval_query=by_case[case_id],
            result_row=result_rows[case_id],
            config=config,
            store=store,
            index=index,
            manifest=manifest,
        )
        case_rows.append(case_row)
        gold_rows.extend(chains)
        for scope in SCOPES:
            pools_by_scope[scope][case_id] = pools[scope]

    if len(gold_rows) != len(evaluable_gold):
        raise ValueError("gold budget chain count is not closed")
    chain_keys = {(row["case_id"], row["gold_index"]) for row in gold_rows}
    if len(chain_keys) != len(gold_rows):
        raise ValueError("duplicate gold budget chain")
    aggregate = _aggregate(
        case_rows=case_rows,
        gold_rows=gold_rows,
        pools_by_scope=pools_by_scope,
        queries=by_case,
        manifest=manifest,
        manifest_sha256=_sha256(manifest_file),
        run_config_sha256=_sha256(run_root / "config.json"),
        run_results_sha256=_sha256(run_root / "results.jsonl"),
        corpus_sha256=_sha256(corpus_path),
        crosswalk_sha256=_sha256(crosswalk_path),
        snapshot_tree_sha256=snapshot_tree_before,
    )
    snapshot_tree_after = _tree_sha256(snapshot_root)
    if snapshot_tree_after != snapshot_tree_before:
        raise ValueError("Snapshot tree changed during offline audit")
    aggregate["execution"]["snapshot_tree_sha256_after"] = snapshot_tree_after
    return case_rows, gold_rows, aggregate


def _audit_case(
    *,
    case_order: int,
    eval_query: EvalQuery,
    result_row: dict[str, Any],
    config: dict[str, Any],
    store: SnapshotStore,
    index: OfflineLocalBM25Index,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Sequence[Paper]]]:
    if result_row.get("status") != "succeeded":
        raise ValueError(f"frozen case not succeeded:{eval_query.query_id}")
    snapshots = {
        str(item["stage"]): item
        for item in result_row["stage_diagnostics"]["snapshots"]
    }
    required_stages = {
        "initial_retrieval",
        "initial_deduplicated",
        "initial_reranked",
    }
    if not required_stages.issubset(snapshots):
        raise ValueError(f"frozen case missing stages:{eval_query.query_id}")
    initial = snapshots["initial_retrieval"]
    deduplicated = snapshots["initial_deduplicated"]
    reranked = snapshots["initial_reranked"]
    if any(snapshots[name].get("status") != "completed" for name in required_stages):
        raise ValueError(f"frozen case has incomplete stage:{eval_query.query_id}")

    query_lists = _load_query_lists(initial, store, index)
    if not query_lists:
        raise ValueError(f"frozen case has no local query list:{eval_query.query_id}")
    adapter_limit = int(manifest["frozen_budget"]["per_executed_list_limit"])
    local_limit = int(manifest["frozen_budget"]["local_source_merged_limit"])
    current_pool = merge_ranked_lists(
        [item.ranking for item in query_lists],
        per_list_depth=adapter_limit,
        candidate_limit=local_limit,
    )
    safe_lists = [
        item for item in query_lists if item.adaptation_strategy == "safe_original"
    ]
    all_subqueries_pool = merge_ranked_lists(
        [item.ranking for item in safe_lists],
        per_list_depth=200,
        candidate_limit=local_limit,
    )
    original_ranking = index.rank(eval_query.query, limit=200)
    original_pool = merge_ranked_lists(
        [original_ranking],
        per_list_depth=200,
        candidate_limit=local_limit,
    )
    pools = {
        "current_adapter_budget": current_pool.papers,
        "original_query_top_200": original_pool.papers,
        "all_subqueries_merge_top_200": all_subqueries_pool.papers,
    }

    gold_rows: list[dict[str, Any]] = []
    for gold_index, gold in enumerate(eval_query.gold_papers):
        if not _gold_evaluable(gold):
            continue
        list_hits = [
            {
                "query_order": item.order,
                "purpose": item.purpose,
                "adaptation_strategy": item.adaptation_strategy,
                "adapted_query": item.query,
                "snapshot_key": item.snapshot_key,
                "adapter_limit": item.adapter_limit,
                "first_rank": _first_rank(item.ranking, gold),
                "score": _first_score(item.ranking, gold),
            }
            for item in query_lists
        ]
        current_position = _position(current_pool.papers, gold)
        original_position = _position(original_pool.papers, gold)
        all_position = _position(all_subqueries_pool.papers, gold)
        formal_retrieval = _matching_candidates(initial["candidates"], gold)
        formal_local_retrieval = [
            candidate
            for candidate in formal_retrieval
            if any(
                str(item.get("source") or "") == "local_bm25"
                for item in candidate.get("provenance") or []
            )
        ]
        formal_dedup = _matching_candidates(deduplicated["candidates"], gold)
        formal_candidate = _matching_candidates(reranked["candidates"], gold)
        current_hit = current_position is not None
        formal_candidate_hit = bool(formal_candidate)
        if current_hit != bool(formal_local_retrieval):
            raise ValueError(
                f"offline adapter pool differs from frozen local retrieval:{eval_query.query_id}"
            )
        if current_hit != formal_candidate_hit:
            # A mismatch is retained as an auditable loss only when the frozen
            # stage chain supplies enough evidence to classify it below.
            pass
        exact_identity = normalize_s2orc_corpus_id(gold.s2orc_corpus_id)
        terminal = classify_gold_budget_gap(
            current_scope_hit=current_hit,
            formal_candidate_hit=formal_candidate_hit,
            any_rank_within_adapter_limit=any(
                item["first_rank"] is not None
                and int(item["first_rank"]) <= int(item["adapter_limit"])
                for item in list_hits
            ),
            current_merged_position=current_position,
            local_source_limit=local_limit,
            formal_retrieval_hit=bool(formal_local_retrieval),
            formal_deduplicated_hit=bool(formal_dedup),
            original_top_200_hit=original_position is not None,
            all_subqueries_top_200_hit=all_position is not None,
            exact_corpus_identity_available=(
                exact_identity is not None and exact_identity in index.document_ids
            ),
        )
        gold_rows.append(
            {
                "schema_version": "1",
                "case_order": case_order,
                "case_id": eval_query.query_id,
                "gold_index": gold_index,
                "gold_id": canonical_paper_id(gold),
                "s2orc_corpus_id": exact_identity,
                "query_lists": list_hits,
                "original_query_first_rank": _first_rank(original_ranking, gold),
                "scope_positions": {
                    "current_adapter_budget": current_position,
                    "original_query_top_200": original_position,
                    "all_subqueries_merge_top_200": all_position,
                },
                "formal_stage": {
                    "initial_local_retrieval_position": _position(
                        initial["candidates"], gold, require_local=True
                    ),
                    "deduplicated_position": _position(
                        deduplicated["candidates"], gold
                    ),
                    "global_budget_candidate_position": _position(
                        reranked["candidates"], gold
                    ),
                },
                "terminal_class": terminal,
                "explains_32_to_35_gap": bool(
                    not formal_candidate_hit and original_position is not None
                ),
            }
        )

    return (
        {
            "schema_version": "1",
            "case_order": case_order,
            "case_id": eval_query.query_id,
            "query_list_count": len(query_lists),
            "safe_subquery_list_count": len(safe_lists),
            "adapter_limit": adapter_limit,
            "global_candidate_limit": int(
                config["budgets"]["max_candidate_papers"]
            ),
            "scopes": {
                "current_adapter_budget": _pool_stats(current_pool),
                "original_query_top_200": _pool_stats(original_pool),
                "all_subqueries_merge_top_200": _pool_stats(all_subqueries_pool),
            },
        },
        gold_rows,
        pools,
    )


def _load_query_lists(
    initial: dict[str, Any],
    store: SnapshotStore,
    index: OfflineLocalBM25Index,
) -> list[QueryList]:
    result: list[QueryList] = []
    seen_keys: set[str] = set()
    for call in initial.get("retrieval_calls") or []:
        if (
            str(call.get("source") or "") != "local_bm25"
            or not call.get("logical_call_executed")
        ):
            continue
        key = str(call.get("snapshot_key") or "")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        snapshot = store.read_retrieval(key)
        if (
            snapshot.source != "local_bm25"
            or snapshot.status != "success"
            or snapshot.adapted_query != call.get("adapted_query")
            or call.get("terminal_status") != "success"
        ):
            raise ValueError(f"frozen local Snapshot request mismatch:{key}")
        ranking = index.rank(snapshot.adapted_query, limit=200)
        expected_ids = [item.corpus_id for item in ranking[: snapshot.limit]]
        observed_ids = [_paper_corpus_id(paper) for paper in snapshot.papers]
        if observed_ids != expected_ids:
            raise ValueError(f"offline BM25 ranking differs from Snapshot:{key}")
        provenance = list(call.get("query_provenance") or [])
        purposes = sorted(
            {
                str(item.get("purpose") or "unknown")
                for item in provenance
                if isinstance(item, dict)
            }
        )
        result.append(
            QueryList(
                order=len(result),
                query=snapshot.adapted_query,
                purpose="+".join(purposes) if purposes else "unknown",
                adaptation_strategy=str(call.get("adaptation_strategy") or ""),
                snapshot_key=key,
                adapter_limit=snapshot.limit,
                ranking=ranking,
            )
        )
    return result


def _aggregate(
    *,
    case_rows: Sequence[dict[str, Any]],
    gold_rows: Sequence[dict[str, Any]],
    pools_by_scope: Mapping[str, Mapping[str, Sequence[Paper]]],
    queries: Mapping[str, EvalQuery],
    manifest: dict[str, Any],
    manifest_sha256: str,
    run_config_sha256: str,
    run_results_sha256: str,
    corpus_sha256: str,
    crosswalk_sha256: str,
    snapshot_tree_sha256: str,
) -> dict[str, Any]:
    curves = {
        scope: scope_depth_curve(pools_by_scope[scope], queries) for scope in SCOPES
    }
    classifications = Counter(str(row["terminal_class"]) for row in gold_rows)
    if sum(classifications.values()) != len(gold_rows):
        raise ValueError("gold terminal classification is not closed")
    gap_rows = [row for row in gold_rows if row["explains_32_to_35_gap"]]
    current_200 = curves["current_adapter_budget"]["200"]
    original_200 = curves["original_query_top_200"]["200"]
    return {
        "schema_version": "1",
        "audit": AUDIT_VERSION,
        "inputs": {
            "manifest_sha256": manifest_sha256,
            "run_config_sha256": run_config_sha256,
            "run_results_sha256": run_results_sha256,
            "corpus_sha256": corpus_sha256,
            "crosswalk_sha256": crosswalk_sha256,
            "snapshot_tree_sha256_before": snapshot_tree_sha256,
        },
        "case_count": len(case_rows),
        "evaluable_query_count": sum(
            any(_gold_evaluable(gold) for gold in query.gold_papers)
            for query in queries.values()
        ),
        "evaluable_gold_relation_count": len(gold_rows),
        "gold_chain_count": len(gold_rows),
        "terminal_classification": dict(sorted(classifications.items())),
        "depth_curves": curves,
        "gap_32_to_35": {
            "current_adapter_matched_gold_relations": int(
                current_200["matched_gold_relation_count"]
            ),
            "original_query_top_200_matched_gold_relations": int(
                original_200["matched_gold_relation_count"]
            ),
            "additional_gold_relation_count": len(gap_rows),
            "additional_gold_terminal_counts": dict(
                sorted(Counter(row["terminal_class"] for row in gap_rows).items())
            ),
            "case_gold_keys": [
                {"case_id": row["case_id"], "gold_index": row["gold_index"]}
                for row in gap_rows
            ],
        },
        "budget_impact": {
            scope: {
                "raw_candidate_total": sum(
                    int(row["scopes"][scope]["raw_count"]) for row in case_rows
                ),
                "deduplicated_candidate_total": sum(
                    int(row["scopes"][scope]["deduplicated_count"])
                    for row in case_rows
                ),
                "cross_query_duplicate_total": sum(
                    int(row["scopes"][scope]["duplicate_count"])
                    for row in case_rows
                ),
                "source_pool_truncated_total": sum(
                    int(row["scopes"][scope]["truncated_count"])
                    for row in case_rows
                ),
            }
            for scope in SCOPES
        },
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "production_connector_invoked": False,
            "gold_used_after_candidate_scope_construction": True,
            "snapshot_tree_sha256_after": None,
        },
        "scope_note": manifest["scope"],
    }


def write_budget_audit(
    output_dir: str | Path,
    case_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "case_audit.jsonl": root / "case_audit.jsonl",
        "gold_chains.jsonl": root / "gold_chains.jsonl",
        "aggregate.json": root / "aggregate.json",
    }
    _write_jsonl(paths["case_audit.jsonl"], case_rows)
    _write_jsonl(paths["gold_chains.jsonl"], gold_rows)
    _write_json(paths["aggregate.json"], aggregate)
    return {name: _sha256(path) for name, path in paths.items()}


def _pool_stats(pool: MergedPool) -> dict[str, int]:
    return {
        "raw_count": pool.raw_count,
        "deduplicated_count": pool.deduplicated_count,
        "duplicate_count": pool.duplicate_count,
        "truncated_count": pool.truncated_count,
        "retained_count": len(pool.papers),
    }


def _gold_evaluable(gold: EvalGoldPaper) -> bool:
    return bool(
        gold.relevance_grade > 0
        and gold_crosswalk_status(gold) == "success"
        and normalize_s2orc_corpus_id(gold.s2orc_corpus_id) is not None
    )


def _matches(candidate: Any, gold: EvalGoldPaper) -> bool:
    return identity_evidence_from_profiles(
        build_identity_profile(candidate),
        build_identity_profile(gold),
    ).equivalent


def _matches_any(candidates: Sequence[Any], gold: EvalGoldPaper) -> bool:
    target = build_identity_profile(gold)
    return any(
        identity_evidence_from_profiles(build_identity_profile(item), target).equivalent
        for item in candidates
    )


def _matching_candidates(
    candidates: Sequence[Any], gold: EvalGoldPaper
) -> list[Any]:
    return [item for item in candidates if _matches(item, gold)]


def _position(
    candidates: Sequence[Any],
    gold: EvalGoldPaper,
    *,
    require_local: bool = False,
) -> int | None:
    for position, item in enumerate(candidates, start=1):
        if require_local and not any(
            str(provenance.get("source") or "") == "local_bm25"
            for provenance in _value(item, "provenance", [])
        ):
            continue
        if _matches(item, gold):
            return position
    return None


def _first_rank(
    ranking: Sequence[RankedDocument], gold: EvalGoldPaper
) -> int | None:
    match = next((item for item in ranking if _matches(item.paper, gold)), None)
    return match.rank if match else None


def _first_score(
    ranking: Sequence[RankedDocument], gold: EvalGoldPaper
) -> float | None:
    match = next((item for item in ranking if _matches(item.paper, gold)), None)
    return match.score if match else None


def _paper_corpus_id(paper: Paper) -> str:
    value = normalize_s2orc_corpus_id(paper.identifiers.s2orc_corpus_id)
    if value is None:
        raise ValueError("local Snapshot paper lacks S2ORC Corpus ID")
    return value


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("audit") != AUDIT_VERSION:
        raise ValueError("unsupported local BM25 budget audit manifest")
    if tuple(manifest.get("frozen_budget", {}).get("depths") or []) != DEPTHS:
        raise ValueError("audit depth policy drift")
    if tuple(manifest.get("audit_scopes") or {}) != SCOPES:
        raise ValueError("audit scope order drift")
    invariants = manifest.get("invariants") or {}
    if any(
        int(invariants.get(field, -1)) != 0
        for field in (
            "network_request_count",
            "llm_request_count",
            "snapshot_write_count",
        )
    ):
        raise ValueError("audit offline invariant drift")


def _validate_frozen_config(
    config: dict[str, Any], manifest: dict[str, Any], corpus: Path
) -> None:
    if config.get("dataset") != "beir_scifact":
        raise ValueError("audit requires frozen SciFact run")
    if config.get("query_planning_policy") != "current_rules":
        raise ValueError("audit requires current_rules")
    if config.get("query_adapter_policy") != "adaptive":
        raise ValueError("audit requires frozen adaptive adapter")
    if config.get("ranking_policy") != "current_rules":
        raise ValueError("audit requires frozen current_rules ranking")
    if config.get("judgement_policy") != "current_rules":
        raise ValueError("audit requires frozen current_rules judgement")
    if int(config.get("budgets", {}).get("max_candidate_papers") or 0) != 200:
        raise ValueError("global candidate budget drift")
    local = config.get("local_bm25") or {}
    if local.get("corpus_sha256") != _sha256(corpus):
        raise ValueError("frozen local corpus config mismatch")
    if local.get("parameters") != manifest["bm25"]["parameters"]:
        raise ValueError("frozen BM25 parameter drift")
    if any(
        bool(config.get(field))
        for field in (
            "enable_query_evolution",
            "enable_refchain",
            "enable_semantic_seed_expansion",
        )
    ):
        raise ValueError("audit requires expansion modules disabled")
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
        raise ValueError("audit requires all LLM capabilities disabled")


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
