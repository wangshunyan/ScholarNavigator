from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint
from scholar_agent.evaluation.snapshots.store import SnapshotMissingError
from scholar_agent.evaluation.source_fusion_ablation import (
    IdentityRegistry,
    SourceFusionAblationError,
    SourceFusionNotEligible,
    build_candidate_pool,
    cluster_summary,
    compare_ranked_lists,
    finite_extrapolated_rbo,
    holm_bonferroni,
    rank_variant,
    reconstruct_source_inputs,
    _source_state,
    validate_full_reconstruction,
    validate_population_closure,
    verify_analysis,
    write_analysis,
)


SOURCES = ["openalex", "arxiv", "semantic_scholar", "pubmed"]


def _paper(
    title: str,
    doi: str,
    source: str,
    *,
    year: int = 2024,
    abstract: str | None = None,
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice Example"],
        year=year,
        abstract=abstract if abstract is not None else title,
        identifiers=PaperIdentifiers(doi=doi),
        sources=[source],
    )


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="causal evidence",
        language="en",
        intent="general",
        domain="general_science",
        constraints=QueryConstraint(must_include_terms=["causal", "evidence"]),
    )


def _protocol(*, main: int = 2, excluded: int = 1) -> dict[str, object]:
    return {
        "analysis_population": {
            "main_case_count": main,
            "excluded_case_count": excluded,
        },
        "bootstrap": {"seed": 23, "iterations": 200},
        "comparison_family": {
            "permutation_seed": 29,
            "permutation_iterations": 200,
        },
    }


def _stage_paper(value: object) -> dict[str, object]:
    paper = getattr(value, "paper", value)
    return paper.model_dump(mode="json")


def _frozen_stages(result: object) -> dict[str, dict[str, object]]:
    return {
        "initial_deduplicated": {
            "candidates": [_stage_paper(item) for item in result.candidates]
        },
        "initial_judged": {
            "candidates": [
                {
                    **_stage_paper(item),
                    "judgement_score": item.score,
                    "category": item.category,
                }
                for item in result.judgements
            ]
        },
        "initial_reranked": {
            "candidates": [
                {
                    **_stage_paper(item),
                    "rank": item.rank,
                    "category": item.category,
                    "final_score": item.final_score,
                }
                for item in result.ranked
            ]
        },
        "final_returned": {
            "candidates": [
                {**_stage_paper(item), "rank": item.rank}
                for item in result.returned
            ]
        },
    }


def test_exact_full_reconstruction_and_mismatch_detection() -> None:
    result = rank_variant(
        _analysis(),
        [
            _paper("Causal evidence", "10.1/a", "arxiv"),
            _paper("Other topic", "10.1/b", "pubmed"),
        ],
        top_k=20,
    )
    frozen = _frozen_stages(result)
    validate_full_reconstruction(result, frozen)
    frozen["initial_reranked"]["candidates"][0]["final_score"] = -1.0
    with pytest.raises(SourceFusionNotEligible, match="reconstruction_mismatch"):
        validate_full_reconstruction(result, frozen)


def test_source_inputs_preserve_call_order_and_missing_source_is_not_eligible() -> None:
    papers = {
        "a" * 64: SimpleNamespace(
            source="arxiv",
            adapted_query="q1",
            status="success",
            limit=20,
            adapter_policy="adaptive",
            papers=[_paper("First", "10.1/shared", "arxiv")],
        ),
        "b" * 64: SimpleNamespace(
            source="openalex",
            adapted_query="q1",
            status="success",
            limit=20,
            adapter_policy="adaptive",
            papers=[_paper("Longer shared title", "10.1/shared", "openalex")],
        ),
    }

    class Store:
        def read_retrieval(self, key: str) -> object:
            if key not in papers:
                raise SnapshotMissingError(key)
            return papers[key]

    initial = {
        "retrieval_calls": [
            {
                "source": "arxiv",
                "adapted_query": "q1",
                "logical_call_executed": True,
                "snapshot_key": "a" * 64,
                "terminal_status": "success",
            },
            {
                "source": "openalex",
                "adapted_query": "q1",
                "logical_call_executed": True,
                "snapshot_key": "b" * 64,
                "terminal_status": "success",
            },
            {
                "source": "pubmed",
                "logical_call_executed": False,
                "terminal_status": "not_started",
            },
        ]
    }
    inputs = reconstruct_source_inputs(
        initial,
        config={"sources": SOURCES, "top_k": 20, "query_adapter_policy": "adaptive"},
        store=Store(),
    )
    pool = build_candidate_pool(
        inputs.ordered_batches,
        included_sources=SOURCES,
        source_order=SOURCES,
        limit=200,
    )
    assert len(pool) == 1
    assert pool[0].title == "Longer shared title"
    assert inputs.terminal_counts["pubmed"]["not_started"] == 1

    initial["retrieval_calls"][0]["snapshot_key"] = "c" * 64
    with pytest.raises(SnapshotMissingError):
        reconstruct_source_inputs(
            initial,
            config={
                "sources": SOURCES,
                "top_k": 20,
                "query_adapter_policy": "adaptive",
            },
            store=Store(),
        )


def test_source_terminal_summary_preserves_partial_failure() -> None:
    from collections import Counter

    assert _source_state(Counter(success=2)) == "success"
    assert _source_state(Counter(success=1, failed=1)) == "partial_failure"
    assert _source_state(Counter(timeout=1)) == "failed"
    assert _source_state(Counter(not_started=3)) == "not_started"


def test_identity_overlap_exclusivity_and_conflicts_use_unified_rules() -> None:
    registry = IdentityRegistry()
    first = _paper("Shared", "10.1/shared", "arxiv")
    duplicate = _paper("Shared from another source", "https://doi.org/10.1/SHARED", "pubmed")
    conflicting = _paper("Conflict", "10.1/other", "openalex")
    first_label = registry.label(first)
    assert registry.label(duplicate) == first_label
    assert registry.label(conflicting) != first_label
    assert first_label.startswith("paper:")
    assert "shared" not in first_label


def test_ablation_reranks_without_changing_full_candidate_identity() -> None:
    shared = _paper("Causal evidence", "10.1/shared", "arxiv")
    duplicate = _paper("Causal evidence expanded", "10.1/shared", "openalex")
    unique = _paper("Causal evidence method", "10.1/unique", "pubmed")
    batches = [("arxiv", [shared]), ("openalex", [duplicate]), ("pubmed", [unique])]
    full_pool = build_candidate_pool(
        batches, included_sources=SOURCES, source_order=SOURCES, limit=200
    )
    loo_pool = build_candidate_pool(
        batches,
        included_sources=[item for item in SOURCES if item != "pubmed"],
        source_order=SOURCES,
        limit=200,
    )
    full = rank_variant(_analysis(), full_pool, top_k=20)
    loo = rank_variant(_analysis(), loo_pool, top_k=20)
    assert len(full.candidates) == 2
    assert len(loo.candidates) == 1
    registry = IdentityRegistry()
    comparison = compare_ranked_lists(
        registry.labels([item.paper for item in full.returned]),
        registry.labels([item.paper for item in loo.returned]),
        persistence=0.9,
        depth=20,
    )
    assert comparison["full_top20_identity_loss_count"] >= 0
    assert 0 <= comparison["rank_biased_overlap"] <= 1


def test_rbo_and_rank_metrics_have_fixed_boundaries() -> None:
    assert finite_extrapolated_rbo([], [], persistence=0.9, depth=20) == 1.0
    assert finite_extrapolated_rbo(["a", "b"], ["a", "b"], persistence=0.9, depth=20) == pytest.approx(1.0)
    changed = compare_ranked_lists(
        ["a", "b", "c"], ["b", "a", "d"], persistence=0.9, depth=20
    )
    assert changed["full_top20_identity_loss_count"] == 1
    assert changed["top20_jaccard"] == 0.5
    assert changed["shared_identity_mean_absolute_rank_displacement"] == 1.0


def test_cluster_bootstrap_and_holm_are_deterministic() -> None:
    cases = [
        {"component_identity": "one"},
        {"component_identity": "one"},
        {"component_identity": "two"},
    ]
    first = cluster_summary(cases, [1.0, 3.0, -1.0], _protocol(), "arxiv", "metric")
    second = cluster_summary(cases, [1.0, 3.0, -1.0], _protocol(), "arxiv", "metric")
    assert first == second
    assert first["component_count"] == 2
    corrected = holm_bonferroni(
        {"a": 0.01, "b": 0.03, "c": 0.2, "d": 0.8}, alpha=0.05
    )
    assert corrected["a"]["reject_at_family_alpha"] is True
    assert corrected["b"]["reject_at_family_alpha"] is False


def test_population_rejects_post_hoc_filtering_or_unregistered_exclusion() -> None:
    included = [
        {"case_order": 0, "analysis_status": "included_main_analysis"},
        {"case_order": 1, "analysis_status": "included_main_analysis"},
    ]
    excluded = [
        {"case_order": 2, "analysis_status": "excluded_no_successful_source"}
    ]
    validate_population_closure(included, excluded, _protocol())
    with pytest.raises(SourceFusionAblationError, match="population"):
        validate_population_closure(included[:1], excluded, _protocol())
    excluded[0]["analysis_status"] = "excluded_after_seeing_ablation"
    with pytest.raises(SourceFusionAblationError, match="unregistered"):
        validate_population_closure(included, excluded, _protocol())


def test_report_files_are_byte_deterministic_and_hash_checked(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol-source.json"
    protocol.write_text(
        json.dumps({"analysis": "source_fusion_ablation_v1", "schema_version": "1"}),
        encoding="utf-8",
    )
    cases = [
        {
            "case_order": 0,
            "query_identity": "query:opaque",
            "analysis_status": "included_main_analysis",
        }
    ]
    aggregate = {
        "status": "completed",
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "gold_or_qrels_loaded": False,
            "quality_metric_count": 0,
        },
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_analysis(first, cases, aggregate, protocol)
    write_analysis(second, cases, aggregate, protocol)
    for name in ("case_diagnostics.jsonl", "aggregate.json", "protocol.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert verify_analysis(first)["status"] == "completed"
    (first / "aggregate.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SourceFusionAblationError, match="(hash|size)_drift"):
        verify_analysis(first)


def test_protocol_is_preregistered_and_forbids_quality_metrics() -> None:
    protocol = json.loads(
        Path("benchmark/source_fusion_ablation_v1_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    encoded = json.dumps(protocol, sort_keys=True).casefold()
    assert protocol["analysis"] == "source_fusion_ablation_v1"
    assert protocol["execution"]["gold_access"] is False
    assert "precision" not in protocol["metrics"]
    assert "candidate_recall" not in encoded
    assert "recall_at" not in encoded
