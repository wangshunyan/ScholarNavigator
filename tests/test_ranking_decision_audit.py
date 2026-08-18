from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from scholar_agent.agents.judgement import (
    PRODUCTION_JUDGEMENT_COMPONENT_ORDER,
    production_category_from_score,
    production_judgement_decision_catalog,
    trace_judgement_decision,
)
from scholar_agent.agents.reranker import (
    production_ranking_decision_catalog,
    trace_ranking_decision,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint
from scholar_agent.evaluation.ranking_decision_audit import (
    RankingDecisionAuditError,
    aggregate_analysis,
    input_permutation_diagnostic,
    load_protocol,
    selection_reason,
    threshold_margin,
    validate_decision_records,
    verify_analysis,
    write_analysis,
)
from scholar_agent.evaluation.source_fusion_ablation import IdentityRegistry, rank_variant


PROTOCOL_PATH = Path("benchmark/ranking_decision_audit_v1_protocol.json")


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="neural retrieval benchmark method",
        language="en",
        intent="general",
        domain="machine_learning",
        constraints=QueryConstraint(),
    )


def _paper(index: int, *, title: str | None = None) -> Paper:
    return Paper(
        title=title or f"Neural retrieval benchmark method {index}",
        abstract="A neural retrieval benchmark method study.",
        year=2024,
        citation_count=index,
        identifiers=PaperIdentifiers(doi=f"10.1234/fixture.{index}"),
        sources=["arxiv"],
    )


def _records_for(papers: list[Paper]):
    analysis = _analysis()
    full = rank_variant(analysis, papers, top_k=20)
    registry = IdentityRegistry()
    identities = registry.labels(full.candidates)
    returned = set(registry.labels([item.paper for item in full.returned]))
    ranked_by_id = {registry.label(item.paper): item for item in full.ranked}
    traces = [
        trace_ranking_decision(analysis, judgement, index)
        for index, judgement in enumerate(full.judgements)
    ]
    title_values = sorted({str(item["sort_key"][5]) for item in traces})
    title_ordinals = {value: index for index, value in enumerate(title_values)}
    records = []
    for index, (identity, judgement, trace) in enumerate(
        zip(identities, full.judgements, traces, strict=True)
    ):
        ranked = ranked_by_id[identity]
        observed = trace_judgement_decision(judgement)
        raw_key = list(trace["sort_key"])
        safe_key = [
            int(raw_key[0]),
            float(raw_key[1]),
            float(raw_key[2]),
            int(raw_key[3]),
            int(raw_key[4]),
            title_ordinals[str(raw_key[5])],
            int(raw_key[6]),
        ]
        in_window = ranked.rank <= 20
        retained = judgement.category in {"highly_relevant", "partially_relevant"}
        records.append(
            {
                "candidate_identity": identity,
                "judgement": {
                    "components": observed["components"],
                    "total_score": judgement.score,
                    "category": judgement.category,
                    "category_reason": observed["category_reason"],
                },
                "ranking": {
                    "score_breakdown": trace["score_breakdown"],
                    "reranked_position": ranked.rank,
                    "sort_key": safe_key,
                    "tie_key": safe_key[:-1],
                },
                "top20": {
                    "within_rank_window": in_window,
                    "category_gate_passed": retained,
                    "final_returned": identity in returned,
                },
            }
        )
    return analysis, full, registry, records


def test_protocol_matches_automatic_production_catalogs() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol["decision_catalog"]["judgement_components"] == list(
        PRODUCTION_JUDGEMENT_COMPONENT_ORDER
    )
    assert protocol["decision_catalog"]["rerank_key"] == (
        production_ranking_decision_catalog()["sort_key"]
    )
    assert production_judgement_decision_catalog()["threshold_comparison"] == (
        "greater_than_or_equal"
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.72, "highly_relevant"),
        (math.nextafter(0.72, 0.0), "partially_relevant"),
        (0.45, "partially_relevant"),
        (math.nextafter(0.45, 0.0), "weakly_relevant"),
        (0.25, "weakly_relevant"),
        (math.nextafter(0.25, 0.0), "irrelevant"),
    ],
)
def test_threshold_equality_and_adjacent_float_semantics(
    score: float, expected: str
) -> None:
    assert production_category_from_score(score) == expected
    assert json.loads(json.dumps({"score": score}, allow_nan=False))["score"] == score


def test_threshold_margin_and_selection_reason_boundaries() -> None:
    assert threshold_margin(0.72, "highly_relevant")["lower_margin"] == 0.0
    assert threshold_margin(0.45, "partially_relevant")["lower_margin"] == 0.0
    assert selection_reason(True, True) == "returned"
    assert selection_reason(True, False) == "category_gate"
    assert selection_reason(False, True) == "beyond_top20_cutline"


def test_records_reconstruct_score_category_rerank_and_top20() -> None:
    _analysis_value, full, registry, records = _records_for(
        [_paper(index) for index in range(25)]
    )
    validate_decision_records(records, full, registry, load_protocol(PROTOCOL_PATH))
    assert sum(item["top20"]["within_rank_window"] for item in records) == 20
    assert any(
        int(item["ranking"]["reranked_position"]) != index + 1
        for index, item in enumerate(records)
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("component_sum", "component_sum"),
        ("unknown_component", "unregistered_judgement_component"),
        ("category", "category_record"),
        ("missing_key", "ranking_key_missing"),
        ("rerank_position", "reranked_position"),
        ("top20", "top20_final_state"),
    ],
)
def test_validator_rejects_score_category_sort_and_cutline_drift(
    mutation: str, reason: str
) -> None:
    _analysis_value, full, registry, records = _records_for([_paper(1), _paper(2)])
    broken = copy.deepcopy(records)
    if mutation == "component_sum":
        broken[0]["judgement"]["components"]["topic_match"] += 0.1
    elif mutation == "unknown_component":
        broken[0]["judgement"]["components"]["unregistered"] = 0.0
    elif mutation == "category":
        broken[0]["judgement"]["category"] = "irrelevant"
    elif mutation == "missing_key":
        broken[0]["ranking"]["sort_key"].pop()
    elif mutation == "rerank_position":
        broken[0]["ranking"]["reranked_position"] = 99
    else:
        broken[0]["top20"]["final_returned"] = not broken[0]["top20"][
            "final_returned"
        ]
    with pytest.raises(RankingDecisionAuditError, match=reason):
        validate_decision_records(
            broken, full, registry, load_protocol(PROTOCOL_PATH)
        )


def test_validator_rejects_nan_before_json_serialization() -> None:
    _analysis_value, full, registry, records = _records_for([_paper(1)])
    records[0]["ranking"]["score_breakdown"]["final_score"] = float("nan")
    with pytest.raises(RankingDecisionAuditError, match="non_finite"):
        validate_decision_records(records, full, registry, load_protocol(PROTOCOL_PATH))


def test_registered_original_index_tie_break_explains_input_permutation() -> None:
    papers = [
        _paper(1, title="Identical title"),
        _paper(2, title="Identical title"),
    ]
    papers[0].citation_count = papers[1].citation_count = 0
    analysis, full, registry, records = _records_for(papers)
    result = input_permutation_diagnostic(analysis, full, registry, records)
    assert result["changed_candidate_count"] == 2
    assert result["input_order_sensitive_tie_group_count"] == 1
    assert result["changed_only_within_registered_tie_groups"] is True


def test_distinct_title_tie_break_is_input_order_independent() -> None:
    papers = [_paper(1, title="A title"), _paper(2, title="B title")]
    papers[0].citation_count = papers[1].citation_count = 0
    analysis, full, registry, records = _records_for(papers)
    result = input_permutation_diagnostic(analysis, full, registry, records)
    assert result["changed_candidate_count"] == 0


def test_aggregate_and_written_outputs_are_byte_deterministic(tmp_path: Path) -> None:
    _analysis_value, full, _registry, records = _records_for([_paper(1), _paper(2)])
    decisions = []
    for order, record in enumerate(records):
        item = copy.deepcopy(record)
        item.update(
            {
                "case_order": 0,
                "candidate_order": order,
                "query_identity": "opaque-query",
                "component_identity": "opaque-component",
                "source_provenance": ["arxiv"],
                "field_lineage": {"fields": {}},
            }
        )
        item["ranking"].update(
            {
                "pre_rerank_position": order + 1,
                "position_delta": order + 1
                - int(item["ranking"]["reranked_position"]),
            }
        )
        item["judgement"].update(
            {
                "threshold_margin": threshold_margin(
                    float(item["judgement"]["total_score"]),
                    str(item["judgement"]["category"]),
                ),
                "failed_constraint_dimensions": [],
            }
        )
        item["top20"].update(
            {
                "rank_margin": 20 - int(item["ranking"]["reranked_position"]),
                "reason": selection_reason(
                    bool(item["top20"]["within_rank_window"]),
                    bool(item["top20"]["category_gate_passed"]),
                ),
            }
        )
        decisions.append(item)
    cases = [
        {
            "case_order": 0,
            "component_identity": "opaque-component",
            "reconstruction": {"exact": True},
            "input_permutation": {"input_order_sensitive_tie_group_count": 0},
        }
    ]
    protocol = load_protocol(PROTOCOL_PATH)
    first = aggregate_analysis(
        cases,
        [],
        decisions,
        protocol,
        protocol_sha256="protocol",
        input_hashes={},
        observed_snapshot_key_count=1,
    )
    second = aggregate_analysis(
        cases,
        [],
        decisions,
        protocol,
        protocol_sha256="protocol",
        input_hashes={},
        observed_snapshot_key_count=1,
    )
    assert first == second
    assert first["closure"]["candidate_decision_count"] == len(full.candidates)
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    write_analysis(output_a, cases, decisions, first, PROTOCOL_PATH)
    write_analysis(output_b, cases, decisions, second, PROTOCOL_PATH)
    for name in (
        "aggregate.json",
        "candidate_decisions.jsonl",
        "case_diagnostics.jsonl",
        "manifest.json",
        "protocol.json",
    ):
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()
    assert verify_analysis(output_a)["exit_code"] == 0
