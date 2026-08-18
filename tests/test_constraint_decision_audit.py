from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scholar_agent.agents.judgement import (
    PRODUCTION_CONSTRAINT_ORDER,
    judge_papers,
    production_constraint_catalog,
    trace_constraint_decisions,
)
from scholar_agent.agents.judgement_config import CURRENT_RULES_CONFIG
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import QueryAnalysis, QueryConstraint, TimeRange
from scholar_agent.evaluation.constraint_decision_audit import (
    ConstraintDecisionAuditError,
    aggregate_analysis,
    assert_variant_equivalent,
    cluster_summary,
    remove_constraint_field,
    reorder_query_constraints,
    validate_trace,
    verify_analysis,
    write_analysis,
)
from scholar_agent.evaluation.source_fusion_ablation import IdentityRegistry, rank_variant


def _analysis(constraints: QueryConstraint | None = None) -> QueryAnalysis:
    return QueryAnalysis(
        original_query="retrieval neural transformer benchmark dataset framework",
        language="en",
        intent="general",
        domain="machine_learning",
        constraints=constraints or QueryConstraint(),
    )


def _by_name(values: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(item["constraint"]): item for item in values}


def test_catalog_enumerates_all_production_constraint_fields() -> None:
    catalog = production_constraint_catalog()
    assert catalog["field_order"] == list(PRODUCTION_CONSTRAINT_ORDER)
    assert catalog["fields"]["domains"]["enforcement"] == (
        "planning_only_not_consumed_by_candidate_predicate"
    )


def test_trace_captures_single_and_multiple_failures_without_short_circuit() -> None:
    analysis = _analysis(
        QueryConstraint(
            exclude_terms=["forbidden"],
            must_include_terms=["required", "missing"],
            methods=["method-x"],
            explicit_fields=["must_include_terms"],
        )
    )
    paper = Paper(
        title="Forbidden required retrieval paper",
        abstract="A transformer study without the requested method.",
    )
    trace = _by_name(trace_constraint_decisions(analysis, paper))
    assert trace["exclude_terms"]["status"] == "failed"
    assert trace["must_include_terms"]["status"] == "failed"
    assert trace["methods"]["status"] == "failed"
    assert trace["datasets"]["status"] == "not_applicable"
    assert trace["exclude_terms"]["reason_code"] == "excluded_term_matched"


def test_trace_distinguishes_missing_empty_and_production_predicate() -> None:
    analysis = _analysis(
        QueryConstraint(
            exclude_terms=["unsafe"],
            must_include_terms=["required"],
            methods=["method"],
            datasets=["dataset"],
            paper_types=["survey"],
            venues=["venue"],
            time_range=TimeRange(start_year=2020, end_year=2024),
            explicit_fields=["must_include_terms", "datasets", "time_range"],
        )
    )
    trace = _by_name(
        trace_constraint_decisions(
            analysis,
            Paper(title="", abstract="", venue=None, year=None),
        )
    )
    for name in (
        "exclude_terms",
        "must_include_terms",
        "methods",
        "datasets",
        "paper_types",
        "venues",
        "time_range",
    ):
        assert trace[name]["status"] == "unknown"
    assert trace["exclude_terms"]["production_predicate_result"] is True
    assert trace["must_include_terms"]["production_predicate_result"] is False
    assert trace["venues"]["reason_code"] == "venue_null"
    assert trace["time_range"]["reason_code"] == "year_null"


def test_venue_and_paper_type_use_production_alias_semantics() -> None:
    analysis = _analysis(
        QueryConstraint(venues=["Neur IPS"], paper_types=["method"])
    )
    trace = _by_name(
        trace_constraint_decisions(
            analysis,
            Paper(
                title="A framework for retrieval",
                abstract="This approach introduces a method.",
                venue="Neur-IPS",
            ),
        )
    )
    assert trace["venues"]["status"] == "passed"
    assert trace["paper_types"]["status"] == "passed"


def test_domains_are_explicitly_not_a_candidate_predicate() -> None:
    trace = _by_name(
        trace_constraint_decisions(
            _analysis(QueryConstraint(domains=["machine_learning"])),
            Paper(title="retrieval", abstract="neural"),
        )
    )
    assert trace["domains"]["status"] == "not_applicable"
    assert trace["domains"]["reason_code"] == "not_consumed_by_candidate_predicate"


def test_constraint_value_order_does_not_change_trace_or_production_output() -> None:
    analysis = _analysis(
        QueryConstraint(
            must_include_terms=["retrieval", "transformer"],
            methods=["framework", "method"],
            datasets=["dataset", "benchmark"],
            paper_types=["method", "benchmark"],
            venues=["NeurIPS", "ICLR"],
            explicit_fields=["datasets", "must_include_terms"],
        )
    )
    papers = [
        Paper(
            title="Retrieval transformer benchmark framework",
            abstract="A method evaluated on a dataset.",
            venue="NeurIPS",
        ),
        Paper(title="Unrelated note", abstract="minimal evidence"),
    ]
    reordered = reorder_query_constraints(analysis)
    assert [trace_constraint_decisions(analysis, paper) for paper in papers] == [
        trace_constraint_decisions(reordered, paper) for paper in papers
    ]
    registry = IdentityRegistry()
    assert_variant_equivalent(
        rank_variant(analysis, papers, top_k=20),
        rank_variant(reordered, papers, top_k=20),
        registry,
    )


def test_trace_validator_rejects_reason_omission_and_wrong_field_alias() -> None:
    analysis = _analysis(QueryConstraint(venues=["ICLR"]))
    paper = Paper(title="retrieval", abstract="study", venue="ICLR")
    judgement = judge_papers(
        analysis,
        [paper],
        use_llm=False,
        config=CURRENT_RULES_CONFIG,
    )[0]
    trace = trace_constraint_decisions(analysis, paper)
    missing_reason = copy.deepcopy(trace)
    missing_reason[-2]["reason_code"] = ""
    with pytest.raises(ConstraintDecisionAuditError, match="reason_missing"):
        validate_trace(missing_reason, judgement.feature_vector)
    wrong_field = copy.deepcopy(trace)
    wrong_field[-2]["field_lineage"] = [
        {"field": "publication_venue", "state": "present"}
    ]
    with pytest.raises(ConstraintDecisionAuditError, match="field_reference"):
        validate_trace(wrong_field, judgement.feature_vector)


def test_single_constraint_shadow_uses_production_path_and_can_restore_candidate() -> None:
    analysis = _analysis(QueryConstraint(exclude_terms=["forbidden"]))
    paper = Paper(
        title="Forbidden retrieval neural transformer benchmark dataset framework",
        abstract="retrieval neural transformer benchmark dataset framework",
    )
    baseline = rank_variant(analysis, [paper], top_k=20)
    shadow = rank_variant(
        remove_constraint_field(analysis, "exclude_terms"), [paper], top_k=20
    )
    assert baseline.judgements[0].category == "irrelevant"
    assert shadow.judgements[0].category in {
        "highly_relevant",
        "partially_relevant",
    }


def _fake_case(query: str, component: str) -> dict[str, object]:
    shadow = {
        name: {
            "active_in_query": name in {"exclude_terms", "must_include_terms"},
            "restored_survivor_identity_count": int(name == "exclude_terms"),
            "lost_survivor_identity_count": 0,
            "top20_fill_delta": int(name == "exclude_terms"),
            "top20_identity_added_count": int(name == "exclude_terms"),
            "top20_identity_removed_count": 0,
        }
        for name in PRODUCTION_CONSTRAINT_ORDER
    }
    return {
        "query_identity": query,
        "component_identity": component,
        "query_structure": {
            "length_bucket": "0_80",
            "has_quote": False,
            "has_boolean_operator": False,
            "has_year": False,
            "unicode_class": "ascii_only",
        },
        "shadow_leave_one_constraint_out": shadow,
        "constraint_order_invariant": True,
        "reconstruction": {"exact": True},
    }


def _fake_decision(
    query: str, failed: list[str], *, retained: bool
) -> dict[str, object]:
    constraints = []
    for name in PRODUCTION_CONSTRAINT_ORDER:
        constraints.append(
            {
                "constraint": name,
                "status": "failed" if name in failed else "not_applicable",
                "explicit": False,
                "production_predicate_result": False if name in failed else None,
                "expected_value_count": int(name in failed),
                "field_lineage": [],
            }
        )
    return {
        "query_identity": query,
        "source_provenance": ["arxiv"],
        "constraints": constraints,
        "failed_constraints": failed,
        "unknown_constraints": [],
        "production": {
            "judgement_category": "irrelevant" if not retained else "partially_relevant",
            "category_reason": "fixture",
            "judgement_score": 0.0 if not retained else 0.5,
            "retained_by_result_policy": retained,
            "selected_top20": retained,
        },
    }


def _protocol() -> dict[str, object]:
    return {
        "implementation_base_commit": "fixture",
        "statistical_method": {"seed": 7, "iterations": 100},
        "warnings": ["diagnostic only"],
    }


def test_aggregate_closes_unique_and_joint_failures_deterministically() -> None:
    cases = [_fake_case("q1", "c1"), _fake_case("q2", "c2")]
    decisions = [
        _fake_decision("q1", ["exclude_terms"], retained=False),
        _fake_decision("q2", ["must_include_terms", "methods"], retained=False),
    ]
    first = aggregate_analysis(
        cases,
        [],
        decisions,
        _protocol(),
        protocol_sha256="protocol",
        input_hashes={},
        observed_snapshot_key_count=2,
    )
    second = aggregate_analysis(
        cases,
        [],
        decisions,
        _protocol(),
        protocol_sha256="protocol",
        input_hashes={},
        observed_snapshot_key_count=2,
    )
    assert first == second
    assert first["constraints"]["exclude_terms"]["unique_failure_candidate_count"] == 1
    assert first["failure_combination_counts"] == {
        "exclude_terms": 1,
        "must_include_terms+methods": 1,
    }


def test_cluster_summary_is_byte_stable_with_uneven_components() -> None:
    cases = [
        {"component_identity": "a"},
        {"component_identity": "a"},
        {"component_identity": "b"},
    ]
    first = cluster_summary(cases, [1.0, 3.0, 8.0], _protocol(), stream="x")
    second = cluster_summary(cases, [1.0, 3.0, 8.0], _protocol(), stream="x")
    assert first == second
    assert first["component_count"] == 2


def test_write_and_verify_are_byte_deterministic(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol-source.json"
    protocol.write_text(
        json.dumps({"analysis": "constraint_decision_audit_v1"}), encoding="utf-8"
    )
    cases = [{"case_order": 0, "query_identity": "opaque"}]
    decisions = [
        {
            "case_order": 0,
            "candidate_order": 0,
            "query_identity": "opaque",
            "candidate_identity": "paper",
        }
    ]
    aggregate = {
        "analysis": "constraint_decision_audit_v1",
        "status": "completed",
        "execution": {"network_request_count": 0},
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_analysis(first, cases, decisions, aggregate, protocol)
    write_analysis(second, cases, decisions, aggregate, protocol)
    for name in (
        "aggregate.json",
        "candidate_decisions.jsonl",
        "case_diagnostics.jsonl",
        "manifest.json",
        "protocol.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert verify_analysis(first)["exit_code"] == 0
