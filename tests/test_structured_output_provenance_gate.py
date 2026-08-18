from __future__ import annotations

import copy
import json
from pathlib import Path

from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import (
    QueryAnalysis,
    QueryPlanningResult,
    SearchSubquery,
)
from scholar_agent.evaluation.structured_output_provenance_gate import (
    validate_structured_result,
    write_structured_output_provenance_gate,
)


def _fixture() -> dict[str, object]:
    query = "query"
    analysis = QueryAnalysis(original_query=query)
    planning = QueryPlanningResult(
        selected_subqueries=[
            SearchSubquery(query=query, purpose="original_query")
        ],
        selected_subquery_count=1,
    )
    identifiers = {"doi": "10.1/one"}
    paper = {
        "title": "Paper one",
        "authors": ["A. Author"],
        "year": 2024,
        "venue": "Venue",
        "abstract": "Paper one reports evidence.",
        "identifiers": copy.deepcopy(identifiers),
        "urls": {
            "landing_page": "https://example.test/paper",
            "pdf": None,
        },
        "sources": ["arxiv"],
    }
    candidate = {
        "authority_digest": "2" * 64,
        "result_identity": "result:" + ("1" * 64),
        "rank": 1,
        "paper": paper,
        "relevance_score": 0.8,
        "category": "partially_relevant",
        "matched_constraints": [],
        "ranking_reason": "frozen",
        "evidence": [
            {"source": "title", "text": "Paper one", "confidence": 0.9}
        ],
    }
    evidence = {
        "row_id": "R1-E1",
        "citation_key": "R1",
        "rank": 1,
        "paper_title": "Paper one",
        "year": 2024,
        "venue": "Venue",
        "sources": ["arxiv"],
        "identifiers": copy.deepcopy(identifiers),
        "category": "partially_relevant",
        "final_score": 0.8,
        "evidence_source": "title",
        "evidence_text": "Paper one",
        "supported_terms": [],
        "supported_claim": "Paper one has title evidence relevant to the query.",
    }
    payload = {
        "run_id": "run",
        "status": "succeeded",
        "partial": False,
        "query_analysis": {
            "intent_type": "general",
            "domain": "general_science",
            "research_topics": [],
            "constraints": {
                "time_range": None,
                "venues": [],
                "methods": [],
                "datasets": [],
                "domains": [],
                "must_have_terms": [],
                "excluded_terms": [],
                "paper_types": [],
                "language": "unknown",
                "needs_expansion": False,
            },
        },
        "search_plan": {
            "expanded_queries": [query],
            "source_preferences": ["arxiv"],
            "max_rounds": 1,
            "query_planning_policy": "current_rules",
            "ranking_policy": "current_rules",
            "query_planning": planning.model_dump(mode="json"),
            "query_evolution_policy": "off",
            "enable_semantic_seed_expansion": False,
        },
        "highly_relevant_papers": [],
        "partially_relevant_papers": [candidate],
        "method_clusters": [
            {
                "name": "general",
                "paper_ranks": [1],
                "summary": (
                    "Ranks R1 are grouped as general results because no "
                    "method-specific keyword evidence was available."
                ),
            }
        ],
        "timeline": [
            {
                "year": 2024,
                "paper_ranks": [1],
                "summary": "Ranks R1 were published in 2024.",
            }
        ],
        "citation_graph": {
            "nodes": [
                {"id": "doi:10.1/one", "label": "Paper one", "rank": 1}
            ],
            "edges": [],
        },
        "warnings": [],
        "missing_evidence": [],
        "synthesis": {
            "answer_summary": (
                'For the query "query", the current general_science search '
                "evidence supports a general synthesis around the retrieved "
                "evidence. The strongest citation-backed candidates are [R1]. "
                "1 finding(s) were generated only from ranked-paper evidence rows."
            ),
            "status": "succeeded",
            "key_findings": [
                {
                    "text": (
                        "Paper one provides title evidence for evidence from "
                        "Venue [R1]."
                    ),
                    "citation_keys": ["R1"],
                    "confidence": 0.8,
                    "evidence_row_ids": ["R1-E1"],
                }
            ],
            "evidence_table": [evidence],
            "citation_coverage": {
                "ranked_paper_count": 1,
                "cited_paper_count": 1,
                "evidence_row_count": 1,
                "cited_evidence_row_count": 1,
                "missing_evidence_count": 0,
                "source_error_count": 0,
                "coverage_ratio": 1.0,
            },
            "limitations": [
                "refchain_not_enabled_or_not_available",
                "full_text_evidence_unavailable",
                "metadata_only_evidence:no_abstract_or_full_text_evidence_used",
            ],
            "warnings": [],
        },
        "retrieval_diagnostics": {},
        "budget_status": {},
        "cost_report": {},
        "judgement_policy": "current_rules",
        "judgement_config_hash": "fixture",
    }
    final = {
        "rank": 1,
        "title": "Paper one",
        "year": 2024,
        "identifiers": copy.deepcopy(identifiers),
        "sources": ["arxiv"],
        "category": "partially_relevant",
        "final_score": 0.8,
    }
    source = Paper.model_validate(paper)
    return {
        "payload": payload,
        "query": query,
        "query_analysis_payload": analysis.model_dump(mode="json"),
        "planning_payload": planning.model_dump(mode="json"),
        "expected_sources": ["arxiv"],
        "expected_top_k": 20,
        "final_ranked_candidates": [final],
        "source_candidates": [source],
    }


def _validate(fixture: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    return validate_structured_result(**fixture)  # type: ignore[arg-type]


def _codes(result: dict[str, object]) -> set[str]:
    return {str(item["code"]) for item in result["issues"]}  # type: ignore[index]


def test_valid_structured_output_is_fully_traceable() -> None:
    result, provenance = _validate(_fixture())
    assert result["terminal_status"] == "passed"
    assert result["issue_count"] == 0
    assert provenance
    assert {item["status"] for item in provenance} == {"verified"}


def test_fabricated_reference_and_wrong_identifier_fail() -> None:
    fabricated = _fixture()
    synthesis = fabricated["payload"]["synthesis"]  # type: ignore[index]
    synthesis["evidence_table"][0]["rank"] = 9  # type: ignore[index]
    result, provenance = _validate(fabricated)
    assert "fabricated_paper_reference" in _codes(result)
    assert any(item["status"] == "unverified" for item in provenance)

    conflict = _fixture()
    synthesis = conflict["payload"]["synthesis"]  # type: ignore[index]
    synthesis["evidence_table"][0]["identifiers"]["doi"] = "10.1/conflict"  # type: ignore[index]
    result, _ = _validate(conflict)
    assert "evidence_identity_conflict" in _codes(result)

    link_drift = _fixture()
    paper = link_drift["payload"]["partially_relevant_papers"][0]["paper"]  # type: ignore[index]
    paper["urls"]["landing_page"] = "https://example.test/fabricated"  # type: ignore[index]
    result, _ = _validate(link_drift)
    assert "candidate_source_field_drift" in _codes(result)


def test_duplicate_identity_and_invalid_group_are_detected() -> None:
    fixture = _fixture()
    payload = fixture["payload"]  # type: ignore[assignment]
    duplicate = copy.deepcopy(payload["partially_relevant_papers"][0])
    duplicate["rank"] = 2
    payload["partially_relevant_papers"].append(duplicate)
    payload["method_clusters"][0]["paper_ranks"].append(99)
    second_final = copy.deepcopy(fixture["final_ranked_candidates"][0])  # type: ignore[index]
    second_final["rank"] = 2
    fixture["final_ranked_candidates"].append(second_final)  # type: ignore[union-attr]
    result, _ = _validate(fixture)
    assert "duplicate_unified_identity" in _codes(result)
    assert "invalid_group_reference" in _codes(result)


def test_missing_required_field_and_missing_synthesis_are_explicit() -> None:
    invalid = _fixture()
    del invalid["payload"]["run_id"]  # type: ignore[index]
    result, _ = _validate(invalid)
    assert result["terminal_status"] == "schema_invalid"
    assert "schema_invalid" in _codes(result)

    old = _fixture()
    old["payload"]["synthesis"] = None  # type: ignore[index]
    result, _ = _validate(old)
    assert result["terminal_status"] == "blocked_missing_synthesis"
    assert "missing_structured_synthesis" in _codes(result)


def test_candidate_order_drift_is_detected() -> None:
    fixture = _fixture()
    payload = fixture["payload"]  # type: ignore[assignment]
    second = copy.deepcopy(payload["partially_relevant_papers"][0])
    second["rank"] = 2
    second["paper"]["title"] = "Paper two"
    second["paper"]["identifiers"]["doi"] = "10.1/two"
    payload["partially_relevant_papers"].insert(0, second)
    result, _ = _validate(fixture)
    assert "candidate_order_drift" in _codes(result)


def test_old_identifier_shape_without_new_aliases_is_compatible() -> None:
    fixture = _fixture()
    payload = fixture["payload"]  # type: ignore[assignment]
    payload["partially_relevant_papers"][0]["paper"]["identifiers"] = {
        "doi": "10.1/one"
    }
    payload["synthesis"]["evidence_table"][0]["identifiers"] = {
        "doi": "10.1/one"
    }
    result, _ = _validate(fixture)
    assert result["terminal_status"] == "passed"


def test_gate_artifacts_are_byte_deterministic(tmp_path: Path) -> None:
    cases = [{"dataset": "x", "case_order": 0, "case_id": "one"}]
    provenance = [
        {
            "dataset": "x",
            "case_order": 0,
            "case_id": "one",
            "kind": "paper",
        }
    ]
    aggregate = {"schema_version": "1", "gate_passed": True}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_structured_output_provenance_gate(first, cases, provenance, aggregate)
    write_structured_output_provenance_gate(second, cases, provenance, aggregate)
    for name in ("case_gate.jsonl", "provenance.jsonl", "aggregate.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_manifest_freezes_offline_scope_before_audit() -> None:
    manifest = json.loads(
        Path("benchmark/structured_output_provenance_gate_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert sum(item["case_count"] for item in manifest["frozen_inputs"]) == 65
    assert manifest["structured_chain"]["generator_kind"] == (
        "deterministic_rule_based"
    )
    assert manifest["structured_chain"]["llm_snapshot_required"] is False
    assert manifest["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
    }
