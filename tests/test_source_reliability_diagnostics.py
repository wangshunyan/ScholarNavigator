from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.snapshots.store import SnapshotMissingError
from scholar_agent.evaluation.source_reliability_diagnostics import (
    SourceReliabilityError,
    _validate_population,
    audit_retrieval_requests,
    cluster_success_yield_summary,
    classify_failure,
    classify_primary_outcome,
    classify_query_structure,
    derive_source_funnel,
    inspect_unknown_snapshot_fields,
    source_query_state,
    verify_analysis,
    write_analysis,
)


SOURCES = ["openalex", "arxiv", "semantic_scholar", "pubmed"]
CONFIG = {"top_k": 20, "query_adapter_policy": "adaptive"}


def _paper(title: str, doi: str, source: str) -> Paper:
    return Paper(
        title=title,
        authors=["A. Example"],
        year=2024,
        abstract=title,
        identifiers=PaperIdentifiers(doi=doi),
        sources=[source],
    )


def _entry(
    *,
    source: str = "arxiv",
    status: str = "success",
    papers: list[Paper] | None = None,
    error: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        source=source,
        adapted_query="q",
        status=status,
        limit=20,
        adapter_policy="adaptive",
        papers=list(papers or []),
        error_message=error,
        warnings=[],
        diagnostics=ConnectorDiagnostics(request_count=1, error_count=status == "failed"),
        recorded_latency_seconds=0.25,
    )


def _call(key: str, *, returned: int = 0, terminal: str | None = None) -> dict[str, object]:
    return {
        "source": "arxiv",
        "adapted_query": "q",
        "logical_call_executed": True,
        "snapshot_key": key,
        "terminal_status": terminal,
        "returned_count": returned,
    }


class _Store:
    def __init__(self, entries: dict[str, object]) -> None:
        self.entries = entries
        self.read_count = 0

    def read_retrieval(self, key: str) -> object:
        self.read_count += 1
        if key not in self.entries:
            raise SnapshotMissingError(key)
        return self.entries[key]


def test_legal_empty_response_and_partial_failure_are_distinct() -> None:
    success_key = "a" * 64
    failed_key = "b" * 64
    store = _Store(
        {
            success_key: _entry(),
            failed_key: _entry(status="failed", error="HTTP 503"),
        }
    )
    audit = audit_retrieval_requests(
        {"retrieval_calls": [_call(success_key), _call(failed_key)]},
        config=CONFIG,
        store=store,
        sources=SOURCES,
    )
    record = audit.source_records["arxiv"]
    assert record["legal_empty_success_count"] == 1
    assert record["failure_category_counts"]["http_failure"] == 1
    assert source_query_state(record) == "partial_failure"


def test_failure_taxonomy_is_predeclared_and_deterministic() -> None:
    assert classify_failure("malformed JSON after HTTP 503", []) == "parse_failure"
    assert classify_failure("HTTP status 429", []) == "http_failure"
    assert classify_failure("TLS connection timeout", []) == "transport_failure"
    assert classify_failure("source_failure", []) == "provider_failure"
    assert classify_failure(None, []) == "unknown_failure"


def test_raw_nonempty_all_filtered_and_not_top20_have_distinct_outcomes() -> None:
    filtered = derive_source_funnel(
        parsed_record_count=2,
        canonical_record_count=2,
        source_identity_set={"a", "b"},
        global_identity_set={"a", "b"},
        budget_identity_set={"a", "b"},
        constraint_identity_set=set(),
        top20_identity_set=set(),
        raw_provider_record_count=2,
    )
    assert filtered["losses"]["parse_loss"]["count"] == 0
    assert filtered["losses"]["constraint_loss"]["count"] == 2
    record = {"legal_empty_success_count": 0}
    assert (
        classify_primary_outcome(filtered, record, query_state="success")
        == "constraint_loss"
    )

    not_top = derive_source_funnel(
        parsed_record_count=2,
        canonical_record_count=2,
        source_identity_set={"a", "b"},
        global_identity_set={"a", "b"},
        budget_identity_set={"a", "b"},
        constraint_identity_set={"a", "b"},
        top20_identity_set=set(),
        raw_provider_record_count=None,
    )
    assert not_top["losses"]["top20_selection_loss"]["count"] == 2
    assert (
        classify_primary_outcome(not_top, record, query_state="success")
        == "valid_not_top20"
    )


def test_identity_dedup_and_budget_losses_are_conserved() -> None:
    funnel = derive_source_funnel(
        parsed_record_count=4,
        canonical_record_count=4,
        source_identity_set={"a", "b", "c"},
        global_identity_set={"a", "b", "c"},
        budget_identity_set={"a", "b"},
        constraint_identity_set={"a"},
        top20_identity_set={"a"},
        raw_provider_record_count=None,
    )
    assert funnel["losses"]["identity_dedup_loss"]["count"] == 1
    assert funnel["losses"]["global_budget_loss"]["count"] == 1
    assert funnel["losses"]["constraint_loss"]["count"] == 1


def test_status_contradiction_is_a_hard_violation() -> None:
    key = "a" * 64
    with pytest.raises(SourceReliabilityError, match="signature_mismatch"):
        audit_retrieval_requests(
            {"retrieval_calls": [_call(key, terminal="failed")]},
            config=CONFIG,
            store=_Store({key: _entry(status="success")}),
            sources=SOURCES,
        )


def test_missing_snapshot_remains_explicit_unknown_evidence() -> None:
    key = "c" * 64
    audit = audit_retrieval_requests(
        {"retrieval_calls": [_call(key)]},
        config=CONFIG,
        store=_Store({}),
        sources=SOURCES,
    )
    record = audit.source_records["arxiv"]
    assert record["snapshot_missing_count"] == 1
    assert record["request_state_counts"]["snapshot_missing"] == 1
    assert record["unknown_evidence_count"] == 1


def test_duplicate_snapshot_reference_is_consumed_once() -> None:
    key = "d" * 64
    paper = _paper("One", "10.1/one", "arxiv")
    store = _Store({key: _entry(papers=[paper])})
    call = _call(key, returned=1)
    audit = audit_retrieval_requests(
        {"retrieval_calls": [call, dict(call)]},
        config=CONFIG,
        store=store,
        sources=SOURCES,
    )
    record = audit.source_records["arxiv"]
    assert store.read_count == 1
    assert record["logical_request_count"] == 2
    assert record["unique_snapshot_count"] == 1
    assert record["duplicate_snapshot_reference_count"] == 1
    assert record["parsed_record_count"] == 1


def test_unknown_snapshot_schema_field_is_detected(tmp_path: Path) -> None:
    key = "e" * 64
    retrieval = tmp_path / "retrieval"
    retrieval.mkdir()
    (retrieval / f"{key}.json").write_text(
        json.dumps({"schema_version": "1", "unexpected_payload": {"value": 1}}),
        encoding="utf-8",
    )
    store = SimpleNamespace(retrieval_dir=retrieval)
    assert inspect_unknown_snapshot_fields(store, key) == ["unexpected_payload"]


def test_query_structure_features_are_syntax_only_and_deterministic() -> None:
    value = classify_query_structure('"Alpha" AND β after 2024')
    assert value == {
        "length_bucket": "0_80",
        "has_quote": True,
        "has_boolean_operator": True,
        "has_year": True,
        "unicode_class": "non_ascii_letter",
    }
    assert classify_query_structure("a" * 321)["length_bucket"] == "321_plus"


def test_cluster_success_yield_interval_is_deterministic() -> None:
    cases = [
        {
            "component_identity": "component:a",
            "source_diagnostics": {
                "arxiv": {
                    "snapshot_success_count": 1,
                    "funnel": {"stages": {"parsed_record_count": 20}},
                }
            },
        },
        {
            "component_identity": "component:a",
            "source_diagnostics": {
                "arxiv": {
                    "snapshot_success_count": 2,
                    "funnel": {"stages": {"parsed_record_count": 20}},
                }
            },
        },
        {
            "component_identity": "component:b",
            "source_diagnostics": {
                "arxiv": {
                    "snapshot_success_count": 1,
                    "funnel": {"stages": {"parsed_record_count": 5}},
                }
            },
        },
    ]
    protocol = {"bootstrap": {"seed": 17, "iterations": 200}}
    first = cluster_success_yield_summary(cases, "arxiv", protocol)
    second = cluster_success_yield_summary(cases, "arxiv", protocol)
    assert first == second
    assert first["component_count"] == 2
    assert first["confidence_interval_95"][0] <= first["mean"]
    assert first["mean"] <= first["confidence_interval_95"][1]


def test_population_rejects_posthoc_filtering() -> None:
    protocol = {
        "analysis_population": {"main_case_count": 2, "excluded_case_count": 1}
    }
    included = [
        {"case_order": 0, "analysis_status": "included_main_analysis"},
        {"case_order": 1, "analysis_status": "included_main_analysis"},
    ]
    excluded = [
        {"case_order": 2, "analysis_status": "excluded_no_successful_source"}
    ]
    _validate_population(included, excluded, protocol)
    with pytest.raises(SourceReliabilityError, match="population"):
        _validate_population(included[:1], excluded, protocol)
    excluded[0]["analysis_status"] = "excluded_after_observing_failure"
    with pytest.raises(SourceReliabilityError, match="unregistered"):
        _validate_population(included, excluded, protocol)


def test_report_is_byte_deterministic_and_hash_checked(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"analysis": "source_reliability_diagnostics_v1"}),
        encoding="utf-8",
    )
    cases = [{"case_order": 0, "query_identity": "query:opaque"}]
    aggregate = {
        "status": "completed",
        "execution": {
            "gold_or_qrels_loaded": False,
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
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
    with pytest.raises(SourceReliabilityError, match="(hash|size)_drift"):
        verify_analysis(first)


def test_protocol_is_frozen_without_quality_metrics() -> None:
    protocol = json.loads(
        Path("benchmark/source_reliability_diagnostics_v1_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    encoded = json.dumps(protocol, sort_keys=True).casefold()
    assert protocol["analysis"] == "source_reliability_diagnostics_v1"
    assert protocol["execution"]["gold_access"] is False
    assert "candidate_recall" not in encoded
    assert "recall_at" not in encoded
    assert "quality_score" in protocol["analysis_population"]["selection_prohibitions"]
