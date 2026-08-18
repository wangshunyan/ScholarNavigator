import sys
from types import SimpleNamespace
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_cross_source_gold import (  # noqa: E402
    _matches,
    _current_signal,
    build_request_plan,
    _request,
    _source_outage_response,
    _request_aborted_response,
    _should_record_missing,
    _exception_response,
    _run_isolated,
    _parent_throttle,
    _is_failed_request_status,
    _validate_probe_evidence,
    _validate_snapshot_payload,
    classify_source,
    potential_coverage_upper_bound,
)


def _blocking_target() -> str:
    time.sleep(5)
    return "late"


def _quick_target() -> str:
    return "completed"


def _large_payload_target() -> str:
    return "x" * (4 * 1024 * 1024)


def test_classification_is_mutually_exclusive_and_prioritizes_current_candidate() -> None:
    assert classify_source(
        oracle={"title_applicable": True, "hit": True, "failed_count": 0},
        current={"candidate_identity": True, "drop_reason": "outside_final_top_k"},
    ) == "entered_candidate_filtered_or_truncated"
    assert classify_source(
        oracle={"title_applicable": True, "hit": True, "failed_count": 0},
        current={"candidate_identity": False, "returned": False, "title_present_unmatched": True},
    ) == "identity_normalization_or_dedup_miss"
    assert classify_source(
        oracle={"title_applicable": True, "hit": False, "failed_count": 0, "complete": True, "fixed_depth_complete": True},
        current={"candidate_identity": False, "returned": False},
    ) == "fixed_depth_title_miss"
    assert classify_source(
        oracle={"title_applicable": True, "hit": True, "fixed_depth_hit": True, "failed_count": 0, "complete": True},
        current={"candidate_identity": False, "returned": False},
    ) == "indexed_but_normalized_query_miss"
    assert classify_source(
        oracle={"title_applicable": True, "hit": False, "failed_count": 2, "complete": False, "fixed_depth_failed": True},
        current={"candidate_identity": False, "returned": False},
    ) == "external_failure"


def test_source_aware_identifier_requests_skip_unsupported_fields() -> None:
    gold = {"arxiv_id": "https://arxiv.org/abs/2302.13971v2", "title": "A Title"}
    assert _request("arxiv", "identifier", gold, 100)["query"] == gold["arxiv_id"]
    assert _request("openalex", "identifier", gold, 100)["query"] == gold["arxiv_id"]
    assert _request("pubmed", "identifier", gold, 100)["available"] is False
    assert _request("pubmed", "identifier", gold, 100)["query"] is None
    assert _request("arxiv", "exact_title", gold, 100)["capability"] == "search_only"
    assert _request("arxiv", "fixed_depth_title", gold, 100)["limit"] == 100


def test_identifier_matching_normalizes_doi_arxiv_versions_pmid_and_source_ids() -> None:
    assert _matches(
        {"identifiers": {"doi": "https://doi.org/10.48550/arXiv.2302.13971"}, "title": "Variant"},
        {"arxiv_id": "arxiv:2302.13971v2", "title": "Other"},
    )[0]
    assert _matches(
        {"identifiers": {"pubmed_id": "https://pubmed.ncbi.nlm.nih.gov/12345/"}, "title": "Variant"},
        {"pubmed_id": "PMID:12345", "title": "Other"},
    )[0]
    assert _matches(
        {"identifiers": {"openalex_id": "https://openalex.org/W123"}, "title": "Variant"},
        {"openalex_id": "W123", "title": "Other"},
    )[0]
    assert _matches(
        {"identifiers": {}, "title": "A  Title: a study"},
        {"title": "A Title: a study"},
    )[1]


def test_depth_and_partial_failure_are_conservative() -> None:
    assert classify_source(
        oracle={"title_applicable": True, "failed_count": 0, "complete": True, "hit": True, "fixed_depth_rank": 21, "depth": 20, "fixed_depth_complete": True},
        current={"candidate_identity": True, "initial_rank": 21, "drop_reason": "outside_final_top_k"},
    ) == "entered_candidate_filtered_or_truncated"
    assert classify_source(
        oracle={"title_applicable": True, "failed_count": 1, "complete": False, "hit": False, "fixed_depth_failed": True},
        current={"candidate_identity": False, "returned": False},
    ) == "external_failure"
    assert classify_source(
        oracle={"title_applicable": True, "failed_count": 0, "complete": True, "hit": False, "fixed_depth_complete": True},
        current={"candidate_identity": False, "returned": False},
    ) == "fixed_depth_title_miss"


def test_fixed_depth_hit_survives_redundant_identifier_failure_and_unreturned_candidate_is_filtered() -> None:
    oracle = {"title_applicable": True, "fixed_depth_hit": True, "fixed_depth_failed": False}
    assert classify_source(oracle=oracle, current={"candidate_identity": False, "returned": False}) == "indexed_but_normalized_query_miss"
    assert classify_source(oracle={**oracle, "identifier_failed": True}, current={"candidate_identity": False, "returned": False}) == "indexed_but_normalized_query_miss"
    assert classify_source(
        oracle={"title_applicable": True, "identifier_hit": True, "fixed_depth_complete": True, "fixed_depth_hit": False},
        current={"candidate_identity": False, "returned": False},
    ) == "indexed_but_normalized_query_miss"
    assert classify_source(oracle=oracle, current={"candidate_identity": True, "returned": False}) == "entered_candidate_filtered_or_truncated"
    assert classify_source(
        oracle={"title_applicable": True, "fixed_depth_failed": True, "fixed_depth_complete": False},
        current={"candidate_identity": False, "returned": False},
    ) == "external_failure"
    assert classify_source(
        oracle={"title_applicable": True, "fixed_depth_outage": True},
        current={"candidate_identity": True, "returned": True},
    ) == "external_failure"
    assert classify_source(
        oracle={"title_applicable": True, "exact_title_hit": True, "fixed_depth_failed": True},
        current={"candidate_identity": False, "returned": False},
    ) == "indexed_but_normalized_query_miss"


def test_potential_coverage_upper_bound_is_per_source_and_excludes_uncertain() -> None:
    records = [{"sources": {
        "arxiv": {"oracle": {"fixed_depth_hit": True}, "current": {"candidate_identity": False}},
        "openalex": {"oracle": {"fixed_depth_hit": False}, "current": {"candidate_identity": False}},
        "semantic_scholar": {"oracle": {"fixed_depth_hit": True}, "current": {"candidate_identity": True}},
        "pubmed": {"oracle": {"fixed_depth_hit": False}, "current": {"candidate_identity": False}},
    }}]
    result = potential_coverage_upper_bound(records)
    assert result["arxiv"]["gold_denominator"] == 1
    assert result["arxiv"]["potential_new_gold_count"] == 1
    assert result["semantic_scholar"]["potential_new_gold_count"] == 0
    outage = [{"sources": {"arxiv": {"oracle": {"fixed_depth_outage": True}, "current": {"candidate_identity": False}}}}]
    outage_result = potential_coverage_upper_bound(outage)
    assert outage_result["arxiv"]["outage_gold_count"] == 1
    assert outage_result["arxiv"]["evaluable_gold_denominator"] == 0
    assert outage_result["arxiv"]["potential_new_gold_count"] is None


def test_snapshot_request_and_key_mismatch_is_rejected() -> None:
    expected = _request("arxiv", "fixed_depth_title", {"arxiv_id": "2302.13971", "title": "A Title"}, 100)
    payload = {"request": dict(expected), "response": {"status": "success"}}
    _validate_snapshot_payload(payload, expected)
    payload["request"]["limit"] = 20
    try:
        _validate_snapshot_payload(payload, expected)
    except ValueError as exc:
        assert "snapshot_request_mismatch" in str(exc)
    else:
        raise AssertionError("mismatched snapshot accepted")


def test_source_outage_evidence_is_structured_and_zero_request() -> None:
    evidence = {"source": "arxiv", "timestamp": "2026-07-20T00:00:00+00:00", "status": "failed", "error_message": "HTTP 429", "error_type": "HTTP 429", "diagnostics": {"request_count": 2}}
    assert _validate_probe_evidence(evidence, "arxiv")["source"] == "arxiv"
    response = _source_outage_response("arxiv", evidence)
    assert response["requested"] is False
    assert response["diagnostics"]["request_count"] == 0
    assert _validate_probe_evidence(response["probe_evidence"], "arxiv") == evidence
    aborted = _request_aborted_response("arxiv", {"source": "arxiv", "stalled_key": "abc", "error_type": "timeout"}, requested=False)
    assert aborted["status"] == "request_aborted"
    assert aborted["requested"] is False
    attempted = _request_aborted_response("arxiv", {"source": "arxiv", "stalled_key": "abc", "error_type": "timeout"}, requested=True)
    assert attempted["requested"] is True
    assert attempted["diagnostics"]["request_count"] == 1
    assert attempted["diagnostics"]["error_count"] == 1
    assert attempted["warnings"] == ["attempted_then_terminated_after_stall"]
    assert attempted["latency_seconds"] == 0.0
    assert _is_failed_request_status("request_aborted") is True


def test_record_missing_retries_per_request_failures_but_preserves_source_outage() -> None:
    assert _should_record_missing({"status": "failed"}) is False
    assert _should_record_missing({"status": "request_aborted"}) is True
    assert _should_record_missing({"status": "source_outage"}) is False
    assert _should_record_missing({"status": "success"}) is False
    assert _should_record_missing(None) is True


def test_unexpected_worker_exception_is_a_terminal_auditable_failure() -> None:
    response = _exception_response(TimeoutError("connector stalled"))
    assert response["status"] == "failed"
    assert response["failure_layer"] == "worker_exception"
    assert response["diagnostics"]["error_count"] == 1


def test_wall_clock_timeout_terminates_one_request_and_next_request_completes() -> None:
    timed_out = _run_isolated(_blocking_target, (), 0.1)
    assert timed_out["failure_layer"] == "audit_wall_clock_timeout"
    assert _run_isolated(_quick_target, (), 2) == "completed"


def test_large_success_payload_is_read_before_child_join() -> None:
    payload = _run_isolated(_large_payload_target, (), 5)
    assert isinstance(payload, str)
    assert len(payload) == 4 * 1024 * 1024


def test_parent_throttle_preserves_interval_across_isolated_requests(monkeypatch) -> None:
    from scholar_agent.connectors.arxiv import _reset_arxiv_throttle_for_tests

    monkeypatch.setenv("SCHOLAR_AGENT_ARXIV_MIN_INTERVAL_SECONDS", "3")
    _reset_arxiv_throttle_for_tests()
    clock = [0.0]
    sleeps = []

    def monotonic() -> float:
        return clock[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    assert _parent_throttle("arxiv", sleep=sleep, monotonic=monotonic) == 0.0
    clock[0] = 1.0
    assert _parent_throttle("arxiv", sleep=sleep, monotonic=monotonic) == 2.0
    assert sleeps == [2.0]
    assert _parent_throttle("openalex", sleep=sleep, monotonic=monotonic) == 0.0
    assert sleeps == [2.0]


def test_current_signal_isolated_to_candidate_source_and_duplicate_title() -> None:
    row = {
        "stage_diagnostics": {
            "snapshots": [
                {"stage": "initial_retrieval", "candidates": [
                    {"title": "A Title", "identifiers": {"arxiv_id": "1"}, "sources": ["arxiv"]},
                    {"title": "A Title", "identifiers": {}, "sources": ["openalex"]},
                ]},
                {"stage": "initial_reranked", "candidates": [
                    {"title": "A Title", "identifiers": {"arxiv_id": "1"}, "sources": ["arxiv"], "rank": 2},
                ]},
                {"stage": "final_ranked", "candidates": [
                    {"title": "A Title", "identifiers": {"arxiv_id": "1"}, "sources": ["arxiv"], "rank": 2, "category": "partially_relevant"},
                ]},
                {"stage": "final_returned", "candidates": []},
            ]
        }
    }
    gold = {"arxiv_id": "1", "title": "A Title"}
    assert _current_signal(row, gold, "arxiv")["candidate_identity"] is True
    openalex = _current_signal(row, gold, "openalex")
    assert openalex["candidate_identity"] is False
    assert openalex["title_present_unmatched"] is True


def test_build_plan_has_one_gold_row_and_four_source_pairs() -> None:
    gold = SimpleNamespace(model_dump=lambda mode: {"arxiv_id": "1", "title": "A Title"})
    query = SimpleNamespace(query_id="AutoScholarQuery_test_0", query="q", gold_papers=[gold])
    current = {"dev": {query.query_id: {}}, "val": {}}
    requests, rows = build_request_plan(query and [query], current, ("arxiv", "openalex", "semantic_scholar", "pubmed"), 100)
    assert len(rows) == 1
    assert len(rows[0]["requests"]) == 4
    assert len(requests) == 12
