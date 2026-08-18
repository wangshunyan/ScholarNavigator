import sys
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_query_depth_curve import (  # noqa: E402
    _classify,
    _actual_queries,
    _match_gold,
    _paper_identity,
    _unique_prefix,
    _update_rate_limit_streak,
    _write_snapshot_if_missing,
    _record,
    _coverage_cell,
)


def _paper(
    title: str,
    arxiv_id: str | None = None,
    doi: str | None = None,
    year: int = 2020,
    authors: list[str] | None = None,
) -> dict:
    return {
        "title": title,
        "year": year,
        "authors": authors or [],
        "identifiers": {"arxiv_id": arxiv_id, "doi": doi},
    }


def test_prefix_dedup_and_first_hit_rank_are_stable() -> None:
    duplicate = _paper("Target", "1234.1")
    papers = [_paper("noise", "1"), duplicate, duplicate, _paper("later", "2")]
    unique = _unique_prefix(papers, 4)
    assert [_paper_identity(item) for item in unique] == ["arxiv_id:1", "arxiv_id:1234.1", "arxiv_id:2"]
    hit, rank, uncertain = _match_gold({"title": "Target", "arxiv_id": "1234.1"}, unique)
    assert (hit, rank, uncertain) == (True, 2, False)
    assert _classify(20, False, False) == "current_depth_hit"
    assert _classify(50, False, False) == "deeper_position_hit"


def test_deeper_and_miss_categories_are_mutually_explicit() -> None:
    assert _classify(50, False, False) == "deeper_position_hit"
    assert _classify(None, False, False) == "depth_200_miss"
    assert _classify(None, False, True) == "identity_match_uncertain"
    assert _classify(None, True, False) == "source_unavailable"
    assert _classify(None, True, False, True) == "source_unavailable"
    assert _classify(None, unavailable=False, uncertain=False, incomplete=True) == "source_unavailable_or_incomplete"
    assert _classify(None, unavailable=False, uncertain=False, incomplete=True) != "depth_200_miss"


def test_outage_coverage_cell_has_zero_evaluable_denominator() -> None:
    cell = _coverage_cell(40, 0, 40)
    assert cell["evaluable_gold_denominator"] == 0
    assert cell["unavailable_or_incomplete_gold"] == 40
    assert cell["observed_matched_gold"] == 0


def test_multiple_gold_matches_do_not_collapse() -> None:
    papers = [_paper("A", "a"), _paper("B", "b"), _paper("A", "a")]
    assert _match_gold({"title": "A", "arxiv_id": "a"}, _unique_prefix(papers, 200))[0]
    assert _match_gold({"title": "B", "arxiv_id": "b"}, _unique_prefix(papers, 200))[0]


def test_current_dedup_merges_cross_source_identifiers_and_title_variants() -> None:
    papers = [
        _paper(
            "A Study of Models",
            None,
            "10.1000/example",
            year=2020,
            authors=["Alice"],
        ),
        _paper(
            "A study of models",
            "arxiv:1234.1v2",
            year=2020,
            authors=["Alice"],
        ),
    ]
    unique = _unique_prefix(papers, 200)
    assert len(unique) == 1
    assert unique[0]["identifiers"]["doi"] == "10.1000/example"
    assert unique[0]["identifiers"]["arxiv_id"] == "arxiv:1234.1v2"


def test_actual_queries_use_executed_calls_in_order_and_keep_zero_result_call() -> None:
    rows = [{
        "case_id": "AutoScholarQuery_test_0",
        "query": "q",
        "stage_diagnostics": {"snapshots": [{"stage": "initial_retrieval", "retrieval_calls": [
            {"source": "arxiv", "adapted_query": "first", "logical_call_executed": True},
            {"source": "arxiv", "adapted_query": "skipped", "logical_call_executed": False},
            {"source": "arxiv", "adapted_query": "zero-result", "logical_call_executed": True},
            {"source": "arxiv", "adapted_query": "first", "logical_call_executed": True},
        ]}]},
    }]
    requests, query_rows = _actual_queries(rows)
    assert query_rows[0]["requests"]["arxiv"] == ["first", "zero-result"]
    assert len(requests) == 2


def test_only_consecutive_http429_failures_build_a_rate_limit_streak() -> None:
    timeout = {"status": "failed", "failure_layer": "network_error"}
    rate_limit = {"status": "failed", "failure_layer": "http_429_rate_limit"}
    streak, evidence = _update_rate_limit_streak(0, [], timeout)
    assert (streak, evidence) == (0, [])
    streak, _ = _update_rate_limit_streak(streak, evidence, rate_limit)
    assert streak == 1
    streak, _ = _update_rate_limit_streak(streak, [], timeout)
    assert streak == 0
    streak, evidence = _update_rate_limit_streak(0, [], rate_limit)
    streak, evidence = _update_rate_limit_streak(streak, evidence, rate_limit)
    assert streak == 2 and len(evidence) == 2


def test_snapshot_writer_never_overwrites_success_or_failed_terminal(tmp_path) -> None:
    path = tmp_path / "request.json"
    original = {"response": {"status": "success", "papers": [{"title": "real"}]}}
    replacement = {"response": {"status": "source_outage"}}
    assert _write_snapshot_if_missing(path, original) is True
    assert _write_snapshot_if_missing(path, replacement) is False
    assert path.read_text() == __import__("json").dumps(original, ensure_ascii=False, indent=2)


def test_runtime_429_circuit_breaker_reuses_immutable_evidence_and_rerun_is_read_only(tmp_path, monkeypatch) -> None:
    import scripts.audit_query_depth_curve as audit

    snapshot = tmp_path / "snap"
    output = tmp_path / "run"
    output.mkdir()
    (output / "probe_semantic_scholar.json").write_text(json.dumps({
        "source": "semantic_scholar", "timestamp": "2026-07-20T00:00:00+00:00",
        "status": "success", "error_message": None, "error_type": None,
        "diagnostics": {"request_count": 1},
    }))
    requests = {str(i): {"key": str(i), "source": "semantic_scholar", "query": f"q{i}", "limit": 200} for i in range(5)}
    calls = []

    def fake_fetch(request, timeout):
        calls.append(request["key"])
        if len(calls) == 1:
            return {"status": "success", "papers": [], "diagnostics": {"request_count": 1}}
        return {"status": "failed", "failure_layer": "http_429_rate_limit", "error_message": "429", "diagnostics": {"request_count": 2}}

    monkeypatch.setattr(audit, "_fetch_isolated", fake_fetch)
    monkeypatch.setattr(audit, "_parent_throttle", lambda source: 0.0)
    args = SimpleNamespace(snapshot=str(snapshot), output=str(output), request_timeout=1)
    _record(args, requests)
    assert calls == ["0", "1", "2"]
    payloads = [json.loads((snapshot / f"{i}.json").read_text()) for i in range(5)]
    assert [item["response"]["status"] for item in payloads] == ["success", "failed", "failed", "source_outage", "source_outage"]
    hashes = {item["response"]["probe_evidence_hash"] for item in payloads[3:]}
    assert len(hashes) == 1
    trigger = json.loads((output / "runtime_outage_semantic_scholar.json").read_text())["source_level_trigger"]
    assert trigger["attempted_count"] == 3 and trigger["not_sent_count"] == 2
    before = [(item["response"]["status"], item["response"].get("probe_evidence_hash")) for item in payloads]
    calls.clear()
    _record(args, requests)
    after = [(json.loads((snapshot / f"{i}.json").read_text())["response"]["status"], json.loads((snapshot / f"{i}.json").read_text())["response"].get("probe_evidence_hash")) for i in range(5)]
    assert calls == [] and before == after
