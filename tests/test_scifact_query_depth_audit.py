import json
from pathlib import Path

from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.scifact_query_depth_audit import (
    SOURCES,
    DepthSnapshotStore,
    _depth_bucket,
    _first_exact_rank,
    build_prefix_pool,
    build_request_plan,
    classify_gold_depth,
    first_hit_evidence,
    list_terminal_status,
    list_prefix,
    page_requests,
    record_missing,
    source_outage_response,
    stable_hash,
    write_replay_artifacts,
)


def _paper(
    title: str,
    *,
    doi: str | None = None,
    s2orc: str | None = None,
    source: str = "pubmed",
) -> Paper:
    return Paper(
        title=title,
        year=2020,
        authors=["A. Author"],
        sources=[source],
        identifiers=PaperIdentifiers(doi=doi, s2orc_corpus_id=s2orc),
    )


def _response(status="success", papers=()):
    return {
        "status": status,
        "requested": status != "source_outage",
        "error_type": None if status == "success" else "network_timeout",
        "http_status": 200 if status == "success" else None,
        "warnings": [],
        "diagnostics": {
            "request_count": int(status != "source_outage"),
            "retry_count": 0,
            "error_count": int(status != "success"),
            "cache_hit_count": 0,
            "rate_limit_wait_seconds": 0.0,
            "latency_seconds": 0.1,
        },
        "latency_seconds": 0.1,
        "papers": [item.model_dump(mode="json") for item in papers],
    }


def _request(source: str, query: str, offset: int = 0, limit: int = 200):
    request = {
        "schema_version": "scifact-query-depth-v1",
        "source": source,
        "adapted_query": query,
        "offset": offset,
        "limit": limit,
        "max_depth": 200,
    }
    return {**request, "key": stable_hash(request)}


def test_build_plan_keeps_all_planned_calls_in_original_order_without_gold() -> None:
    calls = [
        {
            "source": "openalex",
            "adapted_query": "first",
            "origin_subquery": "original",
            "logical_call_executed": False,
            "terminal_status": "not_started",
        },
        {
            "source": "pubmed",
            "adapted_query": "second",
            "origin_subquery": "derived",
            "logical_call_executed": True,
            "terminal_status": "success",
        },
    ]
    row = {
        "case_id": "q1",
        "query": "normal query",
        "stage_diagnostics": {
            "initial_query_planning": {"query_analysis": {"original_query": "q"}},
            "snapshots": [
                {"stage": "initial_retrieval", "retrieval_calls": calls}
            ],
        },
    }
    config = {"case_ids": ["q1"], "sources": list(SOURCES)}
    requests, cases = build_request_plan([row], config)
    assert [item["adapted_query"] for item in cases[0]["lists"]] == [
        "first",
        "second",
    ]
    assert cases[0]["lists"][0]["baseline_logical_call_executed"] is False
    assert len(requests) == 3  # OpenAlex one page; PubMed two pages.
    serialized = json.dumps(list(requests.values()))
    assert "gold" not in serialized and "case_id" not in serialized


def test_page_plan_uses_one_max_page_or_two_stable_pages() -> None:
    assert [(item["offset"], item["limit"]) for item in page_requests("openalex", "q")] == [
        (0, 200)
    ]
    assert [(item["offset"], item["limit"]) for item in page_requests("pubmed", "q")] == [
        (0, 100),
        (100, 100),
    ]


def test_prefixes_come_from_the_same_pages_and_partial_page_is_incomplete() -> None:
    first = _request("pubmed", "q", 0, 100)
    second = _request("pubmed", "q", 100, 100)
    item = {
        "source": "pubmed",
        "page_keys": [first["key"], second["key"]],
    }
    first_page = [_paper(f"paper-{index}", doi=f"10.1/{index}") for index in range(100)]
    responses = {
        first["key"]: _response(papers=first_page),
        second["key"]: _response("failed"),
    }
    papers20, complete20 = list_prefix(item, responses, 20)
    papers100, complete100 = list_prefix(item, responses, 100)
    papers200, complete200 = list_prefix(item, responses, 200)
    assert [len(papers20), len(papers100), len(papers200)] == [20, 100, 100]
    assert (complete20, complete100, complete200) == (True, True, False)
    assert list_terminal_status(item, responses) == "incomplete"


def test_short_first_page_is_an_exhausted_complete_list() -> None:
    first = _request("pubmed", "q", 0, 100)
    second = _request("pubmed", "q", 100, 100)
    item = {
        "source": "pubmed",
        "page_keys": [first["key"], second["key"]],
    }
    papers, complete = list_prefix(
        item,
        {
            first["key"]: _response(papers=[_paper("only", doi="10.1/only")]),
            second["key"]: _response("failed"),
        },
        200,
    )
    assert len(papers) == 1 and complete is True


def test_exact_identity_matching_never_infers_from_title() -> None:
    gold = EvalGoldPaper(title="Same Title", s2orc_corpus_id="42")
    assert _first_exact_rank([_paper("Same Title", doi="10.1/different")], gold) is None
    assert _first_exact_rank([_paper("Different", s2orc="42")], gold) == 1


def test_depth_buckets_and_incomplete_classification_are_mutually_exclusive() -> None:
    assert [_depth_bucket(rank) for rank in (20, 21, 50, 51, 100, 101, 200)] == [
        "current_depth_hit",
        "depth_21_50_hit",
        "depth_21_50_hit",
        "depth_51_100_hit",
        "depth_51_100_hit",
        "depth_101_200_hit",
        "depth_101_200_hit",
    ]
    oracle = {"sources": {"pubmed": {"terminal": "exact_hit"}}}
    complete = {"pubmed": {"complete_at_200": True}}
    incomplete = {"pubmed": {"complete_at_200": False}}
    assert classify_gold_depth(
        first_rank=None, oracle_row=oracle, source_completeness=complete
    ) == "depth_200_miss"
    assert classify_gold_depth(
        first_rank=None, oracle_row=oracle, source_completeness=incomplete
    ) == "source_unavailable_or_incomplete"


def test_multi_gold_first_hit_and_identity_dedup_use_stable_ids() -> None:
    first = _request("openalex", "q")
    item = {
        "list_order": 0,
        "source": "openalex",
        "origin_subquery": "q",
        "adapted_query": "q",
        "page_keys": [first["key"]],
    }
    duplicate_a = _paper("A", doi="https://doi.org/10.1/a", source="openalex")
    duplicate_b = _paper("A variant", doi="10.1/a", source="pubmed")
    target = _paper("B", doi="10.1/b", source="openalex")
    responses = {first["key"]: _response(papers=[duplicate_a, duplicate_b, target])}
    case = {"lists": [item]}
    unbounded, candidates, complete = build_prefix_pool(
        case, responses, prefix=200, candidate_limit=200
    )
    assert complete is True and len(unbounded) == len(candidates) == 2
    assert first_hit_evidence(
        case, responses, EvalGoldPaper(doi="10.1/b")
    )["first_list_rank"] == 3


def test_record_isolates_failure_and_does_not_stop_later_request(tmp_path) -> None:
    requests = [_request("openalex", f"q{index}") for index in range(3)]
    store = DepthSnapshotStore(tmp_path / "snap")
    calls = []

    def runner(request, timeout):
        del timeout
        calls.append(request["adapted_query"])
        if request["adapted_query"] == "q0":
            return _response("failed")
        return _response(papers=[])

    evidence = {
        "schema_version": "scifact-query-depth-v1",
        "source": "openalex",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "status": "success",
        "error_type": None,
        "http_status": 200,
        "diagnostics": {"request_count": 1},
    }
    counts = record_missing(
        requests,
        store,
        {"openalex": evidence},
        runner=runner,
        throttle=lambda source: 0.0,
        wall_timeout_seconds=1,
    )
    assert calls == ["q0", "q1", "q2"]
    assert counts["failed"] == 1 and counts["success"] == 2
    calls.clear()
    rerun = record_missing(
        requests,
        store,
        {"openalex": evidence},
        runner=runner,
        throttle=lambda source: 0.0,
        wall_timeout_seconds=1,
    )
    assert calls == [] and rerun["pending"] == 0


def test_consecutive_429_materializes_shared_outage_without_early_gold_stop(tmp_path) -> None:
    requests = [_request("semantic_scholar", f"q{index}", 0, 100) for index in range(5)]
    store = DepthSnapshotStore(tmp_path / "snap")
    calls = []

    def runner(request, timeout):
        del timeout
        calls.append(request["adapted_query"])
        if len(calls) == 1:
            return _response(papers=[_paper("observed", s2orc="1")])
        response = _response("failed")
        response["error_type"] = "http_429_rate_limit"
        response["http_status"] = 429
        return response

    evidence = {
        "schema_version": "scifact-query-depth-v1",
        "source": "semantic_scholar",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "status": "success",
        "error_type": None,
        "http_status": 200,
        "diagnostics": {"request_count": 1},
    }
    record_missing(
        requests,
        store,
        {"semantic_scholar": evidence},
        runner=runner,
        throttle=lambda source: 0.0,
        wall_timeout_seconds=1,
    )
    assert calls == ["q0", "q1", "q2"]
    stored = [store.read(item) for item in requests]
    assert [item["status"] for item in stored] == [
        "success",
        "failed",
        "failed",
        "source_outage",
        "source_outage",
    ]
    assert len({item["probe_evidence_hash"] for item in stored[3:]}) == 1
    assert all(item["requested"] is False for item in stored[3:])


def test_outage_snapshot_validation_and_replay_output_are_deterministic(tmp_path) -> None:
    request = _request("openalex", "q")
    evidence = {
        "schema_version": "scifact-query-depth-v1",
        "source": "openalex",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "status": "failed",
        "error_type": "http_429_rate_limit",
        "http_status": 429,
        "diagnostics": {"request_count": 2},
    }
    store = DepthSnapshotStore(tmp_path / "snap")
    store.write(request, source_outage_response(evidence))
    assert store.read(request)["status"] == "source_outage"

    records = [{"case_id": "1", "classification": "depth_200_miss"}]
    aggregate = {"network_request_count": 0, "snapshot_write_count": 0}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_replay_artifacts(first, config={"fixed": True}, records=records, aggregate=aggregate)
    write_replay_artifacts(second, config={"fixed": True}, records=records, aggregate=aggregate)
    for name in ("config.json", "gold_depth_audit.jsonl", "aggregate.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
