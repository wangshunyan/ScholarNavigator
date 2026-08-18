from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.evaluation.scifact_source_index_audit import (
    ExactLookupResponse,
    ExactLookupStore,
    _parse_arxiv_identifiers,
    _parse_pubmed_identifiers,
    _source_result,
    aggregate_records,
    build_exact_request,
    build_request_plan,
    classify_gold,
    fetch_exact_lookup,
    record_missing,
    replay_audit,
    run_preflight,
    write_replay_artifacts,
)


FIXED_TIME = "2026-07-21T00:00:00+00:00"


class _Response:
    def __init__(self, payload: bytes | dict[str, Any], status: int = 200) -> None:
        self.payload = (
            json.dumps(payload).encode("utf-8")
            if isinstance(payload, dict)
            else payload
        )
        self.status = status
        self.headers: dict[str, str] = {}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _gold(
    corpus_id: str = "42",
    *,
    doi: str | None = "10.1000/example",
    pmid: str | None = "123",
    arxiv_id: str | None = None,
) -> EvalGoldPaper:
    return EvalGoldPaper(
        title="Gold title",
        s2orc_corpus_id=corpus_id,
        semantic_scholar_id=f"s2-{corpus_id}",
        doi=doi,
        arxiv_id=arxiv_id,
        pubmed_id=pmid,
        metadata={"evaluator_crosswalk": {"status": "success"}},
    )


def _terminal(
    status: str,
    *,
    identifiers: dict[str, str] | None = None,
    requested: bool = True,
    error_type: str | None = None,
    http_status: int | None = 200,
) -> ExactLookupResponse:
    return ExactLookupResponse(
        status=status,
        requested=requested,
        returned_identifiers=identifiers or {},
        error_type=error_type,
        http_status=http_status,
        request_count=int(requested),
        recorded_at=FIXED_TIME,
    )


def test_source_aware_request_selection_uses_only_exact_supported_identifiers() -> None:
    gold = _gold(arxiv_id="arXiv:2401.00001v2")
    assert build_exact_request("arxiv", gold).identifier_value == "2401.00001"
    assert build_exact_request("openalex", gold).identifier_type == "doi"
    assert build_exact_request("semantic_scholar", gold).identifier_type == "s2orc_corpus_id"
    assert build_exact_request("pubmed", gold).identifier_value == "123"

    without_arxiv = _gold(arxiv_id=None)
    request = build_exact_request("arxiv", without_arxiv)
    assert request.applicable is False
    assert request.identifier_value is None


def test_plan_keeps_multi_gold_relations_but_deduplicates_duplicate_source_calls() -> None:
    shared = _gold("42")
    queries = [
        EvalQuery(query_id="q1", query="one", gold_papers=[shared, _gold("43")]),
        EvalQuery(query_id="q2", query="two", gold_papers=[shared]),
    ]
    requests, records = build_request_plan(queries)
    assert len(records) == 3
    assert sum(item.source == "semantic_scholar" for item in requests.values()) == 2
    assert records[0]["requests"]["semantic_scholar"] == records[2]["requests"]["semantic_scholar"]


def test_exact_parsers_cover_arxiv_pubmed_and_missing_records() -> None:
    arxiv = b"""<feed xmlns='http://www.w3.org/2005/Atom' xmlns:a='http://arxiv.org/schemas/atom'><entry><id>https://arxiv.org/abs/2401.00001v2</id><a:doi>10.1000/EXAMPLE</a:doi></entry></feed>"""
    assert _parse_arxiv_identifiers(arxiv) == (
        {"arxiv_id": "2401.00001", "doi": "10.1000/example"},
        None,
    )
    pubmed = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ReferenceList><Reference><ArticleIdList><ArticleId IdType='pubmed'>999</ArticleId><ArticleId IdType='doi'>10.1000/REFERENCE</ArticleId></ArticleIdList></Reference></ReferenceList></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType='pubmed'>123</ArticleId><ArticleId IdType='doi'>10.1000/EXAMPLE</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
    assert _parse_pubmed_identifiers(pubmed) == (
        {"pubmed_id": "123", "doi": "10.1000/example"},
        None,
    )
    assert _parse_pubmed_identifiers(b"<PubmedArticleSet />") == ({}, "not_found")


def test_exact_fetch_retries_429_and_preserves_terminal_without_exception_text() -> None:
    request = build_exact_request("openalex", _gold())
    calls = 0
    sleeps: list[float] = []

    def opener(req: Any, *, timeout: float) -> Any:
        nonlocal calls
        calls += 1
        raise HTTPError(req.full_url, 429, "rate limited", {}, None)

    response = fetch_exact_lookup(
        request,
        opener=opener,
        sleep=sleeps.append,
        timeout_seconds=0.1,
        max_retries=1,
    )
    assert calls == 2
    assert response.status == "failed"
    assert response.error_type == "status_429"
    assert response.http_status == 429
    assert response.request_count == 2
    assert response.retry_count == 1
    assert sleeps == [0.5]


def test_preflight_outage_materializes_unrequested_snapshots_and_replay_validates(
    tmp_path: Path,
) -> None:
    request = build_exact_request("openalex", _gold())
    failed = _terminal(
        "failed", error_type="status_429", http_status=429
    ).model_copy(update={"retry_count": 1, "request_count": 2})
    evidence = run_preflight(
        "openalex", [request], runner=lambda *_args: failed
    )
    assert evidence.status == "failed"
    store = ExactLookupStore(tmp_path / "snapshots")
    counts = record_missing(
        [request],
        store,
        {"openalex": evidence},
        source="openalex",
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("source outage must not issue per-gold HTTP")
        ),
    )
    snapshot = store.read(request)
    assert counts["source_outage"] == 1
    assert snapshot.response.status == "source_outage"
    assert snapshot.response.requested is False
    assert snapshot.response.request_count == 0
    assert snapshot.response.preflight_evidence_hash == evidence.content_hash


def test_unified_identity_conflict_is_not_an_exact_hit() -> None:
    result = _source_result(
        _terminal(
            "success",
            identifiers={
                "s2orc_corpus_id": "42",
                "doi": "10.1000/conflicting",
            },
        ),
        _gold("42", doi="10.1000/original"),
    )
    assert result["terminal"] == "identity_evidence_insufficient"
    assert result["identity_rule"] == "conflicting_stable_identifier"
    assert result["conflicting_identifiers"]


def test_gold_classification_is_mutually_exclusive_and_conservative() -> None:
    rows = {
        "arxiv": {"terminal": "not_applicable"},
        "openalex": {"terminal": "not_found"},
        "semantic_scholar": {"terminal": "exact_hit"},
        "pubmed": {"terminal": "source_unavailable"},
    }
    assert classify_gold(True, rows) == "current_query_recalled"
    assert classify_gold(False, rows) == "source_exactly_locatable_query_miss"
    rows["semantic_scholar"] = {"terminal": "not_found"}
    assert classify_gold(False, rows) == "source_unavailable"
    rows["pubmed"] = {"terminal": "not_found"}
    assert classify_gold(False, rows) == "applicable_sources_not_located"
    rows["openalex"] = {"terminal": "identity_evidence_insufficient"}
    assert classify_gold(False, rows) == "identity_evidence_insufficient"


def test_replay_is_zero_http_cross_source_deduplicated_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    gold = _gold()
    query = EvalQuery(query_id="q", query="original query", gold_papers=[gold])
    requests, plan = build_request_plan([query])
    store = ExactLookupStore(tmp_path / "snapshots")
    returned = {
        "openalex": {"doi": "10.1000/example", "openalex_id": "w1"},
        "semantic_scholar": {"s2orc_corpus_id": "42", "semantic_scholar_id": "s2-42"},
        "pubmed": {"pubmed_id": "123", "doi": "10.1000/example"},
    }
    for request in requests.values():
        if not request.applicable:
            response = _terminal("not_applicable", requested=False, http_status=None)
        else:
            response = _terminal("success", identifiers=returned[request.source])
        store.write(request, response)

    run = tmp_path / "external"
    run.mkdir()
    row = {
        "case_id": "q",
        "status": "succeeded",
        "stage_diagnostics": {
            "snapshots": [
                {
                    "stage": "initial_retrieval",
                    "status": "completed",
                    "candidates": [
                        {"title": "Gold title", "identifiers": {}, "sources": ["openalex"]}
                    ],
                }
            ]
        },
    }
    (run / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    records, aggregate = replay_audit(
        queries=[query],
        requests=requests,
        gold_plan=plan,
        store=store,
        external_run_dir=run,
    )
    assert records[0]["current_query_recalled"] is False
    assert records[0]["classification"] == "source_exactly_locatable_query_miss"
    assert aggregate["joint_exact_coverage"]["matched_gold_count"] == 1
    assert aggregate["current_plus_exact_coverage_upper_bound"]["matched_gold_count"] == 1
    assert aggregate["by_source"]["semantic_scholar"]["exact_hit_gold_count"] == 1
    assert aggregate["strategy_space"]["query_expression_theoretical_max_new_gold_count"] == 1
    assert aggregate["source_pair_count"] == 4
    assert sum(aggregate["classification_counts"].values()) == 1

    config = {"audit": "test", "query_input": "none"}
    first = tmp_path / "first"
    second = tmp_path / "second"
    hashes_one = write_replay_artifacts(
        first, config=config, records=records, aggregate=aggregate
    )
    hashes_two = write_replay_artifacts(
        second, config=config, records=records, aggregate=aggregate
    )
    assert hashes_one == hashes_two
    assert {
        item.name: item.read_bytes() for item in first.iterdir()
    } == {
        item.name: item.read_bytes() for item in second.iterdir()
    }
