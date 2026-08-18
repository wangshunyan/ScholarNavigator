from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.evaluation.retrieval_recall_audit import (
    AuditPaperMetadata,
    AuditSnapshotEntry,
    AuditSnapshotRuntime,
    AuditSnapshotStore,
    adapter_term_loss,
    audit_entry_hash,
    classify_failure,
    collect_audit_requests,
    content_tokens,
    exact_title_query,
    lexical_overlap,
    make_audit_request,
    normalized_title_query,
    parse_arxiv_audit_feed,
    recall_at_k,
    title_core_query,
    write_audit_outputs,
)


ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>Graph Learning for Molecules</title>
    <summary>A message passing benchmark on QM9.</summary>
    <author><name>Alice Example</name></author>
    <author><name>Bob Example</name></author>
    <category term="cs.LG"/>
    <category term="cs.AI"/>
    <arxiv:doi>10.1000/example</arxiv:doi>
  </entry>
</feed>
"""


def _entry(tmp_path: Path) -> tuple[AuditSnapshotStore, AuditSnapshotEntry]:
    request = make_audit_request("current_query", query="all:graph", max_results=100)
    entry = AuditSnapshotEntry(
        key=request.key,
        request=request,
        status="success",
        papers=[
            AuditPaperMetadata(
                arxiv_id="2401.00001",
                title="Graph Learning for Molecules",
                abstract="A message passing benchmark on QM9.",
                authors=["Alice Example"],
                year=2024,
                categories=["cs.LG"],
                doi="10.1000/example",
            )
        ],
        diagnostics=ConnectorDiagnostics(request_count=1),
        recorded_latency_seconds=0.2,
        recorded_at="2026-01-01T00:00:00+00:00",
        content_hash="0" * 64,
    )
    entry = entry.model_copy(update={"content_hash": audit_entry_hash(entry)})
    store = AuditSnapshotStore(tmp_path)
    store.write(entry)
    return store, entry


def test_parse_arxiv_metadata_includes_required_fields() -> None:
    papers = parse_arxiv_audit_feed(ATOM)

    assert papers == [
        AuditPaperMetadata(
            arxiv_id="2401.00001",
            title="Graph Learning for Molecules",
            abstract="A message passing benchmark on QM9.",
            authors=["Alice Example", "Bob Example"],
            year=2024,
            categories=["cs.LG", "cs.AI"],
            doi="10.1000/example",
        )
    ]


def test_lexical_overlap_is_normalized_and_deterministic() -> None:
    first = lexical_overlap(
        "Methods for graph learning on QM9",
        "Graph-learning benchmark with the QM9 molecular dataset",
    )
    second = lexical_overlap(
        "Methods for graph learning on QM9",
        "Graph-learning benchmark with the QM9 molecular dataset",
    )

    assert first == second
    assert first["query_tokens"] == ["methods", "graph", "learning", "qm9"]
    assert first["matched_tokens"] == ["graph", "learning", "qm9"]
    assert first["query_coverage"] == pytest.approx(0.75)


def test_adapter_term_loss_only_counts_gold_relevant_query_terms() -> None:
    loss = adapter_term_loss(
        "graph transformers for QM9 molecules",
        ["(ti:graph OR abs:graph) AND (ti:transformers OR abs:transformers)"],
        "QM9 molecular graph transformers benchmark",
    )

    assert loss["relevant_original_terms"] == [
        "graph",
        "transformers",
        "qm9",
    ]
    assert loss["retained_terms"] == ["graph", "transformers"]
    assert loss["lost_terms"] == ["qm9"]
    assert loss["loss_detected"] is True


@pytest.mark.parametrize(
    ("signals", "expected"),
    [
        ({"source_error": True}, "source_error"),
        ({"identifier_available": False}, "source_unavailable"),
        (
            {"identifier_available": True, "metadata_mismatch": True},
            "metadata_mismatch",
        ),
        (
            {"identifier_available": True, "current_query_rank": 25},
            "result_limit_truncation",
        ),
        (
            {"identifier_available": True, "current_query_rank": 75},
            "query_over_broad_ranked_below_limit",
        ),
        (
            {"identifier_available": True, "adapter_term_loss": True},
            "adapter_term_loss",
        ),
        (
            {"identifier_available": True, "lexical_query_coverage": 0.1},
            "lexical_mismatch",
        ),
        (
            {
                "identifier_available": True,
                "lexical_query_coverage": 0.5,
                "query_over_restrictive": True,
            },
            "query_over_restrictive",
        ),
        (
            {
                "identifier_available": True,
                "lexical_query_coverage": 0.5,
                "any_title_oracle_hit": True,
            },
            "identifier_available_but_query_not_matched",
        ),
        (
            {"identifier_available": True, "lexical_query_coverage": 0.5},
            "unknown",
        ),
    ],
)
def test_failure_reason_classification_is_stable(
    signals: dict,
    expected: str,
) -> None:
    assert classify_failure(signals) == expected


def test_top_k_oracle_statistics() -> None:
    ranks = [1, 20, 21, 50, 51, 100, None]

    assert recall_at_k(ranks, 20) == {
        "k": 20,
        "recovered_count": 2,
        "total_count": 7,
        "recall": pytest.approx(2 / 7),
    }
    assert recall_at_k(ranks, 50)["recovered_count"] == 4
    assert recall_at_k(ranks, 100)["recovered_count"] == 6


def test_audit_replay_never_invokes_network_fetcher(tmp_path: Path) -> None:
    store, entry = _entry(tmp_path)
    runtime = AuditSnapshotRuntime(store, mode="replay")

    replayed = runtime.resolve(
        entry.request,
        lambda _request: pytest.fail("replay must not invoke fetcher"),
    )

    assert replayed == entry
    assert runtime.cost.snapshot_hits == 1
    assert runtime.cost.execution_request_count == 0
    assert runtime.cost.execution_retry_count == 0
    assert runtime.cost.execution_network_wait_seconds == 0


def test_audit_snapshot_detects_content_changes(tmp_path: Path) -> None:
    store, entry = _entry(tmp_path)
    path = store.entries_dir / f"{entry.key}.json"
    payload = path.read_text(encoding="utf-8").replace(
        "Graph Learning for Molecules",
        "Changed Title",
    )
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="hash_mismatch"):
        store.read(entry.key)


def test_collection_stops_after_consecutive_source_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = [
        make_audit_request("current_query", query=f"all:query{index}")
        for index in range(5)
    ]

    def failed_fetcher(request):
        entry = AuditSnapshotEntry(
            key=request.key,
            request=request,
            status="failed",
            error_message="HTTP 429",
            diagnostics=ConnectorDiagnostics(request_count=2, retry_count=1, error_count=1),
            recorded_latency_seconds=0.1,
            recorded_at="2026-01-01T00:00:00+00:00",
            content_hash="0" * 64,
        )
        return entry.model_copy(update={"content_hash": audit_entry_hash(entry)})

    monkeypatch.setattr(
        "scholar_agent.evaluation.retrieval_recall_audit.fetch_arxiv_audit_request",
        failed_fetcher,
    )
    result = collect_audit_requests(
        AuditSnapshotStore(tmp_path),
        requests,
        source_failure_limit=2,
    )

    assert result["attempted_request_count"] == 2
    assert result["remaining_missing_count"] == 3
    assert result["stop_reason"] == "source_failure_limit:2"


def test_record_missing_resumes_without_refetching_existing_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, existing = _entry(tmp_path)
    missing = make_audit_request("current_query", query="all:second")
    fetched: list[str] = []

    def successful_fetcher(request):
        fetched.append(request.key)
        entry = AuditSnapshotEntry(
            key=request.key,
            request=request,
            status="success",
            diagnostics=ConnectorDiagnostics(request_count=1),
            recorded_latency_seconds=0.1,
            recorded_at="2026-01-01T00:00:00+00:00",
            content_hash="0" * 64,
        )
        return entry.model_copy(update={"content_hash": audit_entry_hash(entry)})

    monkeypatch.setattr(
        "scholar_agent.evaluation.retrieval_recall_audit.fetch_arxiv_audit_request",
        successful_fetcher,
    )
    result = collect_audit_requests(store, [existing.request, missing])

    assert fetched == [missing.key]
    assert result["attempted_request_count"] == 1
    assert result["remaining_missing_count"] == 0
    assert result["replay_ready"] is True


def test_oracle_queries_are_deterministic_and_separate() -> None:
    title = "Graph Learning: A QM9 Benchmark"

    assert exact_title_query(title) == 'ti:"Graph Learning: A QM9 Benchmark"'
    assert normalized_title_query(title) == 'ti:"graph learning a qm9 benchmark"'
    assert title_core_query(title) == (
        "all:graph AND all:learning AND all:qm9 AND all:benchmark"
    )
    assert content_tokens(title) == ["graph", "learning", "qm9", "benchmark"]


def test_audit_outputs_are_deterministic(tmp_path: Path) -> None:
    aggregate = {
        "gold_count": 1,
        "existing_retrieved_gold_count": 0,
        "identifier_available_count": 1,
        "identifier_available_rate": 1.0,
        "exact_title_recovered_count": 1,
        "normalized_title_recovered_count": 1,
        "title_core_recovered_count": 1,
        "current_query_oracle_recall_at_k": {
            str(k): recall_at_k([None], k) for k in (20, 50, 100)
        },
        "failure_reason_distribution": {"lexical_mismatch": 1},
        "primary_bottleneck": "lexical_mismatch",
        "primary_bottleneck_family": "lexical_mismatch",
        "next_experiment_recommendation": "固定实验。",
    }
    gold = [{"case_id": "case-1", "arxiv_id": "2401.00001"}]
    queries = [{"case_id": "case-1", "gold_count": 1}]

    write_audit_outputs(tmp_path, gold, queries, aggregate)
    first = {path.name: path.read_bytes() for path in sorted(tmp_path.iterdir())}
    write_audit_outputs(tmp_path, gold, queries, aggregate)
    second = {path.name: path.read_bytes() for path in sorted(tmp_path.iterdir())}

    assert first == second
    assert set(first) == {
        "aggregate.json",
        "gold_audit.jsonl",
        "query_summary.jsonl",
        "summary.md",
    }


def test_gold_audit_module_is_not_imported_by_production_search() -> None:
    root = Path(__file__).resolve().parents[1]
    production_paths = [
        root / "src/scholar_agent/services/search_service.py",
        root / "src/scholar_agent/agents/query_understanding.py",
        root / "src/scholar_agent/retrieval/query_adapter.py",
    ]

    for path in production_paths:
        text = path.read_text(encoding="utf-8")
        assert "retrieval_recall_audit" not in text
        assert "gold_audit" not in text
