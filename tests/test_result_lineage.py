from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from threading import RLock
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.check_result_lineage as lineage_cli  # noqa: E402
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats  # noqa: E402
from scholar_agent.core.dedup import (  # noqa: E402
    deduplicate_papers,
    deduplicate_papers_with_lineage,
)
from scholar_agent.core.paper_schemas import (  # noqa: E402
    Paper,
    PaperIdentifiers,
    PaperUrls,
)
from scholar_agent.core.result_lineage import (  # noqa: E402
    RESULT_LINEAGE_CONTRACT,
    opaque_query_identity,
    restrict_result_lineage_document,
    run_manifest_output_spec,
    stable_json_bytes,
)
from scholar_agent.evaluation.crash_consistency import (  # noqa: E402
    BenchmarkRunCommitStore,
)
from scholar_agent.evaluation.result_lineage import (  # noqa: E402
    EXIT_LINEAGE_VIOLATION,
    EXIT_NOT_ELIGIBLE,
    ResultLineageNotEligible,
    audit_frozen_baseline_eligibility,
    load_protocol,
    run_result_lineage_gate,
    validate_result_lineage_document,
    write_json,
)
from scholar_agent.evaluation.run_provenance import output_identity  # noqa: E402
from scholar_agent.services.search_service import SearchService  # noqa: E402


PROTOCOL = ROOT / "benchmark" / "field_lineage_v1_protocol.json"
QUERY_IDENTITY = opaque_query_identity("local result lineage fixture")


def _paper(
    title: str,
    *,
    source: str = "openalex",
    authors: list[str] | None = None,
    year: int | None = 2024,
    venue: str | None = None,
    abstract: str = "",
    doi: str | None = None,
    arxiv_id: str | None = None,
    semantic_scholar_id: str | None = None,
    s2orc_corpus_id: str | None = None,
    openalex_id: str | None = None,
    pubmed_id: str | None = None,
    landing_page: str | None = None,
    pdf: str | None = None,
    citation_count: int = 0,
) -> Paper:
    return Paper(
        title=title,
        authors=authors or [],
        year=year,
        venue=venue,
        abstract=abstract,
        identifiers=PaperIdentifiers(
            doi=doi,
            arxiv_id=arxiv_id,
            semantic_scholar_id=semantic_scholar_id,
            s2orc_corpus_id=s2orc_corpus_id,
            openalex_id=openalex_id,
            pubmed_id=pubmed_id,
        ),
        urls=PaperUrls(landing_page=landing_page, pdf=pdf),
        sources=[source],
        citation_count=citation_count,
    )


def _lineage(
    papers: list[Paper],
    terminals: list[dict[str, Any]] | None = None,
) -> tuple[list[Paper], dict[str, Any]]:
    output, _, document = deduplicate_papers_with_lineage(
        papers,
        query_identity=QUERY_IDENTITY,
        source_terminals=terminals,
    )
    return output, document


def _decision(document: dict[str, Any], result: int, field: str) -> dict[str, Any]:
    return next(
        item
        for item in document["results"][result]["field_decisions"]
        if item["field"] == field
    )


def test_single_source_direct_mapping_reconstructs_every_paper_field() -> None:
    paper = _paper(
        "Direct Mapping",
        authors=["Alice"],
        venue="Venue",
        abstract="Evidence text",
        doi="https://doi.org/10.1000/DIRECT",
        openalex_id="W1",
        landing_page="https://example.test/paper",
        citation_count=7,
    )
    output, document = _lineage([paper])

    assert output == [paper]
    assert validate_result_lineage_document(document) == []
    result = document["results"][0]
    assert result["contributing_sources"] == ["openalex"]
    assert len(result["field_decisions"]) == 15
    assert _decision(document, 0, "title")["selected_value"] == "Direct Mapping"
    assert _decision(document, 0, "identifiers.doi")["candidates"][0][
        "normalized_value"
    ] == "10.1000/direct"


def test_multisource_duplicate_records_merge_complementary_fields_with_evidence() -> None:
    first = _paper(
        "Short title",
        source="openalex",
        authors=["Alice"],
        doi="10.1000/shared",
        openalex_id="W1",
        abstract="short",
    )
    second = _paper(
        "A longer complete title",
        source="arxiv",
        authors=["Alice", "Bob"],
        doi="https://doi.org/10.1000/SHARED",
        arxiv_id="2401.00001v2",
        abstract="a much longer abstract",
        pdf="https://example.test/paper.pdf",
    )
    third = second.model_copy(deep=True)
    output, document = _lineage([first, second, third])

    assert output == deduplicate_papers([first, second, third])
    assert len(output) == 1
    assert len(document["source_records"]) == 3
    assert len(set(item["record_ref"] for item in document["source_records"])) == 3
    assert [item["action"] for item in document["results"][0]["cluster_members"]] == [
        "cluster_created",
        "merged",
        "merged",
    ]
    assert _decision(document, 0, "abstract")["selected_value"] == (
        "a much longer abstract"
    )
    assert _decision(document, 0, "identifiers.openalex_id")["selected_value"] == "W1"
    assert _decision(document, 0, "identifiers.arxiv_id")["selected_value"] == (
        "2401.00001v2"
    )
    assert validate_result_lineage_document(document) == []


def test_identifier_conflict_stays_in_separate_clusters_and_is_audited() -> None:
    first = _paper(
        "Same title", authors=["Alice"], doi="10.1000/a", openalex_id="W1"
    )
    second = _paper(
        "Same title",
        source="semantic_scholar",
        authors=["Alice"],
        doi="10.1000/b",
        semantic_scholar_id="S2-B",
    )
    output, document = _lineage([first, second])

    assert len(output) == 2
    assert len(document["rejected_identity_comparisons"]) == 1
    conflict = document["rejected_identity_comparisons"][0]
    assert conflict["rule"] == "conflicting_stable_identifier"
    assert conflict["conflicting_identifiers"] == ["doi:10.1000/a!=10.1000/b"]
    assert validate_result_lineage_document(document) == []


def test_shared_identifier_conflicting_title_and_year_uses_existing_merge_rules() -> None:
    first = _paper(
        "Original", authors=["Alice"], year=2022, doi="10.1000/shared"
    )
    second = _paper(
        "Longer conflicting title",
        source="pubmed",
        authors=["Alice"],
        year=2024,
        doi="10.1000/shared",
    )
    output, document = _lineage([first, second])

    assert len(output) == 1
    assert output[0].title == "Longer conflicting title"
    assert output[0].year == 2022
    assert _decision(document, 0, "title")["status"] == "conflict_resolved"
    assert _decision(document, 0, "year")["status"] == "conflict_resolved"
    assert _decision(document, 0, "year")["selection_rule"] == (
        "first_non_null_then_first_seen"
    )


def test_null_empty_and_no_contribution_are_not_silently_collapsed() -> None:
    first = _paper("Null fields", venue=None, doi="10.1000/null")
    second = _paper(
        "Null fields", source="arxiv", venue="", doi="10.1000/null"
    )
    _, document = _lineage([first, second])
    venue = _decision(document, 0, "venue")

    assert [item["state"] for item in venue["candidates"]] == ["null", "empty"]
    assert venue["status"] == "no_contribution"
    assert venue["selected_record_refs"] == []
    assert {item["reason"] for item in venue["rejected"]} == {
        "null_not_selected",
        "empty_not_selected",
    }


def test_year_selection_distinguishes_zero_from_null_like_existing_merge() -> None:
    first = _paper("Year edge", year=0, doi="10.1000/year")
    second = _paper(
        "Year edge", source="pubmed", year=2024, doi="10.1000/year"
    )
    output, document = _lineage([first, second])

    assert output[0].year == 0
    year = _decision(document, 0, "year")
    assert year["selected_value"] == 0
    assert year["selected_record_refs"] == [
        document["source_records"][0]["record_ref"]
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        ("fabricated_field", "title"),
        ("cross_cluster_reference", "title"),
        ("wrong_source_reference", "abstract"),
        ("unregistered_transform", "identifiers.doi"),
    ],
)
def test_lineage_tampering_is_rejected_with_field_context(
    mutation: str, expected_field: str
) -> None:
    _, original = _lineage(
        [
            _paper("First", doi="10.1000/first", abstract="first abstract"),
            _paper("Second", source="arxiv", doi="10.1000/second"),
        ]
    )
    document = copy.deepcopy(original)
    if mutation == "fabricated_field":
        _decision(document, 0, "title")["selected_value"] = "fabricated"
    elif mutation == "cross_cluster_reference":
        _decision(document, 0, "title")["selected_record_refs"] = [
            document["results"][1]["contributing_record_refs"][0]
        ]
    elif mutation == "wrong_source_reference":
        _decision(document, 0, "abstract")["candidates"][0]["record_ref"] = (
            "record:unregistered:0000"
        )
    else:
        _decision(document, 0, "identifiers.doi")["candidates"][0][
            "normalization_steps"
        ] = ["unknown_transform_v999"]

    violations = validate_result_lineage_document(document)

    assert len(violations) == 1
    assert violations[0]["invariant"] == (
        "exact_reconstruction_from_registered_sources"
    )
    assert violations[0]["field"] == expected_field


def test_source_record_hash_tampering_is_rejected_before_reconstruction() -> None:
    _, document = _lineage([_paper("Hash protected", doi="10.1000/hash")])
    document["source_records"][0]["paper"]["title"] = "mutated"

    violations = validate_result_lineage_document(document)

    assert violations[0]["invariant"] == "source_record_hash_matches"
    assert violations[0]["first_difference_path"].endswith(
        ".source_record_sha256"
    )


def test_final_subset_keeps_candidate_lineage_and_reconstructs_selected_fields() -> None:
    papers = [
        _paper("First", doi="10.1000/first"),
        _paper("Second", source="arxiv", doi="10.1000/second"),
    ]
    output, document = _lineage(papers)
    restricted = restrict_result_lineage_document(document, [output[1]])

    assert len(restricted["results"]) == 2
    assert restricted["final_result_order"] == [
        restricted["results"][1]["result_identity"]
    ]
    assert validate_result_lineage_document(restricted) == []


def test_partial_source_failure_is_preserved_without_polluting_healthy_records() -> None:
    terminals = [
        {
            "source": "openalex",
            "status": "success",
            "reason": None,
            "contributed_record_count": 1,
        },
        {
            "source": "pubmed",
            "status": "partial_completion",
            "reason": "one_page_failed",
            "contributed_record_count": 1,
        },
    ]
    papers = [
        _paper("Shared", doi="10.1000/shared"),
        _paper("Shared", source="pubmed", doi="10.1000/shared", pubmed_id="1"),
    ]
    output, document = _lineage(papers, terminals)

    assert output == deduplicate_papers(papers)
    assert [item["status"] for item in document["source_terminals"]] == [
        "success",
        "partial_completion",
    ]
    assert validate_result_lineage_document(document) == []


class _FixtureRetriever:
    def __init__(self) -> None:
        self.lock = RLock()

    def __call__(
        self, query: str, limit_per_source: int = 20, sources: list[str] | None = None
    ) -> RetrievalOutput:
        requested = list(sources or ["openalex", "arxiv"])
        papers = [
            _paper("Observable", source="openalex", doi="10.1000/observable"),
            _paper(
                "Observable longer",
                source="arxiv",
                doi="10.1000/observable",
                arxiv_id="2401.1",
            ),
        ]
        stats = [
            SourceStats(
                source=source,
                terminal_status="success",
                returned_count=sum(source in paper.sources for paper in papers),
                diagnostic_papers=[paper for paper in papers if source in paper.sources],
            )
            for source in requested
        ]
        return RetrievalOutput(
            query=query,
            requested_sources=requested,
            raw_count=len(papers),
            deduplicated_count=1,
            papers=papers,
            source_stats=stats,
        )


def _semantic_output(value: Any) -> dict[str, Any]:
    payload = value.model_dump(mode="json")
    payload["latency_seconds"] = 0.0
    payload["stage_latencies"] = {}
    payload["budget_status"]["elapsed_seconds"] = 0.0
    return payload


def _semantic_events(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(values)
    for item in result:
        item.get("payload", {}).pop("latency_seconds", None)
        status = item.get("payload", {}).get("budget_status")
        if isinstance(status, dict):
            status.pop("elapsed_seconds", None)
    return result


def test_search_service_observation_does_not_change_results_order_or_events() -> None:
    service = SearchService(retriever=_FixtureRetriever(), max_workers=1)
    baseline_events: list[dict[str, Any]] = []
    lineage_events: list[dict[str, Any]] = []
    captured: list[dict[str, Any]] = []
    kwargs = {
        "query": "offline observational lineage",
        "top_k": 2,
        "sources_override": ["openalex", "arxiv"],
        "enable_synthesis": False,
        "enable_llm_query_understanding": False,
        "enable_llm_judgement": False,
    }
    baseline = service.run_search(
        **kwargs,
        event_callback=lambda event, payload: baseline_events.append(
            {"event": event, "payload": payload}
        ),
    )
    observed = service.run_search(
        **kwargs,
        event_callback=lambda event, payload: lineage_events.append(
            {"event": event, "payload": payload}
        ),
        result_lineage_callback=captured.append,
    )

    assert _semantic_output(baseline) == _semantic_output(observed)
    assert _semantic_events(baseline_events) == _semantic_events(lineage_events)
    assert len(captured) == 1
    assert validate_result_lineage_document(captured[0]) == []
    assert captured[0]["final_result_order"] == [
        item["result_identity"] for item in captured[0]["results"]
    ]


@pytest.mark.result_lineage_regression
def test_repository_gate_passes_and_is_byte_deterministic(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL, repository_root=ROOT)
    first = run_result_lineage_gate(
        protocol, repository_root=ROOT, snapshot_root=tmp_path / "snapshots"
    )
    second = run_result_lineage_gate(
        protocol, repository_root=ROOT, snapshot_root=tmp_path / "snapshots"
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(first_path, first)
    write_json(second_path, second)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["status"] == "passed"
    assert first["source_record_count"] == 8
    assert first["accepted_merge_count"] == 3
    assert first["observational_equivalence"]["equal"] is True
    assert first["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "quality_metric_count": 0,
        "controlled_fault": None,
    }


def test_controlled_field_injection_and_cli_exit_codes() -> None:
    protocol = load_protocol(PROTOCOL, repository_root=ROOT)
    report = run_result_lineage_gate(
        protocol, repository_root=ROOT, controlled_fault="field_injection"
    )
    assert report["exit_code"] == EXIT_LINEAGE_VIOLATION
    assert report["violations"][0]["field"] == "title"
    assert lineage_cli.main([]) == 4
    assert (
        lineage_cli.main(
            [
                "--repository-root",
                str(ROOT),
                "--protocol",
                str(PROTOCOL),
                "check",
                "--fault",
                "field_injection",
            ]
        )
        == EXIT_LINEAGE_VIOLATION
    )


def test_frozen_record160_and_full1000_are_not_eligible() -> None:
    protocol = load_protocol(PROTOCOL, repository_root=ROOT)
    report = audit_frozen_baseline_eligibility(protocol, repository_root=ROOT)

    assert report["exit_code"] == EXIT_NOT_ELIGIBLE
    assert {item["profile_id"] for item in report["profiles"]} == {
        "autoscholar_full1000_frozen_baseline",
        "autoscholar_record160_analysis_input",
    }
    assert {item["reason"] for item in report["profiles"]} == {
        "field_level_candidate_and_merge_lineage_unavailable"
    }


def test_lineage_artifact_is_commit_store_and_run_manifest_registered(
    tmp_path: Path,
) -> None:
    store = BenchmarkRunCommitStore(tmp_path / "run")
    store.initialize(
        run_id="lineage-run",
        expected_query_ids=["opaque-query-1"],
        config={"schema_version": "test"},
        dataset_report={"record_count": 1},
    )
    store.commit_record({"case_id": "opaque-query-1", "status": "succeeded"})
    payload = stable_json_bytes(
        {"contract": RESULT_LINEAGE_CONTRACT, "query_identity": QUERY_IDENTITY}
    )
    state = store.commit_completion({"result_lineage.jsonl": payload})
    public = store.public_artifacts(state)
    lineage_path = tmp_path / "result_lineage.jsonl"
    lineage_path.write_bytes(public["result_lineage.jsonl"])
    identity = output_identity(run_manifest_output_spec(), tmp_path)

    assert identity.role == RESULT_LINEAGE_CONTRACT
    assert identity.record_count == 1
    assert identity.size_bytes == len(payload)


def test_protocol_drift_is_not_eligible(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["fixture"]["sha256"] = "0" * 64
    path = tmp_path / "protocol.json"
    path.write_bytes(stable_json_bytes(protocol))

    with pytest.raises(ResultLineageNotEligible, match="fixture_hash_mismatch"):
        load_protocol(path, repository_root=ROOT)
