from __future__ import annotations

from pathlib import Path

from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.fixture_loader import (
    build_fixture_reference_fetcher,
    build_fixture_retriever,
    load_eval_queries,
    load_evaluation_fixtures,
    load_reference_outputs,
    load_retrieval_outputs,
)


SAMPLE_DIR = Path("datasets/eval_fixtures/sample")


def test_load_eval_queries_from_sample_jsonl() -> None:
    queries = load_eval_queries(SAMPLE_DIR / "search_cases.jsonl")

    assert len(queries) == 1
    assert queries[0].query_id == "sample_llm_rerank"
    assert queries[0].gold_papers
    assert queries[0].top_k_values == [5, 10, 20]


def test_load_retrieval_outputs_and_build_fake_retriever() -> None:
    outputs = load_retrieval_outputs(SAMPLE_DIR / "retrieval_outputs.json")
    retriever = build_fixture_retriever(outputs)

    output = retriever(
        "latest LLM reranking methods for scientific literature retrieval",
        sources=["openalex", "arxiv"],
    )
    missing = retriever("missing local fixture query")

    assert output.papers[0].identifiers.doi == "10.123/baseline"
    assert output.requested_sources == ["openalex", "arxiv"]
    assert missing.papers == []
    assert missing.warnings == ["fixture_missing_retrieval:missing local fixture query"]
    assert missing.source_stats[0].error_message


def test_load_reference_outputs_and_build_fake_fetcher() -> None:
    references = load_reference_outputs(SAMPLE_DIR / "reference_outputs.json")
    fetcher = build_fixture_reference_fetcher(references)
    seed = Paper(
        title="Seed",
        authors=[],
        year=2025,
        abstract="",
        identifiers=PaperIdentifiers(doi="10.123/baseline"),
    )

    fetched = fetcher(seed, limit=1)

    assert len(fetched) == 1
    assert fetched[0].identifiers.doi == "10.123/refchain"


def test_load_evaluation_fixtures_bundle() -> None:
    fixtures = load_evaluation_fixtures(SAMPLE_DIR)

    assert len(fixtures.eval_queries) == 1
    assert fixtures.retrieval_outputs
    assert fixtures.reference_outputs
