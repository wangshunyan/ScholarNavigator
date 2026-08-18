from __future__ import annotations

import pytest

from scholar_agent.core.evaluation_schemas import (
    DEDUPLICATED_GOLD_METRIC_VERSION,
    LEGACY_GOLD_METRIC_VERSION,
    EvalGoldPaper,
    EvalMetricSet,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.evaluation.metrics import (
    candidate_count_metrics,
    canonical_paper_id,
    error_rate_metrics,
    evaluate_ranking,
    f1_at_k,
    gold_deduplication_audit,
    matched_paper_ids,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def make_paper(
    title: str,
    *,
    year: int | None = 2024,
    doi: str | None = None,
    arxiv_id: str | None = None,
    openalex_id: str | None = None,
    semantic_scholar_id: str | None = None,
    s2orc_corpus_id: str | None = None,
    pubmed_id: str | None = None,
) -> Paper:
    return Paper(
        title=title,
        authors=["Alice"],
        year=year,
        venue="ACL",
        abstract="A test paper.",
        identifiers=PaperIdentifiers(
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_id=openalex_id,
            semantic_scholar_id=semantic_scholar_id,
            s2orc_corpus_id=s2orc_corpus_id,
            pubmed_id=pubmed_id,
        ),
        sources=["fixture"],
    )


def test_canonical_paper_id_priority_and_normalization() -> None:
    assert (
        canonical_paper_id(
            doi="https://doi.org/10.1000/ABC",
            arxiv_id="2401.00001v2",
            openalex_id="https://openalex.org/W123",
        )
        == "doi:10.1000/abc"
    )
    assert (
        canonical_paper_id(make_paper("arXiv Paper", arxiv_id="arXiv:2401.00001v3"))
        == "arxiv:2401.00001"
    )
    assert (
        canonical_paper_id(
            make_paper("OpenAlex Paper", openalex_id="https://openalex.org/W123")
        )
        == "openalex:w123"
    )
    assert (
        canonical_paper_id(
            make_paper("Semantic Paper", semantic_scholar_id="CorpusId:12345")
        )
        == "s2:12345"
    )
    assert canonical_paper_id(make_paper("PubMed Paper", pubmed_id="PMID:987")) == (
        "pubmed:987"
    )
    assert (
        canonical_paper_id(title="A {LaTeX} Study!", year=2024)
        == "title_year:a latex study:2024"
    )


def test_recall_precision_and_mrr() -> None:
    gold = [
        EvalGoldPaper(doi="10.123/a"),
        EvalGoldPaper(doi="10.123/b"),
        EvalGoldPaper(doi="10.123/c"),
    ]
    ranked = [
        make_paper("Not Relevant", doi="10.999/x"),
        make_paper("Relevant B", doi="10.123/b"),
        make_paper("Relevant C", doi="10.123/c"),
    ]

    assert recall_at_k(ranked, gold, 2) == pytest.approx(1 / 3)
    assert precision_at_k(ranked, gold, 2) == pytest.approx(0.5)
    assert mrr(ranked, gold) == pytest.approx(0.5)
    assert f1_at_k(ranked, gold, 2) == pytest.approx(0.4)


def test_f1_zero_and_default_k_values() -> None:
    gold = [EvalGoldPaper(doi="10.123/gold")]
    ranked = [make_paper("Other", doi="10.123/other")]

    assert f1_at_k(ranked, gold, 5) == 0.0
    metrics = evaluate_ranking(ranked, gold)
    assert metrics.f1_at_k == {5: 0.0, 10: 0.0, 20: 0.0}

    matched = evaluate_ranking(
        [make_paper("Gold", doi="10.123/gold")],
        gold,
    )
    assert matched.f1_at_k[5] == pytest.approx(1 / 3)
    assert matched.f1_at_k[10] == pytest.approx(2 / 11)
    assert matched.f1_at_k[20] == pytest.approx(2 / 21)


def test_conflicting_identifiers_do_not_match_even_with_shared_other_ids() -> None:
    gold = [
        EvalGoldPaper(
            doi="10.123/different",
            openalex_id="openalex:W123",
            pubmed_id="pmid:987",
        ),
        EvalGoldPaper(doi="10.123/duplicate-gold", openalex_id="W123"),
    ]
    ranked = [
        make_paper(
            "Identifier Match",
            doi="10.123/predicted",
            openalex_id="https://openalex.org/W123",
            pubmed_id="pubmed:987",
        ),
        make_paper("Duplicate Prediction", openalex_id="W123"),
    ]

    assert recall_at_k(ranked, gold, 2) == pytest.approx(0.0)
    assert matched_paper_ids(ranked, gold) == []


def test_semantic_scholar_prefixes_match() -> None:
    gold = [EvalGoldPaper(semantic_scholar_id="s2:ABCDEF")]
    ranked = [make_paper("S2", semantic_scholar_id="CorpusId:abcdef")]

    assert recall_at_k(ranked, gold, 1) == pytest.approx(1.0)


def test_s2orc_matches_exact_id_and_never_infers_from_title() -> None:
    gold = [
        EvalGoldPaper(
            title="Shared Title",
            year=2024,
            s2orc_corpus_id=123,
        )
    ]
    exact = make_paper("Different title", s2orc_corpus_id="CorpusId:123")
    title_only = make_paper("Shared Title", year=2024)

    assert canonical_paper_id(gold[0]) == "s2orc:123"
    assert matched_paper_ids([exact], gold) == ["s2orc:123"]
    assert matched_paper_ids([title_only], gold) == []


def test_s2orc_conflict_prevents_other_shared_identifier_match() -> None:
    gold = [EvalGoldPaper(doi="10.123/shared", s2orc_corpus_id="123")]
    ranked = [
        make_paper(
            "Conflicting Corpus ID",
            doi="10.123/shared",
            s2orc_corpus_id="456",
        )
    ]

    assert matched_paper_ids(ranked, gold) == []


def test_s2orc_gold_can_match_another_shared_stable_identifier() -> None:
    gold = [EvalGoldPaper(doi="10.123/shared", s2orc_corpus_id="123")]
    ranked = [make_paper("DOI match", doi="https://doi.org/10.123/shared")]

    assert matched_paper_ids(ranked, gold) == ["doi:10.123/shared"]


@pytest.mark.parametrize(
    ("predicted", "gold", "expected_id"),
    [
        (
            make_paper("OpenAlex", openalex_id="https://openalex.org/W2468"),
            EvalGoldPaper(openalex_id="openalex:W2468"),
            "openalex:w2468",
        ),
        (
            make_paper(
                "PubMed",
                pubmed_id="https://pubmed.ncbi.nlm.nih.gov/13579/",
            ),
            EvalGoldPaper(pubmed_id="pmid:13579"),
            "pubmed:13579",
        ),
    ],
)
def test_openalex_and_pubmed_formats_match(
    predicted: Paper,
    gold: EvalGoldPaper,
    expected_id: str,
) -> None:
    assert matched_paper_ids([predicted], [gold]) == [expected_id]


def test_duplicate_prediction_and_duplicate_gold_count_only_once() -> None:
    ranked = [
        make_paper("First", doi="10.123/same"),
        make_paper("Duplicate", doi="doi:10.123/same"),
    ]
    gold = [
        EvalGoldPaper(doi="10.123/same"),
        EvalGoldPaper(doi="https://doi.org/10.123/same"),
    ]

    assert len(matched_paper_ids(ranked, gold)) == 1
    assert precision_at_k(ranked, gold, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, gold, 2) == pytest.approx(1.0)
    assert recall_at_k(
        ranked,
        gold,
        2,
        metric_version=LEGACY_GOLD_METRIC_VERSION,
    ) == pytest.approx(0.5)


def test_gold_deduplication_merges_cross_source_identity_and_preserves_union() -> None:
    gold = [
        EvalGoldPaper(doi="10.48550/arXiv.2401.00001"),
        EvalGoldPaper(arxiv_id="2401.00001v2", doi="10.48550/arxiv.2401.00001"),
    ]
    ranked = [make_paper("arXiv", arxiv_id="arXiv:2401.00001v3")]

    audit = gold_deduplication_audit(gold)

    assert audit["legacy_evaluable_gold_count"] == 2
    assert audit["deduplicated_evaluable_gold_count"] == 1
    assert audit["duplicate_relation_count"] == 1
    assert audit["duplicate_relations"][0]["rule"] == "shared_stable_identifier"
    assert recall_at_k(ranked, gold, 1) == 1.0


def test_gold_deduplication_keeps_same_type_conflicts_separate() -> None:
    gold = [
        EvalGoldPaper(doi="10.123/shared", arxiv_id="2401.00001"),
        EvalGoldPaper(doi="10.123/shared", arxiv_id="2401.00002"),
    ]

    audit = gold_deduplication_audit(gold)

    assert audit["deduplicated_evaluable_gold_count"] == 2
    assert audit["duplicate_relation_count"] == 0


def test_gold_deduplication_connects_non_conflicting_identifier_chains() -> None:
    gold = [
        EvalGoldPaper(doi="10.123/shared"),
        EvalGoldPaper(arxiv_id="2401.00001"),
        EvalGoldPaper(doi="10.123/shared", arxiv_id="2401.00001"),
    ]

    forward = gold_deduplication_audit(gold)
    reverse = gold_deduplication_audit(list(reversed(gold)))

    assert forward["deduplicated_evaluable_gold_count"] == 1
    assert reverse["deduplicated_evaluable_gold_count"] == 1
    assert forward["duplicate_relation_count"] == 2
    assert reverse["duplicate_relation_count"] == 2


def test_deduplicated_metrics_are_input_order_independent_and_handle_empty_results() -> None:
    gold = [
        EvalGoldPaper(doi="10.123/a"),
        EvalGoldPaper(doi="https://doi.org/10.123/a"),
        EvalGoldPaper(doi="10.123/b"),
    ]
    ranked = [make_paper("A", doi="10.123/a")]

    forward = evaluate_ranking(ranked, gold, [20])
    reverse = evaluate_ranking(ranked, list(reversed(gold)), [20])

    assert forward == reverse
    assert forward.metric_version == DEDUPLICATED_GOLD_METRIC_VERSION
    assert forward.recall_at_k[20] == 0.5
    assert evaluate_ranking([], gold, [20]).recall_at_k[20] == 0.0


def test_legacy_metric_json_without_version_remains_readable() -> None:
    legacy = EvalMetricSet.model_validate({"recall_at_k": {"20": 0.5}})
    current = evaluate_ranking([], [EvalGoldPaper(doi="10.123/a")], [20])

    assert legacy.metric_version == LEGACY_GOLD_METRIC_VERSION
    assert legacy.recall_at_k[20] == 0.5
    assert current.metric_version == DEDUPLICATED_GOLD_METRIC_VERSION


def test_binary_ndcg() -> None:
    gold = [
        EvalGoldPaper(doi="10.123/a"),
        EvalGoldPaper(doi="10.123/b"),
    ]
    ranked = [
        make_paper("Relevant A", doi="10.123/a"),
        make_paper("Irrelevant", doi="10.999/x"),
        make_paper("Relevant B", doi="10.123/b"),
    ]

    expected = (1.0 + 0.5) / (1.0 + (1.0 / 1.584962500721156))
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(expected)
    assert ndcg_at_k(ranked, gold, 5) > 0
    assert ndcg_at_k(ranked, gold, 10) > 0
    assert ndcg_at_k(ranked, gold, 20) > 0


def test_graded_ndcg() -> None:
    gold = [
        EvalGoldPaper(doi="10.123/a", relevance_grade=3),
        EvalGoldPaper(doi="10.123/b", relevance_grade=2),
        EvalGoldPaper(doi="10.123/c", relevance_grade=1),
    ]
    ideal = [
        make_paper("Grade 3", doi="10.123/a"),
        make_paper("Grade 2", doi="10.123/b"),
        make_paper("Grade 1", doi="10.123/c"),
    ]
    non_ideal = [
        make_paper("Grade 2", doi="10.123/b"),
        make_paper("Grade 3", doi="10.123/a"),
        make_paper("Grade 1", doi="10.123/c"),
    ]

    assert ndcg_at_k(ideal, gold, 3) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(non_ideal, gold, 3) < 1


def test_empty_inputs_do_not_crash() -> None:
    assert canonical_paper_id({}) is None
    assert recall_at_k([], [], 10) == 0.0
    assert precision_at_k([], [], 10) == 0.0
    assert mrr([], []) == 0.0
    assert ndcg_at_k([], [], 10) == 0.0
    assert ndcg_at_k([make_paper("No Gold", doi="10.1/x")], [], 10) == 0.0


def test_candidate_count_metrics() -> None:
    metrics = candidate_count_metrics(
        10,
        7,
        ranked_count=5,
        source_stats=[
            {"source": "openalex", "returned_count": 3},
            {"source": "arxiv", "returned_count": 4},
            {"source": "openalex", "returned_count": 3},
        ],
    )

    assert metrics["raw_count"] == 10
    assert metrics["deduplicated_count"] == 7
    assert metrics["ranked_count"] == 5
    assert metrics["duplicate_count"] == 3
    assert metrics["duplicate_ratio"] == pytest.approx(0.3)
    assert metrics["per_source_returned_count"] == {"openalex": 6, "arxiv": 4}


def test_error_rate_metrics() -> None:
    metrics = error_rate_metrics(
        source_stats=[
            {"source": "openalex", "error_message": "HTTP 503"},
            {"source": "arxiv", "error_message": None},
            {"source": "openalex", "error_message": "timeout"},
        ],
        warnings=["HTTP 503", ""],
        failed_case_count=1,
        total_case_count=4,
        warning_case_count=2,
    )

    assert metrics["source_call_count"] == 3
    assert metrics["source_error_count"] == 2
    assert metrics["source_error_rate"] == pytest.approx(2 / 3)
    assert metrics["warning_count"] == 1
    assert metrics["query_warning_rate"] == pytest.approx(0.5)
    assert metrics["failed_case_rate"] == pytest.approx(0.25)
