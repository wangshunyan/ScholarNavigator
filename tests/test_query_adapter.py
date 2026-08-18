from __future__ import annotations

import re

from scholar_agent.core.search_schemas import QueryConstraint
from scholar_agent.retrieval.query_adapter import (
    MAX_ADAPTED_QUERIES_PER_SOURCE,
    MAX_ARXIV_QUERY_LENGTH,
    MAX_OPENALEX_QUERY_LENGTH,
    MIN_COMPACT_RETENTION_RATIO,
    adapt_queries_for_source,
    adapt_query_for_source,
)


def test_safe_original_preserves_rare_dataset_name() -> None:
    adapted = adapt_query_for_source(
        "Find anomaly detection studies using RareTelemetryCorpus",
        "openalex",
    )

    assert adapted.strategy == "safe_original"
    assert "RareTelemetryCorpus" in adapted.query
    assert "RareTelemetryCorpus" in adapted.protected_terms


def test_safe_original_preserves_method_abbreviation_and_digits() -> None:
    adapted = adapt_query_for_source(
        "Compare ZXQ-7 and GNN2Vec for retrieval",
        "semantic_scholar",
    )

    assert "ZXQ-7" in adapted.query
    assert "GNN2Vec" in adapted.query
    assert {"ZXQ-7", "GNN2Vec"}.issubset(set(adapted.protected_terms))


def test_chinese_query_is_not_compressed_to_empty() -> None:
    queries = adapt_queries_for_source("图神经网络在药物发现中的应用", "openalex")

    assert queries
    assert queries[0].query
    assert "图神经网络" in queries[0].query


def test_mixed_language_query_retains_both_languages() -> None:
    safe = adapt_queries_for_source(
        "用 GNN2Vec 处理知识图谱 link prediction",
        "semantic_scholar",
    )[0]

    assert "GNN2Vec" in safe.query
    assert "知识图谱" in safe.query
    assert "link prediction" in safe.query


def test_compact_core_supplements_instead_of_replacing_safe_original() -> None:
    queries = adapt_queries_for_source(
        "Could you list studies about contrastive graph retrieval methods?",
        "openalex",
    )

    assert [item.strategy for item in queries] == ["safe_original", "compact_core"]
    assert queries[0].query.startswith("Could you list studies")
    assert "contrastive" in queries[1].query


def test_low_information_retention_falls_back_to_safe_original() -> None:
    query = " ".join(f"SpecializedTerm{index}" for index in range(30))

    queries = adapt_queries_for_source(
        query, "semantic_scholar", policy="hybrid"
    )

    assert len(queries) == 1
    assert queries[0].strategy == "fallback_original"
    assert "compact_query_fallback_to_safe_original" in queries[0].warnings
    assert MIN_COMPACT_RETENTION_RATIO == 0.5


def test_arxiv_hybrid_keeps_wide_query_and_only_pairs_core_terms() -> None:
    queries = adapt_queries_for_source(
        "contrastive graph retrieval with multilingual supervision",
        "arxiv",
    )

    assert queries[0].strategy == "safe_original"
    assert queries[0].query.startswith("all:")
    assert queries[1].strategy == "compact_core"
    assert " OR " in queries[1].query
    assert queries[1].query.count(" AND ") <= 2
    assert "multilingual" in queries[1].query or "supervision" in queries[1].query


def test_arxiv_safe_original_has_legal_bounded_syntax() -> None:
    adapted = adapt_query_for_source(
        'papers on graph (neural): retrieval? "C++" [survey]' * 20,
        "arxiv",
    )

    assert adapted.query.startswith("all:")
    assert len(adapted.query) <= MAX_ARXIV_QUERY_LENGTH
    assert not re.search(r"[?\[\]{}]", adapted.query)
    assert adapted.query.count('"') % 2 == 0


def test_openalex_safe_original_removes_illegal_punctuation_without_core_loss() -> None:
    adapted = adapt_query_for_source(
        "retrieval\x00 graph; ranking! multilingual 中文: ZXQ-7",
        "openalex",
    )

    assert len(adapted.query) <= MAX_OPENALEX_QUERY_LENGTH
    assert "\x00" not in adapted.query
    assert not re.search(r"[;!:]", adapted.query)
    assert "multilingual" in adapted.query
    assert "中文" in adapted.query
    assert "ZXQ-7" in adapted.query


def test_explicit_method_and_dataset_are_prioritized_in_compact_query() -> None:
    constraints = QueryConstraint(
        methods=["contrastive learning"],
        datasets=["clinical notes"],
        explicit_fields=["methods", "datasets"],
    )

    queries = adapt_queries_for_source(
        "Could you list papers about representation learning?",
        "openalex",
        constraints=constraints,
    )

    assert queries[0].strategy == "safe_original"
    assert queries[1].query.startswith("contrastive learning clinical notes")
    assert {"contrastive learning", "clinical notes"}.issubset(
        set(queries[1].protected_terms)
    )


def test_equivalent_safe_and_compact_queries_merge_without_second_request_variant() -> None:
    queries = adapt_queries_for_source(
        "graph retrieval", "openalex", policy="hybrid"
    )

    assert len(queries) == 1
    assert queries[0].equivalent_strategies == ["safe_original", "compact_core"]


def test_safe_original_policy_emits_only_safe_query() -> None:
    queries = adapt_queries_for_source(
        "Could you list papers about graph retrieval?",
        "arxiv",
        policy="safe_original",
    )

    assert len(queries) == 1
    assert queries[0].strategy == "safe_original"


def test_query_variant_count_is_centrally_bounded_for_every_source() -> None:
    for source in ("arxiv", "openalex", "semantic_scholar", "pubmed"):
        queries = adapt_queries_for_source(
            "graph neural retrieval with contrastive learning",
            source,
            max_queries=99,
        )
        assert len(queries) <= MAX_ADAPTED_QUERIES_PER_SOURCE


def test_all_adaptation_results_are_deterministic() -> None:
    query = '多语言 "Graph Retrieval" with ZXQ-7 benchmark dataset'
    constraints = QueryConstraint(
        methods=["contrastive learning"],
        datasets=["MixedCorpus2"],
        explicit_fields=["methods", "datasets"],
    )

    for source in ("arxiv", "openalex", "semantic_scholar", "pubmed"):
        first = adapt_queries_for_source(query, source, constraints=constraints)
        second = adapt_queries_for_source(query, source, constraints=constraints)
        assert first == second
