from __future__ import annotations

import pytest

from scholar_agent.agents import query_understanding as query_understanding_module
from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.core.search_schemas import QueryConstraint
from scholar_agent.prompts.loader import PromptLoadError, load_prompt


def test_chinese_long_query_generates_search_plan() -> None:
    plan = analyze_query(
        "请检索近三年 LLM reranking 在科研论文搜索中的代表性论文，重点关注 ACL 和 SIGIR。",
        current_year=2026,
    )

    assert plan.query_analysis.original_query.startswith("请检索近三年")
    assert plan.query_analysis.language == "mixed"
    assert plan.query_analysis.domain == "machine_learning"
    assert plan.query_analysis.constraints.time_range is not None
    assert plan.query_analysis.constraints.venues == ["ACL", "SIGIR"]
    assert plan.selected_sources == ["arxiv", "openalex"]
    assert 1 <= len(plan.subqueries) <= 5
    assert all(subquery.purpose for subquery in plan.subqueries)


def test_english_recent_query_detects_recent_progress() -> None:
    plan = analyze_query(
        "latest LLM reranking methods for scientific literature retrieval",
        current_year=2026,
    )

    assert plan.query_analysis.language == "en"
    assert plan.query_analysis.intent == "recent_progress"
    assert plan.query_analysis.constraints.time_range is not None
    assert plan.query_analysis.constraints.time_range.start_year == 2023
    assert plan.query_analysis.constraints.time_range.end_year == 2026


def test_since_year_parses_start_year() -> None:
    plan = analyze_query("LLM reranking papers since 2020", current_year=2026)

    time_range = plan.query_analysis.constraints.time_range
    assert time_range is not None
    assert time_range.start_year == 2020
    assert time_range.end_year is None


def test_year_range_parses_start_and_end_year() -> None:
    plan = analyze_query("RAG retrieval papers 2021-2024", current_year=2026)

    time_range = plan.query_analysis.constraints.time_range
    assert time_range is not None
    assert time_range.start_year == 2021
    assert time_range.end_year == 2024


def test_chinese_recent_three_years_uses_current_year() -> None:
    plan = analyze_query("近三年 LLM reranking 论文", current_year=2026)

    time_range = plan.query_analysis.constraints.time_range
    assert time_range is not None
    assert time_range.start_year == 2023
    assert time_range.end_year == 2026


def test_llm_reranking_retrieval_query_selects_openalex_and_arxiv() -> None:
    plan = analyze_query("LLM reranking for retrieval", current_year=2026)

    assert plan.selected_sources == ["arxiv", "openalex"]
    assert plan.query_analysis.domain == "machine_learning"
    assert all(
        set(subquery.source_hints).issubset({"openalex", "arxiv"})
        for subquery in plan.subqueries
    )


def test_biomedical_query_selects_pubmed() -> None:
    plan = analyze_query(
        "recent clinical gene therapy studies in PubMed",
        current_year=2026,
    )

    assert plan.query_analysis.domain == "biomedical"
    assert plan.selected_sources == ["pubmed", "openalex"]
    assert "pubmed_not_implemented" not in plan.warnings


def test_fast_and_high_recall_profiles_differ() -> None:
    fast = analyze_query(
        "latest LLM reranking benchmark dataset papers",
        run_profile="fast",
        current_year=2026,
    )
    high_recall = analyze_query(
        "latest LLM reranking benchmark dataset papers",
        run_profile="high_recall",
        current_year=2026,
    )

    assert fast.limit_per_source < high_recall.limit_per_source
    assert len(fast.subqueries) < len(high_recall.subqueries)


def test_empty_query_raises_value_error() -> None:
    with pytest.raises(ValueError):
        analyze_query("   ")


def test_subqueries_are_deduplicated() -> None:
    plan = analyze_query("LLM LLM reranking reranking", current_year=2026)
    queries = [subquery.query.casefold() for subquery in plan.subqueries]

    assert len(queries) == len(set(queries))


def test_unknown_benchmark_name_gets_generic_dimension_subqueries() -> None:
    plan = analyze_query(
        "OrionEval benchmark datasets for scientific literature search agents",
        run_profile="high_recall",
        current_year=2026,
    )
    queries = [subquery.query for subquery in plan.subqueries]

    assert queries[0].startswith("OrionEval benchmark datasets")
    assert any("dataset evaluation" in query for query in queries)
    assert any(
        all(term in query.casefold() for term in ("benchmark", "dataset", "evaluation"))
        for query in queries
    )
    assert len(queries) == len({query.casefold() for query in queries})


def test_evaluation_query_uses_only_generic_intent_and_dimensions() -> None:
    plan = analyze_query(
        "retrieval augmented generation evaluation benchmark papers",
        run_profile="high_recall",
        current_year=2026,
    )
    queries = [subquery.query for subquery in plan.subqueries]

    assert queries[0] == "retrieval augmented generation evaluation benchmark papers"
    assert queries[1] == "retrieval augmented generation evaluation benchmark dataset"
    assert any("dataset evaluation" in query for query in queries)
    assert all("automated evaluation framework" not in query for query in queries)


def test_fast_profile_bounds_generic_query_expansion() -> None:
    plan = analyze_query(
        "RAG evaluation benchmark papers",
        run_profile="fast",
        current_year=2026,
    )
    queries = [subquery.query for subquery in plan.subqueries]

    assert queries == [
        "RAG evaluation benchmark papers",
        "RAG evaluation benchmark dataset",
    ]


def test_method_query_gets_generic_method_and_domain_subqueries() -> None:
    plan = analyze_query(
        "neural ranking methods for academic search",
        run_profile="high_recall",
        current_year=2026,
    )
    queries = [subquery.query for subquery in plan.subqueries]

    assert queries[0] == "neural ranking methods for academic search"
    assert queries[1] == "neural ranking methods academic method"
    assert any(subquery.purpose == "method_dimension" for subquery in plan.subqueries)
    assert all("explicit semantic" not in query for query in queries)


def test_unseen_composite_query_decomposes_method_dataset_and_task() -> None:
    plan = analyze_query(
        "compare adaptive evidence retrieval for clinical question answering",
        run_profile="high_recall",
        current_year=2026,
        explicit_constraints=QueryConstraint(
            methods=["contrastive reranking"],
            datasets=["NovaSet"],
            domains=["biomedical"],
            must_include_terms=["clinical question answering"],
            paper_types=["comparison"],
            explicit_fields=[
                "methods",
                "datasets",
                "domains",
                "must_include_terms",
                "paper_types",
            ],
        ),
    )

    purposes = [subquery.purpose for subquery in plan.subqueries]
    assert purposes == [
        "original_query",
        "constraint_expansion",
        "method_comparison_expansion",
        "method_dimension",
        "dataset_dimension",
    ]
    assert "contrastive reranking" in plan.subqueries[1].query
    assert "NovaSet" in plan.subqueries[1].query
    assert "comparison" in plan.subqueries[2].query


def test_query_expansion_never_injects_unseen_target_title() -> None:
    plan = analyze_query(
        "QuasarEval benchmark for adaptive evidence retrieval",
        run_profile="high_recall",
        current_year=2026,
    )

    assert all(
        "Canonical Answer Paper" not in subquery.query
        for subquery in plan.subqueries
    )
    assert any("QuasarEval" in subquery.query for subquery in plan.subqueries)


def test_different_domains_share_generic_expansion_rules() -> None:
    computer_science = analyze_query(
        "database indexing methods for transaction processing",
        run_profile="high_recall",
        current_year=2026,
    )
    biomedical = analyze_query(
        "protein indexing methods for clinical cohorts",
        run_profile="high_recall",
        current_year=2026,
    )

    assert computer_science.query_analysis.domain == "computer_science"
    assert biomedical.query_analysis.domain == "biomedical"
    assert "method_dimension" in {
        subquery.purpose for subquery in computer_science.subqueries
    }
    assert "method_dimension" in {
        subquery.purpose for subquery in biomedical.subqueries
    }


def test_generic_query_expansion_is_stable() -> None:
    first = analyze_query(
        "QuasarEval benchmark for adaptive evidence retrieval",
        run_profile="high_recall",
        current_year=2026,
    )
    second = analyze_query(
        "QuasarEval benchmark for adaptive evidence retrieval",
        run_profile="high_recall",
        current_year=2026,
    )

    assert first.model_dump() == second.model_dump()


def test_llm_json_can_generate_search_plan() -> None:
    client = FakeLLMClient(
        {
            "language": "en",
            "intent": "recent_progress",
            "domain": "machine_learning",
            "constraints": {
                "time_range": {
                    "start_year": 2024,
                    "end_year": 2026,
                    "label": "recent",
                },
                "methods": ["reranking"],
                "must_include_terms": ["LLM", "scientific retrieval"],
            },
            "selected_sources": ["openalex", "arxiv"],
            "subqueries": [
                {
                    "query": "LLM reranking scientific literature retrieval",
                    "source_hints": ["openalex", "arxiv"],
                    "priority": 1,
                    "purpose": "llm_keyword_query",
                }
            ],
            "warnings": ["llm_note"],
        }
    )

    plan = analyze_query(
        "latest LLM reranking methods for scientific literature retrieval",
        current_year=2026,
        use_llm=True,
        llm_client=client,
    )

    assert client.calls == 1
    assert plan.query_analysis.intent == "recent_progress"
    assert plan.query_analysis.constraints.time_range is not None
    assert plan.query_analysis.constraints.time_range.start_year == 2024
    assert plan.selected_sources == ["openalex", "arxiv"]
    assert plan.subqueries[0].query == "LLM reranking scientific literature retrieval"
    assert plan.subqueries[0].purpose == "llm_keyword_query"
    assert "llm_query_understanding_used" in plan.warnings
    assert "llm_note" in plan.warnings


def test_llm_messages_use_packaged_markdown_prompt() -> None:
    query = "latest LLM reranking methods"
    client = FakeLLMClient(
        {
            "language": "en",
            "intent": "recent_progress",
            "domain": "machine_learning",
            "selected_sources": ["arxiv"],
            "subqueries": ["LLM reranking retrieval"],
        }
    )

    analyze_query(query, current_year=2026, use_llm=True, llm_client=client)

    loaded = load_prompt("query_understanding")
    assert client.messages[0][0]["content"] == loaded.system_text
    assert '"query": "latest LLM reranking methods"' in client.messages[0][1]["content"]


@pytest.mark.parametrize(
    ("env_value", "expected_timeout"),
    [
        (None, 20.0),
        ("7.5", 7.5),
        ("invalid", 20.0),
        ("0", 20.0),
        ("-3", 20.0),
    ],
)
def test_llm_query_understanding_passes_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected_timeout: float,
) -> None:
    if env_value is None:
        monkeypatch.delenv(
            "SCHOLAR_AGENT_LLM_QUERY_UNDERSTANDING_TIMEOUT_SECONDS",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "SCHOLAR_AGENT_LLM_QUERY_UNDERSTANDING_TIMEOUT_SECONDS",
            env_value,
        )
    client = FakeLLMClient(
        {
            "language": "en",
            "intent": "recent_progress",
            "domain": "machine_learning",
            "selected_sources": ["arxiv"],
            "subqueries": ["LLM reranking retrieval"],
        }
    )

    analyze_query(
        "latest LLM reranking methods",
        current_year=2026,
        use_llm=True,
        llm_client=client,
    )

    assert client.timeouts == [expected_timeout]


@pytest.mark.parametrize(
    ("raw_intent", "expected_intent"),
    [
        ("recent methods", "recent_progress"),
        ("find recent methods", "recent_progress"),
        ("find papers", "paper_finding"),
    ],
)
def test_llm_intent_aliases_are_normalized(
    raw_intent: str,
    expected_intent: str,
) -> None:
    client = FakeLLMClient(
        {
            "language": "en",
            "intent": raw_intent,
            "domain": "machine learning",
            "selected_sources": ["openalex", "arxiv"],
            "subqueries": ["LLM reranking retrieval"],
        }
    )

    plan = analyze_query(
        "latest LLM reranking methods",
        current_year=2026,
        use_llm=True,
        llm_client=client,
    )

    assert plan.query_analysis.intent == expected_intent
    assert not any(warning.startswith("llm_invalid_intent") for warning in plan.warnings)


@pytest.mark.parametrize(
    ("raw_domain", "expected_domain"),
    [
        ("computer science / information retrieval", "computer_science"),
        ("information retrieval", "computer_science"),
        ("cs", "computer_science"),
        ("computer science", "computer_science"),
        ("ml", "machine_learning"),
        ("machine learning", "machine_learning"),
    ],
)
def test_llm_domain_aliases_are_normalized(
    raw_domain: str,
    expected_domain: str,
) -> None:
    client = FakeLLMClient(
        {
            "language": "en",
            "intent": "recent methods",
            "domain": raw_domain,
            "selected_sources": ["openalex", "arxiv"],
            "subqueries": ["LLM reranking retrieval"],
        }
    )

    plan = analyze_query(
        "latest LLM reranking methods",
        current_year=2026,
        use_llm=True,
        llm_client=client,
    )

    assert plan.query_analysis.domain == expected_domain
    assert not any(warning.startswith("llm_invalid_domain") for warning in plan.warnings)


def test_llm_disabled_falls_back_to_rules_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCHOLAR_AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(
        query_understanding_module,
        "render_messages",
        lambda *_args, **_kwargs: pytest.fail("disabled path loaded a prompt"),
    )

    plan = analyze_query(
        "latest LLM reranking methods",
        current_year=2026,
        use_llm=True,
    )

    assert plan.query_analysis.intent == "recent_progress"
    assert "llm_query_understanding_disabled" in plan.warnings


def test_prompt_load_failure_falls_back_to_rules_with_stable_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise PromptLoadError("sensitive details must not escape")

    monkeypatch.setattr(query_understanding_module, "render_messages", fail_load)
    client = FakeLLMClient({})

    plan = analyze_query(
        "latest LLM reranking methods",
        current_year=2026,
        use_llm=True,
        llm_client=client,
    )

    assert client.calls == 0
    assert plan.query_analysis.intent == "recent_progress"
    assert "llm_query_understanding_prompt_load_failed" in plan.warnings
    assert "sensitive details" not in " ".join(plan.warnings)


def test_llm_exception_falls_back_to_rules_with_warning() -> None:
    plan = analyze_query(
        "latest LLM reranking methods",
        current_year=2026,
        use_llm=True,
        llm_client=FailingLLMClient(RuntimeError("provider unavailable")),
    )

    assert plan.query_analysis.intent == "recent_progress"
    assert any(
        warning.startswith("llm_query_understanding_failed:provider unavailable")
        for warning in plan.warnings
    )


def test_invalid_llm_json_falls_back_to_rules_with_warning() -> None:
    client = FakeLLMClient(["not", "a", "json object"])  # type: ignore[arg-type]

    plan = analyze_query(
        "latest LLM reranking methods",
        current_year=2026,
        use_llm=True,
        llm_client=client,
    )

    assert plan.query_analysis.intent == "recent_progress"
    assert any(
        warning.startswith("llm_query_understanding_failed:")
        for warning in plan.warnings
    )


def test_llm_semantic_scholar_and_pubmed_sources_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-key")
    client = FakeLLMClient(
        {
            "language": "en",
            "intent": "paper_finding",
            "domain": "machine_learning",
            "selected_sources": ["semantic_scholar", "pubmed", "openalex"],
            "subqueries": [
                {
                    "query": "LLM reranking papers",
                    "source_hints": ["semantic_scholar", "arxiv"],
                    "purpose": "unsupported_source_filtering",
                }
            ],
        }
    )

    plan = analyze_query(
        "find papers about LLM reranking",
        current_year=2026,
        use_llm=True,
        llm_client=client,
    )

    assert plan.selected_sources == ["semantic_scholar", "pubmed", "openalex"]
    assert plan.subqueries[0].source_hints == ["semantic_scholar", "arxiv"]
    assert "semantic_scholar" in plan.selected_sources
    assert "pubmed" in plan.selected_sources
    assert "llm_selected_source_not_implemented:pubmed" not in plan.warnings
    assert not any(
        warning == "llm_subquery_source_not_implemented:semantic_scholar"
        for warning in plan.warnings
    )


def test_llm_semantic_scholar_is_not_a_default_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    client = FakeLLMClient(
        {
            "language": "en",
            "intent": "paper_finding",
            "domain": "machine_learning",
            "selected_sources": ["semantic_scholar", "arxiv", "openalex"],
            "subqueries": [],
        }
    )

    plan = analyze_query(
        "find papers about LLM retrieval",
        use_llm=True,
        llm_client=client,
    )

    assert plan.selected_sources == ["arxiv", "openalex"]
    assert any(
        warning.endswith("unavailable_without_key:semantic_scholar")
        for warning in plan.warnings
    )


class FakeLLMClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls = 0
        self.timeouts: list[float | None] = []
        self.messages: list[list[dict[str, str]]] = []

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        self.calls += 1
        self.timeouts.append(timeout)
        self.messages.append(messages)
        assert temperature == 0
        assert messages
        return self.response


class FailingLLMClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        raise self.error
