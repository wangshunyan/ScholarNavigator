from __future__ import annotations

import pytest

from scholar_agent.agents.query_evolution import (
    _retains_required_information,
    analyze_query_coverage,
    evolve_queries,
    filter_evolved_candidates,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    JudgementResult,
    QueryAnalysis,
    QueryConstraint,
    QueryEvolutionOptions,
    RankedPaper,
    RerankScoreBreakdown,
    SearchPlan,
    SearchSubquery,
    TimeRange,
)


def _paper(
    title: str,
    suffix: str,
    *,
    abstract: str = "",
    venue: str = "NeurIPS",
    year: int = 2023,
    sources: list[str] | None = None,
) -> Paper:
    return Paper(
        title=title,
        authors=["A. Researcher"],
        year=year,
        venue=venue,
        abstract=abstract,
        identifiers=PaperIdentifiers(doi=f"10.1000/{suffix}"),
        sources=sources or ["arxiv"],
    )


def _analysis(*, explicit: bool = True) -> QueryAnalysis:
    return QueryAnalysis(
        original_query="graph neural networks for molecular property prediction",
        language="en",
        intent="benchmark_or_dataset",
        domain="machine_learning",
        constraints=QueryConstraint(
            time_range=TimeRange(start_year=2020, end_year=2025),
            venues=["NeurIPS"],
            methods=["message passing"],
            datasets=["QM9"],
            must_include_terms=["molecular"],
            paper_types=["benchmark"],
            explicit_fields=(
                [
                    "time_range",
                    "venues",
                    "methods",
                    "datasets",
                    "must_include_terms",
                    "paper_types",
                ]
                if explicit
                else []
            ),
        ),
    )


def _plan(analysis: QueryAnalysis) -> SearchPlan:
    return SearchPlan(
        query_analysis=analysis,
        subqueries=[
            SearchSubquery(
                query=analysis.original_query,
                source_hints=["arxiv"],
                purpose="original_query",
            )
        ],
        selected_sources=["arxiv"],
        enable_query_evolution=True,
        query_evolution_policy="coverage_gap",
    )


def _judgement(
    paper: Paper,
    *,
    category: str = "partially_relevant",
    score: float = 0.7,
) -> JudgementResult:
    return JudgementResult(
        paper=paper,
        score=score,
        category=category,
        reasoning="规则判断",
        evidence=[EvidenceItem(source="title", text=paper.title, confidence=0.8)],
        matched_terms=["graph neural networks", "molecular"],
    )


def _ranked(result: JudgementResult, rank: int = 1) -> RankedPaper:
    return RankedPaper(
        rank=rank,
        paper=result.paper,
        final_score=result.score,
        category=result.category,
        score_breakdown=RerankScoreBreakdown(
            relevance_score=result.score,
            authority_score=0.5,
            timeliness_score=0.5,
            metadata_score=0.8,
            final_score=result.score,
            relevance_weight=0.65,
            authority_weight=0.08,
            timeliness_weight=0.22,
            metadata_weight=0.05,
        ),
        ranking_reason="稳定排序",
        evidence=result.evidence,
        matched_terms=result.matched_terms,
    )


def test_coverage_gap_records_each_missing_structured_dimension() -> None:
    analysis = _analysis()
    seed = _judgement(
        _paper(
            "Graph Neural Networks for Molecular Prediction",
            "seed",
            abstract="Molecular property prediction with graph models.",
        )
    )

    gap = analyze_query_coverage(analysis, [seed])

    assert gap.needs_evolution is True
    assert gap.missing_methods == ["message passing"]
    assert gap.missing_datasets == ["QM9"]
    assert gap.missing_paper_types == ["benchmark"]
    assert gap.missing_must_have_terms == []
    assert gap.venue_coverage == 1.0
    assert gap.temporal_coverage == 1.0


@pytest.mark.parametrize(
    ("constraint_update", "explicit_field", "expected_dimension"),
    [
        ({"methods": ["message passing"]}, "methods", "method"),
        ({"datasets": ["QM9"]}, "datasets", "dataset"),
        (
            {"must_include_terms": ["protein binding"]},
            "must_include_terms",
            "must_have",
        ),
        ({"paper_types": ["benchmark"]}, "paper_types", "paper_type"),
    ],
)
def test_each_structured_gap_triggers_a_named_query(
    constraint_update: dict[str, object],
    explicit_field: str,
    expected_dimension: str,
) -> None:
    analysis = QueryAnalysis(
        original_query="graph neural networks for molecular property prediction",
        language="en",
        intent="general",
        domain="machine_learning",
        constraints=QueryConstraint.model_validate(
            {
                **constraint_update,
                "explicit_fields": [explicit_field],
            }
        ),
    )
    seed = _judgement(
        _paper(
            "Graph Neural Networks for Molecular Property Prediction",
            f"gap-{expected_dimension}",
            abstract=(
                "Graph neural networks for molecular property prediction."
            ),
        )
    )

    record = evolve_queries(
        analysis,
        _plan(analysis),
        [seed],
        [_ranked(seed)],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert record.generated_queries
    assert expected_dimension in record.generated_queries[0].gap_dimensions


def test_coverage_gap_query_is_bounded_and_retains_required_information() -> None:
    analysis = _analysis()
    seed = _judgement(
        _paper(
            "A Seed Title That Must Never Become the Search Query",
            "bounded",
            abstract="Graph neural networks for molecular prediction.",
        )
    )

    record = evolve_queries(
        analysis,
        _plan(analysis),
        [seed],
        [_ranked(seed)],
        used_queries=set(),
        options=QueryEvolutionOptions(
            policy="coverage_gap",
            max_evolved_queries=9,
            max_seed_papers=9,
        ),
    )

    assert record.policy == "coverage_gap"
    assert record.seed_count == 1
    assert len(record.generated_queries) == 2
    assert all(item.generation_policy == "coverage_gap" for item in record.generated_queries)
    assert all("graph" in item.query.casefold() for item in record.generated_queries)
    assert all("molecular" in item.query.casefold() for item in record.generated_queries)
    assert all("2020-2025" in item.query for item in record.generated_queries)
    assert all("A Seed Title" not in item.query for item in record.generated_queries)


def test_question_boilerplate_does_not_replace_original_core_topic() -> None:
    analysis = QueryAnalysis(
        original_query=(
            "Can you tell me some papers about hybrid architectures in "
            "reconstruction techniques?"
        ),
        language="en",
        intent="general",
        domain="machine_learning",
        constraints=QueryConstraint(
            methods=["diffusion"],
            must_include_terms=[
                "Can",
                "you",
                "tell",
                "some",
                "hybrid",
                "architectures",
                "reconstruction",
                "techniques",
            ],
        ),
    )
    seed = _judgement(
        _paper(
            "Hybrid Architectures for Reconstruction",
            "boilerplate",
            abstract="Hybrid reconstruction techniques for inverse problems.",
        )
    )

    record = evolve_queries(
        analysis,
        _plan(analysis),
        [seed],
        [_ranked(seed)],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert record.generated_queries
    query = record.generated_queries[0].query.casefold()
    assert "hybrid" in query
    assert "architectures" in query
    assert "can you tell" not in query


def test_coverage_sufficient_skips_evolution() -> None:
    analysis = _analysis()
    seed = _judgement(
        _paper(
            "Message Passing Benchmark on QM9",
            "covered",
            abstract=(
                "Graph neural networks benchmark molecular property prediction "
                "with message passing on QM9."
            ),
        ),
        category="highly_relevant",
        score=0.9,
    )

    record = evolve_queries(
        analysis,
        _plan(analysis),
        [seed],
        [_ranked(seed)],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert record.generated_queries == []
    assert record.skipped_reasons == ["coverage_sufficient"]


def test_no_reliable_seed_skips_even_when_gap_exists() -> None:
    analysis = _analysis()
    irrelevant = _judgement(
        _paper("Unrelated Vision Paper", "irrelevant"),
        category="irrelevant",
        score=0.1,
    )

    record = evolve_queries(
        analysis,
        _plan(analysis),
        [irrelevant],
        [_ranked(irrelevant)],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert record.generated_queries == []
    assert record.eligible_seed_count == 0
    assert record.skipped_reasons == ["no_reliable_seed"]


def test_weak_and_insufficient_results_are_never_seeds() -> None:
    analysis = _analysis()
    weak = _judgement(
        _paper("Graph Models", "weak"),
        category="weakly_relevant",
        score=0.8,
    )
    insufficient = _judgement(
        _paper("Sparse Metadata", "insufficient"),
        category="insufficient_evidence",
        score=0.8,
    )

    record = evolve_queries(
        analysis,
        _plan(analysis),
        [weak, insufficient],
        [_ranked(weak), _ranked(insufficient, 2)],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert record.eligible_seed_count == 0
    assert record.seed_count == 0


def test_seed_selection_avoids_near_duplicates_and_caps_at_three() -> None:
    analysis = _analysis()
    papers = [
        _paper(
            "Graph Networks for Molecular Prediction",
            "seed-a",
            abstract="Graph molecular prediction.",
            sources=["arxiv"],
        ),
        _paper(
            "Graph Networks for Molecular Prediction",
            "seed-a-duplicate",
            abstract="Graph molecular prediction.",
            sources=["openalex"],
        ),
        _paper(
            "Molecular Property Learning with Graph Models",
            "seed-b",
            abstract="Graph neural networks for molecular prediction.",
            sources=["openalex"],
        ),
        _paper(
            "Neural Molecular Graph Prediction",
            "seed-c",
            abstract="Graph neural networks for molecular prediction.",
            sources=["arxiv"],
        ),
    ]
    judgements = [_judgement(paper, score=0.9) for paper in papers]

    record = evolve_queries(
        analysis,
        _plan(analysis),
        judgements,
        [_ranked(item, rank=index + 1) for index, item in enumerate(judgements)],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert record.seed_count == 3
    assert record.seed_paper_titles.count(
        "Graph Networks for Molecular Prediction"
    ) == 1


def test_coverage_gap_deduplicates_used_queries() -> None:
    analysis = _analysis()
    analysis = analysis.model_copy(
        update={
            "constraints": analysis.constraints.model_copy(
                update={"datasets": [], "paper_types": []}
            )
        }
    )
    seed = _judgement(
        _paper(
            "Graph Neural Networks for Molecular Prediction",
            "dedupe",
            abstract=(
                "Graph neural networks for molecular property prediction."
            ),
        )
    )
    first = evolve_queries(
        analysis,
        _plan(analysis),
        [seed],
        [_ranked(seed)],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )
    used = {item.query for item in first.generated_queries}

    second = evolve_queries(
        analysis,
        _plan(analysis),
        [seed],
        [_ranked(seed)],
        used_queries=used,
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert first.generated_queries
    assert second.generated_queries == []
    assert "duplicate_query" in second.skipped_reasons


def test_broad_query_without_actionable_dimension_does_not_evolve() -> None:
    analysis = QueryAnalysis(
        original_query="research papers",
        language="en",
        intent="general",
        domain="general_science",
    )
    seed = _judgement(_paper("General Research", "broad"))

    record = evolve_queries(
        analysis,
        _plan(analysis),
        [seed],
        [_ranked(seed)],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert record.generated_queries == []
    assert "coverage_sufficient" in record.skipped_reasons


def test_low_information_retention_guard_rejects_missing_core_or_must_have() -> None:
    assert not _retains_required_information(
        "graph retrieval",
        ["graph", "neural", "molecular"],
        ["QM9"],
    )


def test_policy_off_is_a_stable_noop() -> None:
    analysis = _analysis()
    first = evolve_queries(
        analysis,
        _plan(analysis),
        [],
        [],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="off"),
    )
    second = evolve_queries(
        analysis,
        _plan(analysis),
        [],
        [],
        used_queries=set(),
        options=QueryEvolutionOptions(policy="off"),
    )

    assert first.model_dump() == second.model_dump()
    assert first.policy == "off"
    assert first.generated_queries == []


def test_quality_gate_filters_duplicates_exclusions_and_dimension_misses() -> None:
    analysis = _analysis()
    initial = _paper(
        "Graph Neural Networks for Molecular Prediction",
        "initial",
        abstract="Molecular graph prediction.",
    )
    duplicate = initial.model_copy(deep=True)
    accepted = _paper(
        "Message Passing for Molecular Graphs",
        "accepted",
        abstract="A QM9 molecular property prediction method.",
        sources=["arxiv", "openalex"],
    )
    no_topic = _paper(
        "Message Passing for Traffic Forecasting",
        "no-topic",
        abstract="A temporal road network method on METR-LA.",
    )
    excluded_analysis = analysis.model_copy(
        update={
            "constraints": analysis.constraints.model_copy(
                update={"exclude_terms": ["survey"]}
            )
        }
    )
    excluded = _paper(
        "Survey of Molecular Message Passing",
        "excluded",
        abstract="A survey on graph neural networks and QM9.",
    )

    papers, gate = filter_evolved_candidates(
        excluded_analysis,
        [initial],
        [duplicate, accepted, accepted.model_copy(deep=True), no_topic, excluded],
    )

    assert [paper.identifiers.doi for paper in papers] == ["10.1000/accepted"]
    assert gate.raw_candidate_count == 5
    assert gate.unique_candidate_count == 4
    assert gate.duplicate_candidate_count == 1
    assert gate.duplicate_with_initial_count == 1
    assert gate.accepted_candidate_count == 1
    assert gate.filtered_candidate_count == 4
    assert gate.filtered_reason_counts == {
        "duplicate_with_initial": 1,
        "excluded_term_match": 1,
        "no_topic_match": 1,
    }
    assert gate.accepted_source_counts == {"arxiv": 1, "openalex": 1}


def test_coverage_gap_output_is_deterministic() -> None:
    analysis = _analysis()
    seed = _judgement(
        _paper(
            "Graph Neural Networks for Molecular Prediction",
            "stable-gap",
            abstract="Molecular property prediction with graph models.",
        )
    )
    arguments = (
        analysis,
        _plan(analysis),
        [seed],
        [_ranked(seed)],
    )

    first = evolve_queries(
        *arguments,
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )
    second = evolve_queries(
        *arguments,
        used_queries=set(),
        options=QueryEvolutionOptions(policy="coverage_gap"),
    )

    assert first.model_dump() == second.model_dump()
