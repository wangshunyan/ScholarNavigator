from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.agents.reranker import (
    DEFAULT_TIEBREAK_POLICY,
    DETERMINISTIC_TIEBREAK_V2,
    DeterministicTieBreakUnavailable,
    deterministic_tiebreak_v2_catalog,
    deterministic_tiebreak_v2_key,
    rerank_papers,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    JudgementResult,
    QueryAnalysis,
    QueryConstraint,
)
from scholar_agent.evaluation.source_fusion_ablation import (
    IdentityRegistry,
    rank_variant,
)
from scholar_agent.evaluation.tiebreak_qualification import (
    EXIT_NOT_QUALIFIED,
    EXIT_QUALIFIED,
    aggregate_analysis,
    build_permutations,
    load_protocol,
    non_tie_order_drift,
    qualify_candidate_pool,
    write_analysis,
)


PROTOCOL_PATH = Path("benchmark/deterministic_tiebreak_qualification_v1_protocol.json")


def _analysis() -> QueryAnalysis:
    return QueryAnalysis(
        original_query="deterministic literature ranking",
        language="en",
        intent="general",
        domain="computer_science",
        constraints=QueryConstraint(),
    )


def _paper(
    identifier: str,
    *,
    title: str = "Shared exact tie title",
    citation_count: int = 0,
    with_identity: bool = True,
    abstract: str = "A deterministic literature ranking study.",
) -> Paper:
    return Paper(
        title=title,
        authors=["A. Author"] if with_identity else [],
        year=2024 if with_identity else None,
        abstract=abstract,
        identifiers=(
            PaperIdentifiers(doi=f"10.1000/{identifier}")
            if with_identity
            else PaperIdentifiers()
        ),
        sources=["arxiv"],
        citation_count=citation_count,
    )


def _judgement(paper: Paper) -> JudgementResult:
    return JudgementResult(
        paper=paper,
        score=0.5,
        category="partially_relevant",
        reasoning="fixture",
        evidence=[EvidenceItem(source="title", text="fixture", confidence=1.0)],
        matched_terms=[],
        warnings=[],
    )


def _qualification(papers: list[Paper]):
    analysis = _analysis()
    current = rank_variant(analysis, papers, top_k=20)
    registry = IdentityRegistry()
    identities = registry.labels(current.candidates)
    refs = {
        identity: (f"record:arxiv:{index:016x}:0001",)
        for index, identity in enumerate(identities)
    }
    profiles = build_permutations(
        current.candidates,
        identities,
        query_identity="query:fixture",
        protocol=load_protocol(PROTOCOL_PATH),
    )
    return qualify_candidate_pool(
        analysis,
        current.candidates,
        current=current,
        registry=registry,
        refs_by_identity=refs,
        lineage_hash_by_identity={identity: "a" * 64 for identity in identities},
        event_digest="b" * 64,
        profiles=profiles,
        sources=["openalex", "arxiv", "semantic_scholar", "pubmed"],
        top_k=20,
        query_identity="query:fixture",
        case_order=0,
    )


def test_protocol_freezes_exact_tie_and_default_off() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    assert protocol["current_policy"]["tie_definition"].endswith(
        "no tolerance, bucket, or approximate comparison is permitted"
    )
    assert DEFAULT_TIEBREAK_POLICY == "original_index_v1"
    assert deterministic_tiebreak_v2_catalog()["default_enabled"] is False


def test_v2_exact_tie_is_input_order_independent_but_default_is_not() -> None:
    analysis = _analysis()
    papers = [_paper("zeta"), _paper("alpha")]
    judgements = [_judgement(paper) for paper in papers]
    refs = {
        0: ("record:arxiv:0000000000000000:0001",),
        1: ("record:arxiv:0000000000000001:0001",),
    }
    default_forward = rerank_papers(analysis, judgements)
    default_reverse = rerank_papers(analysis, list(reversed(judgements)))
    assert [item.paper.identifiers.doi for item in default_forward] == list(
        reversed([item.paper.identifiers.doi for item in default_reverse])
    )

    v2_forward = rerank_papers(
        analysis,
        judgements,
        tie_break_policy=DETERMINISTIC_TIEBREAK_V2,
        source_record_refs=refs,
    )
    v2_reverse = rerank_papers(
        analysis,
        list(reversed(judgements)),
        tie_break_policy=DETERMINISTIC_TIEBREAK_V2,
        source_record_refs={0: refs[1], 1: refs[0]},
    )
    assert [item.paper.identifiers.doi for item in v2_forward] == [
        item.paper.identifiers.doi for item in v2_reverse
    ]


def test_non_exact_tie_keeps_primary_order() -> None:
    analysis = _analysis()
    papers = [_paper("low", citation_count=1), _paper("high", citation_count=10)]
    ranked = rerank_papers(
        analysis,
        [_judgement(paper) for paper in reversed(papers)],
        tie_break_policy=DETERMINISTIC_TIEBREAK_V2,
        source_record_refs={
            0: ("record:arxiv:0000000000000001:0001",),
            1: ("record:arxiv:0000000000000000:0001",),
        },
    )
    assert ranked[0].paper.citation_count == 10


def test_missing_identity_uses_registered_fallback_and_rejects_missing_refs() -> None:
    first = _paper("first", with_identity=False, abstract="first abstract")
    second = _paper("second", with_identity=False, abstract="second abstract")
    first_key = deterministic_tiebreak_v2_key(
        first, source_record_refs=("record:arxiv:aaaaaaaaaaaaaaaa:0001",)
    )
    second_key = deterministic_tiebreak_v2_key(
        second, source_record_refs=("record:arxiv:bbbbbbbbbbbbbbbb:0001",)
    )
    assert first_key.startswith("2:")
    assert first_key != second_key
    with pytest.raises(
        DeterministicTieBreakUnavailable,
        match="registered_source_record_reference_required",
    ):
        deterministic_tiebreak_v2_key(first)


def test_duplicate_authoritative_identity_in_exact_tie_is_rejected() -> None:
    analysis = _analysis()
    first = _paper("duplicate")
    second = first.model_copy(deep=True)
    with pytest.raises(
        DeterministicTieBreakUnavailable,
        match="stable_key_collision_in_exact_tie",
    ):
        rerank_papers(
            analysis,
            [_judgement(first), _judgement(second)],
            tie_break_policy=DETERMINISTIC_TIEBREAK_V2,
        )


def test_all_registered_permutations_are_stable_for_exact_tie() -> None:
    result = _qualification([_paper("zeta"), _paper("alpha"), _paper("beta")])
    assert result["case"]["all_permutations_byte_equivalent"] is True
    assert result["case"]["non_tie_order_changed"] is False
    assert result["case"]["authoritative_identity_changed"] is False
    assert len(result["ties"]) == 1
    assert result["ties"][0]["permutation_stable"] is True


def test_cutline_tie_that_changes_membership_is_not_qualified() -> None:
    papers = [_paper(str(index)) for index in range(21)]
    papers.sort(key=deterministic_tiebreak_v2_key, reverse=True)
    result = _qualification(papers)
    assert result["case"]["top20_membership_changed"] is True
    assert result["ties"][0]["crosses_top20_cutline"] is True
    aggregate = aggregate_analysis(
        [{"component_identity": "component:1", **result["case"]}],
        [],
        result["candidates"],
        result["ties"],
        load_protocol(PROTOCOL_PATH),
        protocol_sha256="c" * 64,
        input_hashes={},
        observed_snapshot_key_count=0,
    )
    assert aggregate["status"] == "not_qualified"
    assert aggregate["exit_code"] == EXIT_NOT_QUALIFIED


def test_non_tie_order_drift_is_detected() -> None:
    primary = {"a": (0,), "b": (1,), "c": (2,)}
    assert non_tie_order_drift(["a", "b", "c"], ["a", "c", "b"], primary)
    assert not non_tie_order_drift(["a", "b"], ["a", "b"], primary)


def test_qualified_aggregate_keeps_v2_default_off() -> None:
    result = _qualification([_paper("zeta"), _paper("alpha")])
    aggregate = aggregate_analysis(
        [{"component_identity": "component:1", **result["case"]}],
        [],
        result["candidates"],
        result["ties"],
        load_protocol(PROTOCOL_PATH),
        protocol_sha256="d" * 64,
        input_hashes={},
        observed_snapshot_key_count=0,
    )
    assert aggregate["status"] == "qualified_for_review"
    assert aggregate["exit_code"] == EXIT_QUALIFIED
    assert aggregate["qualification"]["automatic_enable_permitted"] is False
    assert aggregate["policy"]["v2_enabled_by_default"] is False


def test_report_files_are_byte_deterministic(tmp_path: Path) -> None:
    result = _qualification([_paper("zeta"), _paper("alpha")])
    cases = [{"case_order": 0, "component_identity": "component:1", **result["case"]}]
    aggregate = aggregate_analysis(
        cases,
        [],
        result["candidates"],
        result["ties"],
        load_protocol(PROTOCOL_PATH),
        protocol_sha256="e" * 64,
        input_hashes={},
        observed_snapshot_key_count=0,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_analysis(
        first,
        cases,
        result["candidates"],
        result["ties"],
        aggregate,
        PROTOCOL_PATH,
    )
    write_analysis(
        second,
        cases,
        result["candidates"],
        result["ties"],
        aggregate,
        PROTOCOL_PATH,
    )
    assert {
        path.name: path.read_bytes() for path in sorted(first.iterdir())
    } == {path.name: path.read_bytes() for path in sorted(second.iterdir())}
    assert json.loads((first / "aggregate.json").read_text())["status"] == (
        "qualified_for_review"
    )
