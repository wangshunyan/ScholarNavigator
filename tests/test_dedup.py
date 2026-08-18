from __future__ import annotations

from scholar_agent.core.dedup import (
    deduplicate_papers,
    deduplicate_papers_with_audit,
    paper_identity_evidence,
)
from scholar_agent.core.identity import build_identity_profile, normalize_title
from scholar_agent.core.identity import identity_evidence, paper_identifier_set
import scholar_agent.core.dedup as dedup_module
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers, PaperUrls


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
    sources: list[str] | None = None,
    citation_count: int = 0,
    abstract: str = "",
    authors: list[str] | None = None,
    venue: str | None = None,
    landing_page: str | None = None,
    pdf: str | None = None,
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
            openalex_id=openalex_id,
            semantic_scholar_id=semantic_scholar_id,
            s2orc_corpus_id=s2orc_corpus_id,
            pubmed_id=pubmed_id,
        ),
        urls=PaperUrls(landing_page=landing_page, pdf=pdf),
        sources=sources or [],
        citation_count=citation_count,
    )


def test_deduplicate_by_doi_and_merge_metadata() -> None:
    first = make_paper(
        "Short Title",
        doi="https://doi.org/10.1000/ABC",
        sources=["openalex"],
        citation_count=3,
        abstract="short",
        authors=["Alice"],
        openalex_id="W1",
        landing_page="https://example.org/a",
    )
    second = make_paper(
        "Longer and More Complete Title",
        doi="10.1000/abc",
        sources=["arxiv"],
        citation_count=12,
        abstract="This is a much longer abstract with more context.",
        authors=["Alice", "Bob"],
        arxiv_id="2401.12345v2",
        pdf="https://example.org/a.pdf",
        venue="ACL",
    )

    papers = deduplicate_papers([first, second])

    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "Longer and More Complete Title"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.venue == "ACL"
    assert paper.abstract == "This is a much longer abstract with more context."
    assert paper.identifiers.doi == "https://doi.org/10.1000/ABC"
    assert paper.identifiers.arxiv_id == "2401.12345v2"
    assert paper.identifiers.openalex_id == "W1"
    assert paper.urls.landing_page == "https://example.org/a"
    assert paper.urls.pdf == "https://example.org/a.pdf"
    assert paper.sources == ["openalex", "arxiv"]
    assert paper.citation_count == 12


def test_deduplicate_by_arxiv_id_ignores_version() -> None:
    papers = deduplicate_papers(
        [
            make_paper("Version One", arxiv_id="2407.18940v1", sources=["arxiv"]),
            make_paper("Version Two", arxiv_id="2407.18940v3", sources=["openalex"]),
        ]
    )

    assert len(papers) == 1
    assert papers[0].sources == ["arxiv", "openalex"]


def test_deduplicate_requires_author_and_exact_year_for_title_fallback() -> None:
    papers = deduplicate_papers(
        [
            make_paper(
                "A Survey of LLM-Based Reranking!",
                year=2024,
                sources=["openalex"],
                citation_count=5,
            ),
            make_paper(
                "a survey of llm based reranking",
                year=2025,
                sources=["arxiv"],
                citation_count=7,
            ),
        ]
    )

    assert len(papers) == 2


def test_deduplicate_title_author_year_is_order_independent() -> None:
    first = make_paper(
        "A Study: On Identity",
        year=2024,
        authors=["Alice Smith", "Bob Jones"],
        sources=["openalex"],
    )
    second = make_paper(
        "a study on identity",
        year=2024,
        authors=["Bob Jones", "Alice Smith"],
        sources=["arxiv"],
    )

    forward = deduplicate_papers([first, second])
    reverse = deduplicate_papers([second, first])

    assert len(forward) == len(reverse) == 1
    assert set(forward[0].sources) == set(reverse[0].sources) == {
        "openalex",
        "arxiv",
    }


def test_deduplicate_keeps_conflicting_identifiers_separate() -> None:
    papers = deduplicate_papers(
        [
            make_paper(
                "Same Work",
                year=2024,
                authors=["Alice"],
                doi="10.1000/first",
                openalex_id="W1",
            ),
            make_paper(
                "Same Work",
                year=2024,
                authors=["Alice"],
                doi="10.1000/second",
                openalex_id="W2",
            ),
        ]
    )

    assert len(papers) == 2


def test_deduplicate_normalizes_all_stable_identifier_formats() -> None:
    papers = deduplicate_papers(
        [
            make_paper(
                "Stable Paper",
                doi="https://doi.org/10.1000/ABC?x=1",
                arxiv_id="https://arxiv.org/abs/2401.00001v1",
            ),
            make_paper(
                "Different Metadata",
                doi="doi:10.1000/abc",
                arxiv_id="2401.00001v3",
                openalex_id="https://openalex.org/W1",
            ),
        ]
    )

    assert len(papers) == 1


def test_identity_audit_reports_rule_and_conflict_evidence() -> None:
    first = make_paper(
        "Audited Paper",
        doi="https://doi.org/10.1000/A",
        openalex_id="W1",
        authors=["Alice"],
    )
    second = make_paper(
        "Audited Paper Copy",
        doi="10.1000/a",
        openalex_id="W1",
        authors=["Alice"],
    )

    papers, audit = deduplicate_papers_with_audit([first, second])

    assert len(papers) == 1
    assert audit == [
        {
            "existing_index": 0,
            "incoming_title": "Audited Paper Copy",
            "rule": "shared_stable_identifier",
            "shared_identifiers": ["doi:10.1000/a", "openalex:w1"],
            "conflicting_identifiers": [],
            "propagated_identifiers": [],
            "title": None,
            "author_overlap": [],
            "year": None,
        }
    ]
    conflict = paper_identity_evidence(
        make_paper("Audited Paper", doi="10.1000/a"),
        make_paper("Audited Paper", doi="10.1000/b"),
    )
    assert conflict.equivalent is False
    assert conflict.rule == "conflicting_stable_identifier"


def test_identity_profile_reuses_normalized_fields_and_unicode_punctuation() -> None:
    assert normalize_title("A—Study… of “Models”") == "a study of models"
    profile = build_identity_profile(
        make_paper(
            "A—Study… of “Models”",
            authors=["Alice Smith", "Bob Jones"],
            year=2024,
        )
    )
    assert profile.title == "a study of models"
    assert profile.authors == {"alice smith", "bob jones"}
    assert profile.year == 2024


def test_batch_dedup_builds_one_profile_per_unique_input(monkeypatch) -> None:
    original = dedup_module.build_identity_profile
    calls = 0

    def counted(paper):
        nonlocal calls
        calls += 1
        return original(paper)

    monkeypatch.setattr(dedup_module, "build_identity_profile", counted)
    papers = [
        make_paper(f"Paper {index}", arxiv_id=f"2401.{index:05d}")
        for index in range(3)
    ]
    deduplicate_papers_with_audit(papers)
    assert calls == len(papers)


def test_deduplicate_keeps_distinct_title_when_year_far_apart() -> None:
    papers = deduplicate_papers(
        [
            make_paper("A Survey of LLM Based Reranking", year=2020),
            make_paper("A Survey of LLM Based Reranking", year=2024),
        ]
    )

    assert len(papers) == 2


def test_deduplicate_by_other_identifiers() -> None:
    papers = deduplicate_papers(
        [
            make_paper("OpenAlex Paper", openalex_id="https://openalex.org/W123"),
            make_paper("OpenAlex Paper Copy", openalex_id="w123"),
            make_paper("S2 Paper", semantic_scholar_id="S2-1"),
            make_paper("S2 Paper Copy", semantic_scholar_id="s2-1"),
            make_paper("S2ORC Paper", s2orc_corpus_id="CorpusId:123"),
            make_paper("S2ORC Paper Copy", s2orc_corpus_id="123"),
            make_paper("PubMed Paper", pubmed_id="https://pubmed.ncbi.nlm.nih.gov/999/"),
            make_paper("PubMed Paper Copy", pubmed_id="999"),
        ]
    )

    assert len(papers) == 4


def test_s2orc_aliases_and_numeric_values_use_exact_identity() -> None:
    assert paper_identifier_set({"corpus_id": 123}) == {"s2orc:123"}
    assert paper_identifier_set({"identifiers": {"corpusId": "123"}}) == {
        "s2orc:123"
    }
    assert paper_identifier_set({"metadata": {"s2orc_corpus_id": "123"}}) == {
        "s2orc:123"
    }
    assert identity_evidence(
        {"s2orc_corpus_id": 123},
        {"identifiers": {"CorpusId": "CorpusId:123"}},
    ).equivalent


def test_dedup_audit_records_s2orc_propagation_from_source_candidate() -> None:
    papers, audit = deduplicate_papers_with_audit(
        [
            make_paper("First source", doi="10.123/shared", sources=["arxiv"]),
            make_paper(
                "Authority source",
                doi="10.123/shared",
                s2orc_corpus_id="CorpusId:123",
                sources=["semantic_scholar"],
            ),
        ]
    )

    assert len(papers) == 1
    assert papers[0].identifiers.s2orc_corpus_id == "CorpusId:123"
    assert audit[0]["rule"] == "shared_stable_identifier"
    assert audit[0]["shared_identifiers"] == ["doi:10.123/shared"]
    assert audit[0]["propagated_identifiers"] == ["s2orc:123"]


def test_conflicting_s2orc_id_is_never_propagated_across_shared_doi() -> None:
    papers, audit = deduplicate_papers_with_audit(
        [
            make_paper(
                "First source",
                doi="10.123/shared",
                s2orc_corpus_id="123",
            ),
            make_paper(
                "Conflicting source",
                doi="10.123/shared",
                s2orc_corpus_id="456",
            ),
        ]
    )

    assert len(papers) == 2
    assert audit == []


def test_s2orc_conflict_blocks_title_author_year_fallback() -> None:
    shared_metadata = {
        "title": "An Exact Shared Title",
        "authors": ["Alice"],
        "year": 2024,
    }
    conflict = identity_evidence(
        {**shared_metadata, "s2orc_corpus_id": "123"},
        {**shared_metadata, "s2orc_corpus_id": "456"},
    )
    missing_candidate_id = identity_evidence(
        {**shared_metadata, "s2orc_corpus_id": "123"},
        shared_metadata,
    )

    assert conflict.equivalent is False
    assert conflict.rule == "conflicting_stable_identifier"
    assert conflict.conflicting_identifiers == ("s2orc_corpus_id:123!=456",)
    assert missing_candidate_id.equivalent is False
    assert missing_candidate_id.rule == "s2orc_requires_exact_identifier"


def test_old_paper_payload_without_s2orc_field_remains_compatible() -> None:
    paper = Paper.model_validate(
        {
            "title": "Legacy snapshot paper",
            "identifiers": {"semantic_scholar_id": "S2-legacy"},
        }
    )

    assert paper.identifiers.s2orc_corpus_id is None
