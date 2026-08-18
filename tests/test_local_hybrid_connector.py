from __future__ import annotations

from scholar_agent.connectors.local_hybrid import _fuse_ranked_lists
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers


def _paper(
    arxiv_id: str,
    title: str,
    *,
    abstract: str = "",
    sources: list[str] | None = None,
) -> Paper:
    return Paper(
        title=title,
        abstract=abstract,
        identifiers=PaperIdentifiers(arxiv_id=arxiv_id),
        sources=sources or [],
    )


def test_local_hybrid_rrf_merges_channels_and_prefers_shared_candidates() -> None:
    bm25 = [
        _paper("a", "Graph retrieval methods", sources=["local_bm25"]),
        _paper("b", "Causal bandits", sources=["local_bm25"]),
    ]
    semantic = [
        _paper(
            "b",
            "Causal bandits",
            abstract="Interventions selected through causal inference.",
            sources=["local_semantic"],
        ),
        _paper("c", "Graph neural retrieval"),
    ]

    fused = _fuse_ranked_lists(bm25, semantic, limit=3, rrf_k=60)

    assert [item.identifiers.arxiv_id for item in fused] == ["b", "a", "c"]
    assert fused[0].abstract.startswith("Interventions")
    assert fused[0].sources == ["local_hybrid", "local_bm25", "local_semantic"]


def test_local_hybrid_rrf_is_deterministic_for_equal_scores() -> None:
    left = [_paper("b", "B"), _paper("a", "A")]
    right = [_paper("a", "A"), _paper("b", "B")]

    first = _fuse_ranked_lists(left, right, limit=2, rrf_k=60)
    second = _fuse_ranked_lists(left, right, limit=2, rrf_k=60)

    assert [item.identifiers.arxiv_id for item in first] == [
        item.identifiers.arxiv_id for item in second
    ]
