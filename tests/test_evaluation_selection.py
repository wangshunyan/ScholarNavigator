from __future__ import annotations

from scholar_agent.evaluation.selection import select_ranked_results


def _item(title: str, category: str, rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "category": category,
        "paper": {"title": title, "year": 2024},
    }


def test_highly_and_partial_selection_uses_category_then_stable_rank() -> None:
    result = {
        "ranked_papers": [
            _item("Partial 2", "partially_relevant", 2),
            _item("Weak", "weakly_relevant", 1),
            _item("High 3", "highly_relevant", 3),
            _item("Irrelevant", "irrelevant", 1),
            _item("High 1", "highly_relevant", 1),
            _item("Partial 1", "partially_relevant", 1),
            _item("Insufficient", "insufficient_evidence", 1),
        ]
    }

    selected = select_ranked_results(result)

    assert [item["paper"]["title"] for item in selected] == [
        "High 1",
        "High 3",
        "Partial 1",
        "Partial 2",
    ]


def test_highly_only_excludes_every_other_category() -> None:
    result = {
        "ranked_papers": [
            _item("Partial", "partially_relevant", 1),
            _item("High", "highly_relevant", 2),
            _item("Weak", "weakly_relevant", 3),
        ]
    }

    selected = select_ranked_results(result, policy="highly_only")

    assert [item["paper"]["title"] for item in selected] == ["High"]
