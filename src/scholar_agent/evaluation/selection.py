"""Shared result selection policy for all evaluation entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal


ResultPolicy = Literal["highly_only", "highly_and_partial"]
DEFAULT_RESULT_POLICY: ResultPolicy = "highly_and_partial"

_ALLOWED_CATEGORIES: dict[ResultPolicy, tuple[str, ...]] = {
    "highly_only": ("highly_relevant",),
    "highly_and_partial": ("highly_relevant", "partially_relevant"),
}
_CATEGORY_ORDER = {"highly_relevant": 0, "partially_relevant": 1}


def select_ranked_results(
    result: Any,
    *,
    policy: ResultPolicy = DEFAULT_RESULT_POLICY,
) -> list[Any]:
    """Select the formal ranked result set with stable category/rank ordering."""

    if policy not in _ALLOWED_CATEGORIES:
        raise ValueError(f"unsupported result policy: {policy}")
    allowed = set(_ALLOWED_CATEGORIES[policy])
    candidates = _result_candidates(result)
    selected = [
        (item, category, index)
        for index, (item, category) in enumerate(candidates)
        if category in allowed
    ]
    selected.sort(
        key=lambda entry: (
            _CATEGORY_ORDER[entry[1]],
            _rank_value(entry[0], entry[2]),
            entry[2],
        )
    )
    return [item for item, _, _ in selected]


def _result_candidates(result: Any) -> list[tuple[Any, str]]:
    internal_ranked = _get_value(result, "ranked_papers")
    if isinstance(internal_ranked, Sequence) and not isinstance(
        internal_ranked,
        (str, bytes),
    ):
        return [
            (item, str(_get_value(item, "category") or ""))
            for item in internal_ranked
        ]

    candidates: list[tuple[Any, str]] = []
    for key, implied_category in (
        ("highly_relevant_papers", "highly_relevant"),
        ("partially_relevant_papers", "partially_relevant"),
    ):
        values = _get_value(result, key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for item in values:
            if not isinstance(item, Mapping) and not hasattr(item, "paper"):
                continue
            category = str(_get_value(item, "category") or implied_category)
            candidates.append((item, category))
    return candidates


def _rank_value(item: Any, fallback: int) -> int:
    raw_rank = _get_value(item, "rank")
    try:
        rank = int(raw_rank)
    except (TypeError, ValueError):
        return fallback + 1
    return rank if rank > 0 else fallback + 1


def _get_value(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)
