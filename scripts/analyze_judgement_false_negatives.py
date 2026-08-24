#!/usr/bin/env python3
"""Audit retrieved gold papers that the rule judgement filtered out.

This is an offline diagnostic only.  It reads a completed, local evidence
copy and never feeds gold/qrels into retrieval, ranking, prompting, or
production code.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _walk(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _features_by_case(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Collect the latest judgement feature vector from committed deltas."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    generations = run_dir / ".run_commits" / "generations"
    for delta in sorted(generations.glob("generation-*/delta.json")):
        try:
            payload = json.loads(delta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record = payload.get("record") if isinstance(payload, dict) else None
        if not isinstance(record, dict):
            continue
        case_id = str(record.get("case_id") or "")
        if not case_id:
            continue
        for item in _walk(record):
            identifiers = item.get("identifiers") if isinstance(item, dict) else None
            features = item.get("judgement_features") if isinstance(item, dict) else None
            if not isinstance(identifiers, dict) or not isinstance(features, dict):
                continue
            arxiv_id = str(identifiers.get("arxiv_id") or "")
            if arxiv_id:
                found[(case_id, arxiv_id)] = {
                    "judgement_score": item.get("judgement_score"),
                    "category": item.get("category"),
                    "title": item.get("title"),
                    "features": features,
                }
    return found


def analyze(run_dir: Path) -> dict[str, Any]:
    diagnostics = list(_read_jsonl(run_dir / "gold_diagnostics.jsonl"))
    feature_map = _features_by_case(run_dir)
    rows: list[dict[str, Any]] = []
    for gold in diagnostics:
        reason = str(gold.get("drop_reason") or "")
        if reason not in {"judged_weakly_relevant", "judged_irrelevant"}:
            continue
        case_id = str(gold.get("case_id") or "")
        arxiv_id = str(gold.get("gold_id") or "").removeprefix("arxiv:")
        match = feature_map.get((case_id, arxiv_id), {})
        features = match.get("features") or {}
        components = features.get("score_components") or {}
        rows.append(
            {
                "case_id": case_id,
                "gold_id": gold.get("gold_id"),
                "gold_title": gold.get("gold_title"),
                "query": gold.get("query"),
                "drop_reason": reason,
                "initial_rank": gold.get("initial_rank"),
                "judgement_score": match.get("judgement_score"),
                "category": match.get("category"),
                "matched_terms": features.get("title_matched_terms") or features.get("matched_terms") or [],
                "matched_must_have_terms": features.get("matched_must_have_terms") or [],
                "score_components": components,
                "thresholds": {
                    "weakly_relevant": features.get("weakly_relevant_threshold"),
                    "partially_relevant": features.get("partially_relevant_threshold"),
                    "highly_relevant": features.get("highly_relevant_threshold"),
                },
                "feature_found": bool(match),
            }
        )

    component_totals: Counter[str] = Counter()
    term_counts: Counter[str] = Counter()
    reason_counts = Counter(row["drop_reason"] for row in rows)
    missing_feature_count = 0
    for row in rows:
        if not row["feature_found"]:
            missing_feature_count += 1
        for key, value in row["score_components"].items():
            if isinstance(value, (int, float)):
                component_totals[key] += float(value)
        for term in row["matched_must_have_terms"]:
            term_counts[str(term).casefold()] += 1

    return {
        "protocol": {
            "purpose": "offline judgement false-negative diagnosis",
            "gold_used_after_retrieval_only": True,
            "official_competition_metric": False,
        },
        "run_dir": str(run_dir),
        "retrieved_judgement_drop_count": len(rows),
        "drop_reason_counts": dict(sorted(reason_counts.items())),
        "feature_vectors_found": len(rows) - missing_feature_count,
        "feature_vectors_missing": missing_feature_count,
        "component_sum_across_rows": dict(sorted(component_totals.items())),
        "must_have_term_frequency": dict(term_counts.most_common()),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "retrieved_judgement_drop_count", "drop_reason_counts",
        "feature_vectors_found", "feature_vectors_missing",
        "must_have_term_frequency",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
