from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_judgement_false_negatives import analyze


def test_analyze_judgement_false_negatives_reads_committed_feature_vectors(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    generations = run / ".run_commits" / "generations" / "generation-00000001"
    generations.mkdir(parents=True)
    (run / "gold_diagnostics.jsonl").write_text(
        json.dumps(
            {
                "case_id": "q0",
                "query": "query",
                "gold_id": "arxiv:1234.1",
                "gold_title": "Gold",
                "found": True,
                "initial_rank": 3,
                "drop_reason": "judged_weakly_relevant",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = {
        "case_id": "q0",
        "result": {
            "papers": [
                {
                    "identifiers": {"arxiv_id": "1234.1"},
                    "title": "Gold",
                    "category": "weakly_relevant",
                    "judgement_score": 0.3,
                    "judgement_features": {
                        "matched_must_have_terms": ["topic"],
                        "score_components": {
                            "topic_match": 0.2,
                            "constraint_coverage_adjustment": -0.06,
                        },
                    },
                }
            ]
        },
    }
    (generations / "delta.json").write_text(
        json.dumps({"kind": "record", "record": record}), encoding="utf-8"
    )

    result = analyze(run)

    assert result["retrieved_judgement_drop_count"] == 1
    assert result["feature_vectors_found"] == 1
    assert result["feature_vectors_missing"] == 0
    assert result["must_have_term_frequency"] == {"topic": 1}
    assert result["rows"][0]["score_components"]["topic_match"] == 0.2
