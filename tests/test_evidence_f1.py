from __future__ import annotations

import pytest

from scholar_agent.evaluation.evidence_f1 import (
    EvidenceF1Error,
    evaluate_evidence_f1,
    pending_evidence_f1,
)


def test_evidence_f1_scores_exact_paragraph_ids() -> None:
    result = evaluate_evidence_f1(
        [
            {"case_id": "q1", "paper_id": "p1", "evidence_ids": ["a", "b"]},
            {"case_id": "q2", "paper_id": "p2", "evidence_ids": ["c"]},
        ],
        [
            {"case_id": "q1", "paper_id": "p1", "evidence_ids": ["a", "x"]},
            {"case_id": "q2", "paper_id": "p2", "evidence_ids": ["c"]},
        ],
    )
    assert result["annotation_status"] == "complete"
    assert result["micro"]["true_positive_count"] == 2
    assert result["micro"]["precision"] == pytest.approx(2 / 3)
    assert result["micro"]["recall"] == pytest.approx(2 / 3)
    assert result["micro"]["f1"] == pytest.approx(2 / 3)


def test_evidence_f1_rejects_missing_and_duplicate_pairs() -> None:
    with pytest.raises(EvidenceF1Error, match="prediction_empty"):
        evaluate_evidence_f1(
            [{"case_id": "q1", "paper_id": "p1", "evidence_ids": ["a"]}],
            [],
        )
    with pytest.raises(EvidenceF1Error, match="gold_duplicate_pair"):
        evaluate_evidence_f1(
            [
                {"case_id": "q1", "paper_id": "p1", "evidence_ids": ["a"]},
                {"case_id": "q1", "paper_id": "p1", "evidence_ids": ["b"]},
            ],
            [{"case_id": "q1", "paper_id": "p1", "evidence_ids": ["a"]}],
        )


def test_pending_evidence_f1_is_not_a_zero_score() -> None:
    result = pending_evidence_f1()
    assert result["annotation_status"] == "pending_human_labels"
    assert result["micro"] is None
    assert result["macro"] is None
