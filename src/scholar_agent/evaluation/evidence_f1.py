"""Offline paragraph-level Evidence F1 evaluation.

The evaluator deliberately treats evidence annotations as a separate artifact
from paper relevance gold/qrels.  It scores exact paragraph evidence IDs only;
it never fetches documents, discovers licenses, or changes ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "evidence-f1-v1"
INTERNAL_METRIC_SCOPE = "not_official_competition_scorer"


class EvidenceF1Error(ValueError):
    """Raised when an evidence evaluation input violates its closed schema."""


def evaluate_evidence_f1(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score exact paragraph evidence IDs for a fixed query/paper set.

    Each row is identified by ``case_id`` and ``paper_id`` and contains a
    de-duplicated ``evidence_ids`` list.  A prediction row may be empty, but a
    gold row must contain at least one labelled evidence paragraph.  The
    caller is responsible for freezing the query set and recording the source
    document hashes in the surrounding manifest.
    """

    gold = _index_rows(gold_rows, label="gold", require_non_empty=True)
    predictions = _index_rows(
        prediction_rows, label="prediction", require_non_empty=False
    )
    missing_predictions = sorted(set(gold) - set(predictions))
    extra_predictions = sorted(set(predictions) - set(gold))
    if missing_predictions:
        raise EvidenceF1Error(
            "prediction_cases_missing:" + ",".join(missing_predictions[:5])
        )
    if extra_predictions:
        raise EvidenceF1Error(
            "prediction_cases_extra:" + ",".join(extra_predictions[:5])
        )

    per_case: list[dict[str, Any]] = []
    total_gold = 0
    total_predicted = 0
    total_true_positive = 0
    for key in sorted(gold):
        gold_ids = set(gold[key]["evidence_ids"])
        predicted_ids = set(predictions[key]["evidence_ids"])
        true_positive = len(gold_ids & predicted_ids)
        total_gold += len(gold_ids)
        total_predicted += len(predicted_ids)
        total_true_positive += true_positive
        per_case.append(
            {
                "case_id": gold[key]["case_id"],
                "paper_id": gold[key]["paper_id"],
                "gold_count": len(gold_ids),
                "predicted_count": len(predicted_ids),
                "true_positive_count": true_positive,
                "precision": _ratio(true_positive, len(predicted_ids)),
                "recall": _ratio(true_positive, len(gold_ids)),
                "f1": _f1(true_positive, len(predicted_ids), len(gold_ids)),
            }
        )

    macro_f1 = (
        sum(float(row["f1"]) for row in per_case) / len(per_case)
        if per_case
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_status": "complete",
        "internal_metric_scope": INTERNAL_METRIC_SCOPE,
        "case_count": len(per_case),
        "evaluable_case_count": len(per_case),
        "micro": {
            "gold_count": total_gold,
            "predicted_count": total_predicted,
            "true_positive_count": total_true_positive,
            "precision": _ratio(total_true_positive, total_predicted),
            "recall": _ratio(total_true_positive, total_gold),
            "f1": _f1(total_true_positive, total_predicted, total_gold),
        },
        "macro": {"f1": macro_f1},
        "per_case": per_case,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "gold_or_qrels_used_for_generation": False,
        },
    }


def pending_evidence_f1(reason: str = "human_evidence_labels_missing") -> dict[str, Any]:
    """Return an explicit non-score when annotations are not available."""

    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_status": "pending_human_labels",
        "internal_metric_scope": INTERNAL_METRIC_SCOPE,
        "case_count": 0,
        "evaluable_case_count": 0,
        "micro": None,
        "macro": None,
        "per_case": [],
        "reason": reason,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "gold_or_qrels_used_for_generation": False,
        },
    }


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    require_non_empty: bool,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise EvidenceF1Error(f"{label}_row_not_object:{position}")
        case_id = str(raw.get("case_id") or "").strip()
        paper_id = str(raw.get("paper_id") or "").strip()
        if not case_id or not paper_id:
            raise EvidenceF1Error(f"{label}_identity_missing:{position}")
        key = f"{case_id}\x1f{paper_id}"
        if key in indexed:
            raise EvidenceF1Error(f"{label}_duplicate_pair:{case_id}:{paper_id}")
        raw_ids = raw.get("evidence_ids")
        if not isinstance(raw_ids, list):
            raise EvidenceF1Error(f"{label}_evidence_ids_not_list:{case_id}:{paper_id}")
        evidence_ids = [str(item).strip() for item in raw_ids]
        if any(not item for item in evidence_ids):
            raise EvidenceF1Error(f"{label}_empty_evidence_id:{case_id}:{paper_id}")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise EvidenceF1Error(f"{label}_duplicate_evidence_id:{case_id}:{paper_id}")
        if require_non_empty and not evidence_ids:
            raise EvidenceF1Error(f"{label}_evidence_empty:{case_id}:{paper_id}")
        indexed[key] = {
            "case_id": case_id,
            "paper_id": paper_id,
            "evidence_ids": evidence_ids,
        }
    if not indexed:
        raise EvidenceF1Error(f"{label}_empty")
    return indexed


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(true_positive: int, predicted: int, gold: int) -> float:
    precision = _ratio(true_positive, predicted)
    recall = _ratio(true_positive, gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
