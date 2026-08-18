from __future__ import annotations

from pathlib import Path

import pytest

from scholar_agent.core.identity import build_identity_profile
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.evaluation.full_swap_precision_annotation import (
    deduplicate_swap_occurrences,
    evaluate_full_swap_annotations,
    partition_prior_overlaps,
    write_full_swap_package,
)
from scholar_agent.evaluation.precision_annotation import assert_blinded_rows


ROOT = Path(__file__).resolve().parents[1]


def _occurrence(
    *,
    query: str = "How does β work?",
    case_id: str = "private-case",
    direction: str = "experiment_admitted",
    doi: str = "10.1000/example",
    title: str = "Unicode β paper",
) -> dict[str, object]:
    return {
        "query": query,
        "query_fingerprint": "query-fingerprint",
        "paper": Paper(title=title, year=2024, identifiers={"doi": doi}),
        "case_id": case_id,
        "case_order": 0,
        "direction": direction,
        "rank": 20,
        "successful_source_count": 2,
        "overlaps_prior_auto_dev_val": False,
    }


def _mapping(*, prior_overlap: bool = False) -> dict[str, object]:
    mapping: dict[str, object] = {
        "schema_version": "1",
        "top_k": 20,
        "case_count": 1,
        "samples": [
            {
                "sample_id": "SPAR-LX-160-0001",
                "occurrences": [
                    {
                        "case_id": "q1",
                        "case_order": 0,
                        "direction": "experiment_admitted",
                        "rank": 20,
                        "successful_source_count": 2,
                        "overlaps_prior_auto_dev_val": False,
                    }
                ],
            },
            {
                "sample_id": "SPAR-LX-160-0002",
                "occurrences": [
                    {
                        "case_id": "q1",
                        "case_order": 0,
                        "direction": "baseline_removed",
                        "rank": 20,
                        "successful_source_count": 2,
                        "overlaps_prior_auto_dev_val": False,
                    }
                ],
            },
        ],
        "prior_package_overlaps": [],
    }
    if prior_overlap:
        mapping["prior_package_overlaps"] = [
            {
                "prior_sample_id": "SPAR-LX-0042",
                "occurrences": [
                    {
                        "case_id": "q1",
                        "case_order": 0,
                        "direction": "experiment_admitted",
                        "rank": 19,
                        "successful_source_count": 2,
                        "overlaps_prior_auto_dev_val": True,
                    }
                ],
            }
        ]
    return mapping


def _annotations(
    annotator_id: str, labels: tuple[str | None, str | None]
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"SPAR-LX-160-{index:04d}",
            "annotator_id": annotator_id,
            "label": label,
            "notes": "",
        }
        for index, label in enumerate(labels, start=1)
    ]


def _adjudication(
    labels: tuple[str | None, str | None]
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"SPAR-LX-160-{index:04d}",
            "adjudicator_id": "adjudicator",
            "final_label": label,
            "rationale": "",
        }
        for index, label in enumerate(labels, start=1)
    ]


def test_swap_dedup_uses_identity_and_preserves_identifier_conflict() -> None:
    duplicate = _occurrence(case_id="q2", doi="https://doi.org/10.1000/EXAMPLE")
    clusters, reasons = deduplicate_swap_occurrences(
        [_occurrence(), duplicate]
    )
    assert len(clusters) == 1
    assert len(clusters[0]["occurrences"]) == 2
    assert reasons == [
        {
            "reason": "same_query_unified_identity_equivalent",
            "kept_case_id": "private-case",
            "duplicate_case_id": "q2",
            "direction": "experiment_admitted",
        }
    ]

    conflicts, conflict_reasons = deduplicate_swap_occurrences(
        [_occurrence(), _occurrence(case_id="q2", doi="10.1000/conflict")]
    )
    assert len(conflicts) == 2
    assert conflict_reasons == []


def test_prior_package_overlap_is_referenced_not_reannotated() -> None:
    items, _ = deduplicate_swap_occurrences([_occurrence()])
    prior = [
        {
            "sample_id": "SPAR-LX-0017",
            "query_fingerprint": "query-fingerprint",
            "profile": build_identity_profile(
                Paper(
                    title="Different display title",
                    identifiers={"doi": "https://doi.org/10.1000/EXAMPLE"},
                )
            ),
        }
    ]
    fresh, overlaps = partition_prior_overlaps(items, prior)
    assert fresh == []
    assert overlaps[0]["prior_sample_id"] == "SPAR-LX-0017"
    assert overlaps[0]["identity_rule"] == "shared_stable_identifier"


def test_blind_rows_reject_nested_hidden_fields() -> None:
    row = {
        "sample_id": "SPAR-LX-160-0001",
        "query": "query",
        "title": "title",
        "abstract": {"text": "abstract", "source": "private"},
        "year": 2024,
    }
    with pytest.raises(ValueError, match="forbidden blind fields"):
        assert_blinded_rows([row], ["source", "rank", "case_id"])


def test_missing_labels_keep_all_precision_and_kappa_null() -> None:
    result = evaluate_full_swap_annotations(
        _mapping(),
        _annotations("person-a", (None, None)),
        _annotations("person-b", (None, None)),
        _adjudication((None, None)),
    )
    assert result["annotation_status"] == "pending_human_labels"
    assert result["agreement"]["cohen_kappa"] is None
    assert result["metrics"]["precision_at_20"] == {
        "baseline": None,
        "experiment": None,
    }
    assert result["metrics"]["paired_precision_at_20_difference"] is None


def test_disagreement_adjudication_scores_changed_precision_delta() -> None:
    first = _annotations("person-a", ("relevant", "not_relevant"))
    second = _annotations("person-b", ("partially_relevant", "not_relevant"))
    pending = evaluate_full_swap_annotations(
        _mapping(), first, second, _adjudication((None, None))
    )
    assert pending["annotation_status"] == "pending_adjudication"
    assert pending["metrics"]["paired_precision_at_20_difference"] is None

    complete = evaluate_full_swap_annotations(
        _mapping(), first, second, _adjudication(("relevant", None))
    )
    assert complete["annotation_status"] == "complete"
    assert complete["agreement"]["cohen_kappa"] == pytest.approx(1 / 3)
    metrics = complete["metrics"]
    assert metrics["changed_components"]["experiment_admitted"][
        "changed_item_precision"
    ] == 1.0
    assert metrics["changed_components"]["baseline_removed"][
        "changed_item_precision"
    ] == 0.0
    assert metrics["paired_precision_at_20_difference"] == 0.05
    assert metrics["admitted_false_admission_rate"] == 0.0
    assert metrics["removed_relevance_rate"] == 0.0
    assert metrics["precision_at_20"]["baseline"] is None


def test_prior_overlap_label_is_required_for_closed_scoring() -> None:
    pending = evaluate_full_swap_annotations(
        _mapping(prior_overlap=True),
        _annotations("person-a", ("relevant", "not_relevant")),
        _annotations("person-b", ("relevant", "not_relevant")),
        _adjudication((None, None)),
    )
    assert pending["annotation_status"] == "pending_prior_package_labels"
    assert pending["missing_prior_label_count"] == 1

    complete = evaluate_full_swap_annotations(
        _mapping(prior_overlap=True),
        _annotations("person-a", ("relevant", "not_relevant")),
        _annotations("person-b", ("relevant", "not_relevant")),
        _adjudication((None, None)),
        prior_resolved_labels={"SPAR-LX-0042": "not_relevant"},
    )
    assert complete["annotation_status"] == "complete"
    assert complete["metrics"]["admitted_false_admission_rate"] == 0.5


def test_package_writer_is_byte_deterministic(tmp_path: Path) -> None:
    public = [
        {
            "sample_id": "SPAR-LX-160-0001",
            "query": "query",
            "title": "title",
            "abstract": "abstract",
            "year": 2024,
        }
    ]
    package = {
        "manifest": {"package": "fixture"},
        "summary": {"public_new_sample_count": 1},
        "readme": "# fixture\n",
        "blind_samples": public,
        "annotation_schema": {"labels": ["relevant"]},
        "annotator_1": [
            {
                "sample_id": "SPAR-LX-160-0001",
                "annotator_id": "one",
                "label": None,
                "notes": "",
            }
        ],
        "annotator_2": [
            {
                "sample_id": "SPAR-LX-160-0001",
                "annotator_id": "two",
                "label": None,
                "notes": "",
            }
        ],
        "adjudication": [
            {
                "sample_id": "SPAR-LX-160-0001",
                "adjudicator_id": "",
                "final_label": None,
                "rationale": "",
            }
        ],
        "private_mapping": {"samples": []},
        "metrics": {"metrics": None},
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_full_swap_package(first, package)
    write_full_swap_package(second, package)
    first_files = sorted(
        path.relative_to(first) for path in first.rglob("*") if path.is_file()
    )
    assert first_files == sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    assert all(
        (first / relative).read_bytes() == (second / relative).read_bytes()
        for relative in first_files
    )


def test_tracked_package_closes_every_change_and_has_no_metrics() -> None:
    import json

    root = ROOT / "benchmark/lexical_normalization_record160_precision_annotation"
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    result = json.loads(
        (
            ROOT / "benchmark/lexical_normalization_record160_precision_result.json"
        ).read_text(encoding="utf-8")
    )
    public = [
        json.loads(line)
        for line in (root / "public/blind_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    mapping = json.loads(
        (root / "private/mapping.json").read_text(encoding="utf-8")
    )
    assert summary["top20_change_relation_count"] == 471
    assert summary["coverage_closure"] == {
        "covered_by_new_public_sample": 439,
        "covered_by_prior_package_reference": 32,
        "uncovered_relation_count": 0,
    }
    assert len(public) == len(mapping["samples"]) == 439
    assert len(mapping["prior_package_overlaps"]) == 32
    assert all(set(row) == {"sample_id", "query", "title", "abstract", "year"} for row in public)
    assert all(value is None for value in result["metrics"].values())
    assert result["public_blinding"]["recursive_forbidden_field_match_count"] == 0
