from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from scholar_agent.core.paper_schemas import Paper
from scholar_agent.evaluation.precision_annotation import (
    assert_blinded_rows,
    balanced_sample,
    cohen_kappa,
    evaluate_annotations,
    merge_strategy_candidates,
    validate_annotation_rows,
    write_precision_annotation_package,
)


def _item(dataset: str, stratum: str, index: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "stratum": stratum,
        "query": f"query {index}",
        "paper_identity": f"doi:10.1000/{dataset}-{stratum}-{index}",
        "case_id": f"case-{index}",
    }


def _mapping() -> dict[str, object]:
    return {
        "schema_version": "1",
        "top_k": 20,
        "population": {
            "datasets": {
                "fixture": {
                    "case_count": 1,
                    "baseline_returned_pair_count": 1,
                    "experiment_returned_pair_count": 1,
                }
            }
        },
        "samples": [
            {
                "sample_id": "SPAR-LX-0001",
                "dataset": "fixture",
                "stratum": "normalization_added",
                "baseline_returned": False,
                "experiment_returned": True,
                "cell_population_count": 1,
                "cell_sample_count": 1,
            },
            {
                "sample_id": "SPAR-LX-0002",
                "dataset": "fixture",
                "stratum": "baseline_only",
                "baseline_returned": True,
                "experiment_returned": False,
                "cell_population_count": 1,
                "cell_sample_count": 1,
            },
        ],
    }


def _annotations(
    annotator_id: str, labels: tuple[str | None, str | None]
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"SPAR-LX-{index:04d}",
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
            "sample_id": f"SPAR-LX-{index:04d}",
            "adjudicator_id": "adjudicator",
            "final_label": label,
            "rationale": "",
        }
        for index, label in enumerate(labels, start=1)
    ]


def test_balanced_sampling_is_deterministic_and_water_fills_cells() -> None:
    universe = [
        *[_item("a", "added", index) for index in range(5)],
        *[_item("a", "shared", index + 10) for index in range(5)],
        *[_item("b", "added", index + 20) for index in range(1)],
        *[_item("b", "shared", index + 30) for index in range(5)],
    ]
    first = balanced_sample(
        universe,
        dataset_order=["a", "b"],
        stratum_order=["added", "shared"],
        maximum=8,
        seed="fixed",
    )
    second = balanced_sample(
        list(reversed(universe)),
        dataset_order=["a", "b"],
        stratum_order=["added", "shared"],
        maximum=8,
        seed="fixed",
    )
    assert first == second
    counts = Counter((item["dataset"], item["stratum"]) for item in first)
    assert counts == {
        ("a", "added"): 3,
        ("a", "shared"): 2,
        ("b", "added"): 1,
        ("b", "shared"): 2,
    }


def test_blind_rows_reject_hidden_fields() -> None:
    clean = {
        "sample_id": "SPAR-LX-0001",
        "query": "What is the evidence?",
        "title": "A Unicode β paper",
        "abstract": "Abstract text.",
        "year": 2024,
    }
    assert_blinded_rows([clean], ["strategy", "rank", "case_id"])
    with pytest.raises(ValueError, match="public schema"):
        assert_blinded_rows(
            [{**clean, "strategy": "experiment"}],
            ["strategy", "rank", "case_id"],
        )

    nested = dict(clean)
    nested["abstract"] = {"text": "Abstract text.", "rank": 1}
    with pytest.raises(ValueError, match="forbidden blind fields"):
        assert_blinded_rows([nested], ["strategy", "rank", "case_id"])


def test_strategy_union_uses_unified_identity_and_preserves_conflicts() -> None:
    baseline = [
        Paper(
            title="Same paper",
            identifiers={"doi": "https://doi.org/10.1000/ABC"},
        )
    ]
    experiment = [
        Paper(title="Same paper", identifiers={"doi": "10.1000/abc"})
    ]
    merged = merge_strategy_candidates(baseline, experiment)
    assert len(merged) == 1
    assert merged[0]["baseline_rank"] == 1
    assert merged[0]["experiment_rank"] == 1

    conflict = merge_strategy_candidates(
        baseline,
        [Paper(title="Same paper", identifiers={"doi": "10.1000/different"})],
    )
    assert len(conflict) == 2


def test_annotation_validation_rejects_invalid_and_duplicate_labels() -> None:
    expected = ["SPAR-LX-0001"]
    with pytest.raises(ValueError, match="invalid annotation label"):
        validate_annotation_rows(
            [{"sample_id": expected[0], "label": "maybe"}], expected
        )
    with pytest.raises(ValueError, match="unique"):
        validate_annotation_rows(
            [
                {"sample_id": expected[0], "label": "relevant"},
                {"sample_id": expected[0], "label": "not_relevant"},
            ],
            expected,
        )


def test_disagreement_requires_adjudication_and_then_scores() -> None:
    first = _annotations("person-a", ("relevant", "not_relevant"))
    second = _annotations(
        "person-b", ("partially_relevant", "not_relevant")
    )
    pending = evaluate_annotations(
        _mapping(), first, second, _adjudication((None, None))
    )
    assert pending["annotation_status"] == "pending_adjudication"
    assert pending["agreement"]["disagreement_count"] == 1
    assert pending["metrics"] is None

    complete = evaluate_annotations(
        _mapping(), first, second, _adjudication(("relevant", None))
    )
    assert complete["annotation_status"] == "complete"
    assert complete["agreement"]["cohen_kappa"] == pytest.approx(1 / 3)
    fixture = complete["metrics"]["fixture"]
    assert fixture["strategies"]["experiment"]["sample_precision"] == 1.0
    assert fixture["strategies"]["experiment"]["precision_at_20"] == 0.05
    assert fixture["strategies"]["baseline"]["sample_precision"] == 0.0
    assert fixture["normalization_added_false_admission_rate"] == 0.0


def test_pending_templates_never_fabricate_metrics() -> None:
    result = evaluate_annotations(
        _mapping(),
        _annotations("person-a", (None, None)),
        _annotations("person-b", (None, None)),
        _adjudication((None, None)),
    )
    assert result["annotation_status"] == "pending_human_labels"
    assert result["agreement"] is None
    assert result["metrics"] is None


def test_annotators_must_be_distinct_and_adjudication_only_resolves_disagreement(
) -> None:
    agreed = _annotations("same-person", ("relevant", "not_relevant"))
    with pytest.raises(ValueError, match="distinct"):
        evaluate_annotations(
            _mapping(), agreed, agreed, _adjudication((None, None))
        )
    with pytest.raises(ValueError, match="must be blank"):
        evaluate_annotations(
            _mapping(),
            _annotations("person-a", ("relevant", "not_relevant")),
            _annotations("person-b", ("relevant", "not_relevant")),
            _adjudication(("relevant", None)),
        )


def test_cohen_kappa_boundaries() -> None:
    assert cohen_kappa([], []) is None
    assert cohen_kappa(["relevant"], ["relevant"]) == 1.0
    assert cohen_kappa(
        ["relevant", "not_relevant"],
        ["not_relevant", "relevant"],
    ) == -1.0


def test_package_writer_is_byte_deterministic(tmp_path: Path) -> None:
    package = {
        "manifest": {"package": "fixture"},
        "summary": {"sample_count": 1},
        "readme": "# fixture\n",
        "blind_samples": [
            {
                "sample_id": "SPAR-LX-0001",
                "query": "query",
                "title": "title",
                "abstract": "abstract",
                "year": 2024,
            }
        ],
        "annotation_schema": {"labels": ["relevant"]},
        "annotator_1": _annotations("person-a", (None, None))[:1],
        "annotator_2": _annotations("person-b", (None, None))[:1],
        "adjudication": _adjudication((None, None))[:1],
        "private_mapping": {"samples": [{"sample_id": "SPAR-LX-0001"}]},
        "metrics": {"annotation_status": "pending_human_labels"},
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_precision_annotation_package(first, package)
    write_precision_annotation_package(second, package)
    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    assert all(
        (first / relative).read_bytes() == (second / relative).read_bytes()
        for relative in first_files
    )
