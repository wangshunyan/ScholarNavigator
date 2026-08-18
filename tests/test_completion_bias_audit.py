from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from scholar_agent.evaluation.completion_bias_audit import (
    CompletionBiasError,
    CompletionBiasNotEligible,
    _component_coverage,
    _order_diagnostics,
    build_population,
    canonical_json,
    compare_groups,
    component_permutation_labels,
    extract_query_features,
    load_protocol,
    opaque_query_identity,
    validate_feature_registry,
    write_analysis,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/completion_bias_audit_v1_protocol.json"


def _small_protocol(count: int = 4) -> dict[str, object]:
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["identity"].update(  # type: ignore[index]
        {
            "expected_query_count": count,
            "expected_recorded_count": 2,
            "expected_main_count": 1,
            "expected_excluded_count": 1,
            "expected_component_count": count,
        }
    )
    protocol["statistics"]["bootstrap_count"] = 30  # type: ignore[index]
    protocol["statistics"]["permutation_count"] = 30  # type: ignore[index]
    protocol["classifier"]["permutation_count"] = 30  # type: ignore[index]
    return protocol


def _membership_fixture() -> tuple[list[dict[str, str]], list[dict[str, object]], list[dict[str, object]]]:
    queries = [
        {"query_id": f"q-{index}", "query": f"query number {index}"}
        for index in range(4)
    ]
    records = [
        {
            "query_identity": opaque_query_identity("q-0"),
            "analysis_status": "included_main_analysis",
            "case_order": 0,
        },
        {
            "query_identity": opaque_query_identity("q-1"),
            "analysis_status": "excluded_no_successful_source",
            "case_order": 1,
        },
    ]
    components = [
        {
            "query_id": f"q-{index}",
            "component_id": f"component-{index}",
            "component_query_count": 1,
        }
        for index in range(4)
    ]
    return queries, records, components


def _zero_features(order: float) -> dict[str, object]:
    return {
        "character_count": 1.0,
        "digit_count": 0.0,
        "mean_token_length": 1.0,
        "non_ascii_count": 0.0,
        "normalized_order_position": order,
        "punctuation_count": 0.0,
        "quote_character_count": 0.0,
        "token_count": 1.0,
        "unique_token_count": 1.0,
        "year_pattern_count": 0.0,
        "has_boolean_operator": False,
        "has_parentheses": False,
        "has_question_mark": False,
        "has_quoted_span": False,
        "has_unicode": False,
        "has_year_pattern": False,
        "token_length_bucket": "short",
    }


def _population(statuses: list[str], components: list[str] | None = None) -> list[dict[str, object]]:
    rows = []
    for index, status in enumerate(statuses):
        component = components[index] if components else f"component:{index}"
        rows.append(
            {
                "query_identity": f"{index:064x}",
                "component_identity": component,
                "component_query_count": components.count(component) if components else 1,
                "order_index": index,
                "completion_status": status,
                "is_main": status == "included_main_analysis",
                "is_recorded": status != "unrecorded",
                "is_excluded": status == "excluded_no_successful_source",
                "features": _zero_features(0.0),
            }
        )
    return rows


def test_exact_membership_closure_and_opaque_output() -> None:
    queries, records, components = _membership_fixture()
    population = build_population(queries, records, components, _small_protocol())
    assert len(population) == 4
    assert sum(item["is_recorded"] for item in population) == 2
    assert sum(item["is_main"] for item in population) == 1
    assert sum(item["is_excluded"] for item in population) == 1
    assert all("q-" not in str(item["query_identity"]) for item in population)


def test_duplicate_and_missing_identity_are_rejected() -> None:
    queries, records, components = _membership_fixture()
    duplicate = [*queries, queries[-1]]
    protocol = _small_protocol(5)
    with pytest.raises(CompletionBiasError, match="duplicate_dataset_identity"):
        build_population(duplicate, records, components, protocol)

    with pytest.raises(CompletionBiasNotEligible, match="component_membership_not_closed"):
        build_population(queries, records, components[:-1], _small_protocol())


def test_source_order_prefix_is_detected_without_quality_data() -> None:
    population = _population(
        [
            "included_main_analysis",
            "included_main_analysis",
            "excluded_no_successful_source",
            "unrecorded",
        ]
    )
    order = _order_diagnostics(population)
    assert order["recorded_terminal"]["prefix_length"] == 3
    assert order["included_main_analysis"]["prefix_length"] == 2
    assert order["excluded_no_successful_source"]["prefix_length"] == 0


def test_component_level_imbalance_and_entire_missing_components() -> None:
    population = _population(
        ["included_main_analysis", "unrecorded", "unrecorded", "unrecorded"],
        ["component:a", "component:a", "component:b", "component:c"],
    )
    coverage = _component_coverage(population)["included_main_analysis"]
    assert coverage["touched_component_count"] == 1
    assert coverage["mixed_component_count"] == 1
    assert coverage["entirely_missing_component_count"] == 2
    assert coverage["entirely_missing_query_count"] == 2


def test_no_difference_sample_has_zero_effect_and_distance() -> None:
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["statistics"]["bootstrap_count"] = 20
    protocol["statistics"]["permutation_count"] = 20
    population = _population(
        [
            "included_main_analysis",
            "unrecorded",
            "included_main_analysis",
            "unrecorded",
        ]
    )
    result = compare_groups(
        population,
        {
            "comparison_id": "main160_vs_remaining840",
            "inferential": True,
            "left": "included_main_analysis",
            "right": "not_included_main_analysis",
        },
        protocol,
    )
    assert result["continuous"]["character_count"]["mean_difference"] == 0.0
    assert result["continuous"]["character_count"]["ks_distance"] == 0.0
    assert result["categorical"]["has_unicode"]["jensen_shannon_distance"] == 0.0


def test_forbidden_post_retrieval_feature_is_rejected() -> None:
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["continuous_features"][0] = "quality_metric"
    with pytest.raises(CompletionBiasError, match="forbidden_post_retrieval_feature"):
        validate_feature_registry(protocol)


def test_component_permutation_is_deterministic_and_preserves_counts() -> None:
    population = _population(
        ["included_main_analysis", "unrecorded", "included_main_analysis", "unrecorded"],
        ["component:a", "component:a", "component:b", "component:b"],
    )
    labels = [True, False, True, False]
    first = component_permutation_labels(population, labels, random.Random(42))
    second = component_permutation_labels(population, labels, random.Random(42))
    assert first == second
    assert sum(first) == sum(labels)


def test_unicode_boolean_quote_and_year_features_are_pre_retrieval() -> None:
    features = extract_query_features(
        '“β-cell” AND treatment after 2020?', order_index=4, query_count=10
    )
    assert features["has_unicode"] is True
    assert features["has_boolean_operator"] is True
    assert features["has_quoted_span"] is True
    assert features["has_year_pattern"] is True
    assert features["normalized_order_position"] == pytest.approx(4 / 9)


def test_report_files_are_byte_deterministic(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    queries = [
        {
            "schema_version": "1",
            "query_identity": "0" * 64,
            "component_identity": "component:test",
            "component_query_count": 1,
            "completion_status": "included_main_analysis",
            "order_index": 0,
            "features": _zero_features(0.0),
        }
    ]
    aggregate = {
        "schema_version": "1",
        "analysis": "completion_bias_audit_v1",
        "protocol_version": "completion-bias-audit-protocol-v1",
        "status": "completed",
        "inputs": {
            name: {"path": spec["path"], "sha256": spec["sha256"]}
            for name, spec in protocol["inputs"].items()
        },
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_analysis(first, queries, aggregate, PROTOCOL_PATH)
    write_analysis(second, queries, aggregate, PROTOCOL_PATH)
    assert {
        path.name: path.read_bytes() for path in sorted(first.iterdir())
    } == {
        path.name: path.read_bytes() for path in sorted(second.iterdir())
    }
    assert json.loads((first / "bundle.json").read_text())["status"] == "completed"
