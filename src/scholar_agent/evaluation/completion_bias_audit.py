"""Gold-free completion-bias audit for the frozen AutoScholarQuery prefix.

The module consumes only the ordered query-only input, a frozen opaque Record
membership projection, and the already frozen query-component assignment.  It
never reads retrieval candidates, source output, evaluator inputs, or quality
metrics, and it never extrapolates unobserved retrieval behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import socket
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from unittest.mock import patch


SCHEMA_VERSION = "1"
ANALYSIS_NAME = "completion_bias_audit_v1"
PROTOCOL_VERSION = "completion-bias-audit-protocol-v1"
EXIT_COMPLETED = 0
EXIT_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_BOOLEAN_RE = re.compile(r"(?i)(?<![^\W_])(?:and|or|not)(?![^\W_])|&&|\|\|")
_QUOTE_CHARACTERS = frozenset('"\'“”‘’「」『』')
_FORBIDDEN_FEATURE_TOKENS = frozenset(
    {
        "gold",
        "qrels",
        "case_id",
        "target_paper",
        "retrieval_candidates",
        "source_output",
        "source_terminal_state",
        "ranking",
        "quality_metric",
    }
)


class CompletionBiasError(RuntimeError):
    """Identity, protocol, or analysis invariants were violated."""


class CompletionBiasNotEligible(CompletionBiasError):
    """Frozen inputs are insufficient for an exact audit."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _repo_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise CompletionBiasError("unsafe_input_path")
    candidate = (root / Path(*value.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CompletionBiasError("input_path_escape") from exc
    return candidate


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompletionBiasNotEligible("invalid_json_input") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise CompletionBiasNotEligible("jsonl_row_not_object")
                    rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompletionBiasNotEligible("invalid_jsonl_input") from exc
    return rows


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = _read_json(Path(path))
    if not isinstance(value, dict):
        raise CompletionBiasError("protocol_not_object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise CompletionBiasError("unsupported_schema_version")
    if value.get("analysis") != ANALYSIS_NAME:
        raise CompletionBiasError("unsupported_analysis")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise CompletionBiasError("unsupported_protocol_version")
    expected_execution = {
        "gold_or_qrels_loaded": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "retrieval_result_feature_count": 0,
        "snapshot_read_count": 0,
        "snapshot_write_count": 0,
    }
    if value.get("execution") != expected_execution:
        raise CompletionBiasError("offline_execution_contract_drift")
    validate_feature_registry(value)
    identity = value.get("identity") or {}
    expected_counts = {
        "expected_query_count": 1000,
        "expected_recorded_count": 162,
        "expected_main_count": 160,
        "expected_excluded_count": 2,
        "expected_component_count": 715,
    }
    if any(identity.get(key) != expected for key, expected in expected_counts.items()):
        raise CompletionBiasError("population_contract_drift")
    comparisons = [item.get("comparison_id") for item in value.get("comparisons") or []]
    if comparisons != [
        "main160_vs_remaining840",
        "recorded162_vs_unrecorded838",
        "excluded2_vs_main160",
    ]:
        raise CompletionBiasError("comparison_contract_drift")
    return value


def validate_feature_registry(protocol: Mapping[str, Any]) -> None:
    continuous = [str(value) for value in protocol.get("continuous_features") or []]
    categorical = [str(value) for value in protocol.get("categorical_features") or []]
    if len(continuous) != len(set(continuous)) or len(categorical) != len(set(categorical)):
        raise CompletionBiasError("duplicate_feature_name")
    if set(continuous) & set(categorical):
        raise CompletionBiasError("feature_type_collision")
    names = {value.casefold() for value in [*continuous, *categorical]}
    if names & _FORBIDDEN_FEATURE_TOKENS:
        raise CompletionBiasError("forbidden_post_retrieval_feature")
    registered = {
        "character_count",
        "digit_count",
        "mean_token_length",
        "non_ascii_count",
        "normalized_order_position",
        "punctuation_count",
        "quote_character_count",
        "token_count",
        "unique_token_count",
        "year_pattern_count",
        "has_boolean_operator",
        "has_parentheses",
        "has_question_mark",
        "has_quoted_span",
        "has_unicode",
        "has_year_pattern",
        "token_length_bucket",
    }
    if names != registered:
        raise CompletionBiasError("feature_registry_drift")


def extract_query_features(query: str, *, order_index: int, query_count: int) -> dict[str, Any]:
    if not isinstance(query, str):
        raise CompletionBiasNotEligible("query_text_missing")
    tokens = _TOKEN_RE.findall(unicodedata.normalize("NFKC", query).casefold())
    unique_tokens = set(tokens)
    token_count = len(tokens)
    quote_count = sum(character in _QUOTE_CHARACTERS for character in query)
    year_count = len(_YEAR_RE.findall(query))
    if token_count <= 8:
        bucket = "short"
    elif token_count <= 20:
        bucket = "medium"
    else:
        bucket = "long"
    return {
        "character_count": float(len(query)),
        "digit_count": float(sum(character.isdigit() for character in query)),
        "mean_token_length": (
            sum(len(token) for token in tokens) / token_count if token_count else 0.0
        ),
        "non_ascii_count": float(sum(ord(character) > 127 for character in query)),
        "normalized_order_position": (
            order_index / (query_count - 1) if query_count > 1 else 0.0
        ),
        "punctuation_count": float(
            sum(unicodedata.category(character).startswith("P") for character in query)
        ),
        "quote_character_count": float(quote_count),
        "token_count": float(token_count),
        "unique_token_count": float(len(unique_tokens)),
        "year_pattern_count": float(year_count),
        "has_boolean_operator": bool(_BOOLEAN_RE.search(query)),
        "has_parentheses": "(" in query or ")" in query,
        "has_question_mark": "?" in query or "？" in query,
        "has_quoted_span": quote_count >= 2,
        "has_unicode": any(ord(character) > 127 for character in query),
        "has_year_pattern": year_count > 0,
        "token_length_bucket": bucket,
    }


def opaque_query_identity(raw_identity: str) -> str:
    return hashlib.sha256(f"query\0{raw_identity}".encode("utf-8")).hexdigest()


def opaque_component_identity(raw_identity: str) -> str:
    return "component:" + hashlib.sha256(
        f"completion-bias-component-v1\0{raw_identity}".encode("utf-8")
    ).hexdigest()[:24]


def build_population(
    ordered_queries: Sequence[Mapping[str, Any]],
    record_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join all inputs by exact opaque identity and close the 1000/162/160 sets."""

    expected = protocol["identity"]
    if len(ordered_queries) != int(expected["expected_query_count"]):
        raise CompletionBiasNotEligible("ordered_query_count_drift")
    query_by_opaque: dict[str, dict[str, Any]] = {}
    raw_to_opaque: dict[str, str] = {}
    for order_index, item in enumerate(ordered_queries):
        if set(item) != set(protocol["inputs"]["ordered_queries"]["allowed_fields"]):
            raise CompletionBiasNotEligible("ordered_query_schema_drift")
        raw_identity = str(item.get("query_id") or "")
        if not raw_identity:
            raise CompletionBiasNotEligible("dataset_identity_missing")
        identity = opaque_query_identity(raw_identity)
        if identity in query_by_opaque or raw_identity in raw_to_opaque:
            raise CompletionBiasError("duplicate_dataset_identity")
        raw_to_opaque[raw_identity] = identity
        query_by_opaque[identity] = {
            "query_identity": identity,
            "order_index": order_index,
            "features": extract_query_features(
                str(item.get("query")),
                order_index=order_index,
                query_count=len(ordered_queries),
            ),
        }

    component_by_opaque: dict[str, tuple[str, int]] = {}
    for item in component_rows:
        raw_identity = str(item.get("query_id") or "")
        identity = raw_to_opaque.get(raw_identity)
        if identity is None:
            raise CompletionBiasError("component_identity_outside_dataset")
        if identity in component_by_opaque:
            raise CompletionBiasError("duplicate_component_assignment")
        raw_component = str(item.get("component_id") or "")
        component_size = int(item.get("component_query_count") or 0)
        if not raw_component or component_size <= 0:
            raise CompletionBiasNotEligible("component_metadata_missing")
        component_by_opaque[identity] = (
            opaque_component_identity(raw_component),
            component_size,
        )
    if set(component_by_opaque) != set(query_by_opaque):
        raise CompletionBiasNotEligible("component_membership_not_closed")
    actual_component_sizes = Counter(value[0] for value in component_by_opaque.values())
    if len(actual_component_sizes) != int(expected["expected_component_count"]):
        raise CompletionBiasNotEligible("component_count_drift")
    for identity, (component_id, declared_size) in component_by_opaque.items():
        if actual_component_sizes[component_id] != declared_size:
            raise CompletionBiasError("component_size_inconsistent")
        query_by_opaque[identity]["component_identity"] = component_id
        query_by_opaque[identity]["component_query_count"] = declared_size

    record_by_identity: dict[str, str] = {}
    record_order: list[str] = []
    allowed_statuses = {"included_main_analysis", "excluded_no_successful_source"}
    for item in record_rows:
        identity = str(item.get("query_identity") or "")
        status = str(item.get("analysis_status") or "")
        case_order = int(item.get("case_order", -1))
        if identity not in query_by_opaque:
            raise CompletionBiasError("record_identity_outside_dataset")
        if identity in record_by_identity:
            raise CompletionBiasError("duplicate_record_identity")
        if status not in allowed_statuses:
            raise CompletionBiasNotEligible("record_status_unknown")
        if case_order != len(record_order):
            raise CompletionBiasError("record_order_not_closed")
        record_by_identity[identity] = status
        record_order.append(identity)
    status_counts = Counter(record_by_identity.values())
    if len(record_by_identity) != int(expected["expected_recorded_count"]):
        raise CompletionBiasNotEligible("recorded_count_drift")
    if status_counts["included_main_analysis"] != int(expected["expected_main_count"]):
        raise CompletionBiasNotEligible("main_count_drift")
    if status_counts["excluded_no_successful_source"] != int(
        expected["expected_excluded_count"]
    ):
        raise CompletionBiasNotEligible("excluded_count_drift")

    population: list[dict[str, Any]] = []
    for identity, base in sorted(
        query_by_opaque.items(), key=lambda pair: int(pair[1]["order_index"])
    ):
        completion_status = record_by_identity.get(identity, "unrecorded")
        population.append(
            {
                **base,
                "completion_status": completion_status,
                "is_main": completion_status == "included_main_analysis",
                "is_recorded": completion_status != "unrecorded",
                "is_excluded": completion_status == "excluded_no_successful_source",
            }
        )
    return population


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return _rounded(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return _rounded(ordered[lower])
    weight = position - lower
    return _rounded(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "maximum": None,
            "mean": None,
            "median": None,
            "minimum": None,
            "q1": None,
            "q3": None,
            "standard_deviation": None,
        }
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    return {
        "count": len(values),
        "maximum": _rounded(max(values)),
        "mean": _rounded(mean),
        "median": _quantile(values, 0.5),
        "minimum": _rounded(min(values)),
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
        "standard_deviation": _rounded(math.sqrt(max(variance, 0.0))),
    }


def _ks_distance(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right:
        return None
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    points = sorted(set(left_sorted) | set(right_sorted))
    left_index = right_index = 0
    maximum = 0.0
    for point in points:
        while left_index < len(left_sorted) and left_sorted[left_index] <= point:
            left_index += 1
        while right_index < len(right_sorted) and right_sorted[right_index] <= point:
            right_index += 1
        maximum = max(
            maximum,
            abs(left_index / len(left_sorted) - right_index / len(right_sorted)),
        )
    return _rounded(maximum)


def _standardized_difference(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = (
        sum((value - left_mean) ** 2 for value in left) / (len(left) - 1)
        if len(left) > 1
        else 0.0
    )
    right_var = (
        sum((value - right_mean) ** 2 for value in right) / (len(right) - 1)
        if len(right) > 1
        else 0.0
    )
    denominator = math.sqrt((left_var + right_var) / 2)
    if denominator == 0:
        return 0.0 if left_mean == right_mean else None
    return _rounded((left_mean - right_mean) / denominator)


def _js_distance(left_counts: Mapping[str, int], right_counts: Mapping[str, int]) -> float | None:
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    if not left_total or not right_total:
        return None
    labels = sorted(set(left_counts) | set(right_counts))
    divergence = 0.0
    for label in labels:
        left = left_counts.get(label, 0) / left_total
        right = right_counts.get(label, 0) / right_total
        middle = (left + right) / 2
        if left:
            divergence += 0.5 * left * math.log2(left / middle)
        if right:
            divergence += 0.5 * right * math.log2(right / middle)
    return _rounded(math.sqrt(max(divergence, 0.0)))


def _group_masks(population: Sequence[Mapping[str, Any]], comparison_id: str) -> tuple[list[bool], list[bool]]:
    if comparison_id == "main160_vs_remaining840":
        left = [bool(item["is_main"]) for item in population]
        right = [not bool(item["is_main"]) for item in population]
    elif comparison_id == "recorded162_vs_unrecorded838":
        left = [bool(item["is_recorded"]) for item in population]
        right = [not bool(item["is_recorded"]) for item in population]
    elif comparison_id == "excluded2_vs_main160":
        left = [bool(item["is_excluded"]) for item in population]
        right = [bool(item["is_main"]) for item in population]
    else:
        raise CompletionBiasError("unknown_comparison")
    if any(l and r for l, r in zip(left, right, strict=True)):
        raise CompletionBiasError("comparison_groups_overlap")
    if not any(left) or not any(right):
        raise CompletionBiasNotEligible("comparison_group_empty")
    return left, right


def _component_indices(population: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(population):
        grouped[str(item["component_identity"])].append(index)
    return [
        sorted(indices, key=lambda index: str(population[index]["query_identity"]))
        for _component, indices in sorted(grouped.items())
    ]


def _bootstrap_differences(
    population: Sequence[Mapping[str, Any]],
    left: Sequence[bool],
    right: Sequence[bool],
    continuous_names: Sequence[str],
    categorical_levels: Mapping[str, Sequence[str]],
    *,
    count: int,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    components = _component_indices(population)
    rng = random.Random(seed)
    continuous_output = {name: [] for name in continuous_names}
    categorical_output = {
        name: {level: [] for level in levels}
        for name, levels in categorical_levels.items()
    }
    for _iteration in range(count):
        selected = [components[rng.randrange(len(components))] for _ in components]
        left_indices = [index for component in selected for index in component if left[index]]
        right_indices = [index for component in selected for index in component if right[index]]
        if not left_indices or not right_indices:
            continue
        for name in continuous_names:
            left_mean = sum(float(population[index]["features"][name]) for index in left_indices) / len(left_indices)
            right_mean = sum(float(population[index]["features"][name]) for index in right_indices) / len(right_indices)
            continuous_output[name].append(left_mean - right_mean)
        for name, levels in categorical_levels.items():
            for level in levels:
                def normalized(index: int) -> str:
                    value = population[index]["features"][name]
                    return str(value).lower() if isinstance(value, bool) else str(value)

                left_rate = sum(normalized(index) == level for index in left_indices) / len(left_indices)
                right_rate = sum(normalized(index) == level for index in right_indices) / len(right_indices)
                categorical_output[name][level].append(left_rate - right_rate)
    return continuous_output, categorical_output


def component_permutation_labels(
    population: Sequence[Mapping[str, Any]], labels: Sequence[bool], rng: random.Random
) -> list[bool]:
    """Permute complete label patterns among equally sized components."""

    components = _component_indices(population)
    by_size: dict[int, list[list[int]]] = defaultdict(list)
    for component in components:
        by_size[len(component)].append(component)
    output = [False] * len(population)
    for size in sorted(by_size):
        members = by_size[size]
        patterns = [[bool(labels[index]) for index in component] for component in members]
        rng.shuffle(patterns)
        for component, pattern in zip(members, patterns, strict=True):
            for index, value in zip(component, pattern, strict=True):
                output[index] = value
    if sum(output) != sum(bool(value) for value in labels):
        raise CompletionBiasError("permutation_label_count_drift")
    return output


def _permutation_p_values(
    population: Sequence[Mapping[str, Any]],
    left: Sequence[bool],
    right: Sequence[bool],
    continuous_names: Sequence[str],
    categorical_names: Sequence[str],
    *,
    count: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    active = [l or r for l, r in zip(left, right, strict=True)]
    selected_population = [item for item, keep in zip(population, active, strict=True) if keep]
    selected_left = [l for l, keep in zip(left, active, strict=True) if keep]
    observed_continuous: dict[str, float] = {}
    observed_categorical: dict[str, float] = {}
    for name in continuous_names:
        left_values = [float(item["features"][name]) for item, flag in zip(selected_population, selected_left, strict=True) if flag]
        right_values = [float(item["features"][name]) for item, flag in zip(selected_population, selected_left, strict=True) if not flag]
        observed_continuous[name] = abs(sum(left_values) / len(left_values) - sum(right_values) / len(right_values))
    for name in categorical_names:
        left_counts = Counter(str(item["features"][name]) for item, flag in zip(selected_population, selected_left, strict=True) if flag)
        right_counts = Counter(str(item["features"][name]) for item, flag in zip(selected_population, selected_left, strict=True) if not flag)
        observed_categorical[name] = float(_js_distance(left_counts, right_counts) or 0.0)

    continuous_extreme = Counter()
    categorical_extreme = Counter()
    rng = random.Random(seed)
    for _iteration in range(count):
        permuted = component_permutation_labels(selected_population, selected_left, rng)
        left_indices = [index for index, flag in enumerate(permuted) if flag]
        right_indices = [index for index, flag in enumerate(permuted) if not flag]
        if not left_indices or not right_indices:
            continue
        for name in continuous_names:
            left_mean = sum(float(selected_population[index]["features"][name]) for index in left_indices) / len(left_indices)
            right_mean = sum(float(selected_population[index]["features"][name]) for index in right_indices) / len(right_indices)
            if abs(left_mean - right_mean) >= observed_continuous[name] - 1e-15:
                continuous_extreme[name] += 1
        for name in categorical_names:
            left_counts = Counter(str(selected_population[index]["features"][name]) for index in left_indices)
            right_counts = Counter(str(selected_population[index]["features"][name]) for index in right_indices)
            value = float(_js_distance(left_counts, right_counts) or 0.0)
            if value >= observed_categorical[name] - 1e-15:
                categorical_extreme[name] += 1
    return (
        {name: _rounded((continuous_extreme[name] + 1) / (count + 1)) for name in continuous_names},
        {name: _rounded((categorical_extreme[name] + 1) / (count + 1)) for name in categorical_names},
    )


def compare_groups(
    population: Sequence[Mapping[str, Any]],
    comparison: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_id = str(comparison["comparison_id"])
    left_mask, right_mask = _group_masks(population, comparison_id)
    continuous_names = [str(value) for value in protocol["continuous_features"]]
    categorical_names = [str(value) for value in protocol["categorical_features"]]
    categorical_levels = {
        name: (
            ["false", "true"]
            if name != "token_length_bucket"
            else ["short", "medium", "long"]
        )
        for name in categorical_names
    }
    result: dict[str, Any] = {
        "comparison_id": comparison_id,
        "inferential": bool(comparison["inferential"]),
        "left_group": str(comparison["left"]),
        "left_query_count": sum(left_mask),
        "right_group": str(comparison["right"]),
        "right_query_count": sum(right_mask),
        "continuous": {},
        "categorical": {},
    }
    bootstrap_continuous: dict[str, list[float]] = {}
    bootstrap_categorical: dict[str, dict[str, list[float]]] = {}
    continuous_p: dict[str, float] = {}
    categorical_p: dict[str, float] = {}
    if comparison["inferential"]:
        bootstrap_continuous, bootstrap_categorical = _bootstrap_differences(
            population,
            left_mask,
            right_mask,
            continuous_names,
            categorical_levels,
            count=int(protocol["statistics"]["bootstrap_count"]),
            seed=int(protocol["random_seed"]) ^ int(hashlib.sha256((comparison_id + "bootstrap").encode()).hexdigest()[:8], 16),
        )
        continuous_p, categorical_p = _permutation_p_values(
            population,
            left_mask,
            right_mask,
            continuous_names,
            categorical_names,
            count=int(protocol["statistics"]["permutation_count"]),
            seed=int(protocol["random_seed"]) ^ int(hashlib.sha256((comparison_id + "permutation").encode()).hexdigest()[:8], 16),
        )
    for name in continuous_names:
        left_values = [float(item["features"][name]) for item, flag in zip(population, left_mask, strict=True) if flag]
        right_values = [float(item["features"][name]) for item, flag in zip(population, right_mask, strict=True) if flag]
        differences = bootstrap_continuous.get(name, [])
        result["continuous"][name] = {
            "left": _distribution(left_values),
            "right": _distribution(right_values),
            "mean_difference": _rounded(sum(left_values) / len(left_values) - sum(right_values) / len(right_values)),
            "standardized_mean_difference": _standardized_difference(left_values, right_values),
            "ks_distance": _ks_distance(left_values, right_values),
            "confidence_interval_95": (
                [_quantile(differences, 0.025), _quantile(differences, 0.975)]
                if differences
                else None
            ),
            "permutation_p_value": continuous_p.get(name),
            "holm_adjusted_p_value": None,
        }
    for name in categorical_names:
        def category(item: Mapping[str, Any]) -> str:
            value = item["features"][name]
            return str(value).lower() if isinstance(value, bool) else str(value)

        left_counts = Counter(category(item) for item, flag in zip(population, left_mask, strict=True) if flag)
        right_counts = Counter(category(item) for item, flag in zip(population, right_mask, strict=True) if flag)
        levels = categorical_levels[name]
        result["categorical"][name] = {
            "left_counts": {level: left_counts[level] for level in levels},
            "right_counts": {level: right_counts[level] for level in levels},
            "left_rates": {level: _rounded(left_counts[level] / sum(left_counts.values())) for level in levels},
            "right_rates": {level: _rounded(right_counts[level] / sum(right_counts.values())) for level in levels},
            "rate_difference_confidence_interval_95": {
                level: (
                    [
                        _quantile(bootstrap_categorical[name][level], 0.025),
                        _quantile(bootstrap_categorical[name][level], 0.975),
                    ]
                    if bootstrap_categorical
                    else None
                )
                for level in levels
            },
            "jensen_shannon_distance": _js_distance(left_counts, right_counts),
            "permutation_p_value": categorical_p.get(name),
            "holm_adjusted_p_value": None,
        }
    return result


def _apply_holm(comparisons: Sequence[dict[str, Any]]) -> None:
    tests: list[tuple[float, str, str, dict[str, Any]]] = []
    for comparison in comparisons:
        if not comparison["inferential"]:
            continue
        for kind in ("continuous", "categorical"):
            for name, item in comparison[kind].items():
                p_value = item["permutation_p_value"]
                if p_value is not None:
                    tests.append((float(p_value), str(comparison["comparison_id"]), f"{kind}:{name}", item))
    ordered = sorted(tests, key=lambda item: (item[0], item[1], item[2]))
    running = 0.0
    count = len(ordered)
    for index, (p_value, _comparison, _name, item) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_value))
        item["holm_adjusted_p_value"] = _rounded(running)


def _order_diagnostics(population: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, predicate in (
        ("included_main_analysis", lambda item: bool(item["is_main"])),
        ("recorded_terminal", lambda item: bool(item["is_recorded"])),
        ("excluded_no_successful_source", lambda item: bool(item["is_excluded"])),
    ):
        positions = [int(item["order_index"]) for item in population if predicate(item)]
        prefix = 0
        selected = set(positions)
        while prefix in selected:
            prefix += 1
        quartiles = {
            f"q{index + 1}": sum(index * 250 <= position < (index + 1) * 250 for position in positions)
            for index in range(4)
        }
        deciles = {
            f"d{index + 1}": sum(index * 100 <= position < (index + 1) * 100 for position in positions)
            for index in range(10)
        }
        output[name] = {
            "count": len(positions),
            "maximum_zero_based_position": max(positions) if positions else None,
            "mean_zero_based_position": _rounded(sum(positions) / len(positions)) if positions else None,
            "minimum_zero_based_position": min(positions) if positions else None,
            "prefix_length": prefix,
            "quartile_counts": quartiles,
            "decile_counts": deciles,
        }
    return output


def _component_coverage(population: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in population:
        grouped[str(item["component_identity"])].append(item)
    output: dict[str, Any] = {"component_count": len(grouped)}
    for name, predicate in (
        ("included_main_analysis", lambda item: bool(item["is_main"])),
        ("recorded_terminal", lambda item: bool(item["is_recorded"])),
        ("excluded_no_successful_source", lambda item: bool(item["is_excluded"])),
    ):
        completion_rates = []
        touched = complete = missing = mixed = 0
        missing_query_count = 0
        for rows in grouped.values():
            count = sum(predicate(item) for item in rows)
            rate = count / len(rows)
            completion_rates.append(rate)
            if count == 0:
                missing += 1
                missing_query_count += len(rows)
            elif count == len(rows):
                touched += 1
                complete += 1
            else:
                touched += 1
                mixed += 1
        output[name] = {
            "complete_component_count": complete,
            "component_coverage_rate": _rounded(touched / len(grouped)),
            "completion_rate_distribution": _distribution(completion_rates),
            "entirely_missing_component_count": missing,
            "entirely_missing_component_rate": _rounded(missing / len(grouped)),
            "entirely_missing_query_count": missing_query_count,
            "mixed_component_count": mixed,
            "touched_component_count": touched,
        }
    return output


def _classifier_matrix(population: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[list[float]]:
    continuous = [str(value) for value in protocol["continuous_features"]]
    categorical = [str(value) for value in protocol["categorical_features"]]
    rows: list[list[float]] = []
    for item in population:
        features = item["features"]
        row = [float(features[name]) for name in continuous]
        for name in categorical:
            if name == "token_length_bucket":
                row.extend(float(features[name] == level) for level in ("short", "medium", "long"))
            else:
                row.append(float(bool(features[name])))
        rows.append(row)
    return rows


def _fit_logistic(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[bool],
    test_x: Sequence[Sequence[float]],
    classifier: Mapping[str, Any],
) -> list[float]:
    width = len(train_x[0])
    means = [sum(row[index] for row in train_x) / len(train_x) for index in range(width)]
    deviations = []
    for index in range(width):
        variance = sum((row[index] - means[index]) ** 2 for row in train_x) / max(1, len(train_x) - 1)
        deviations.append(math.sqrt(variance) or 1.0)
    normalized_train = [[(row[index] - means[index]) / deviations[index] for index in range(width)] for row in train_x]
    normalized_test = [[(row[index] - means[index]) / deviations[index] for index in range(width)] for row in test_x]
    weights = [0.0] * (width + 1)
    positives = sum(train_y)
    negatives = len(train_y) - positives
    if not positives or not negatives:
        raise CompletionBiasNotEligible("classifier_fold_single_class")
    positive_weight = len(train_y) / (2 * positives)
    negative_weight = len(train_y) / (2 * negatives)
    learning_rate = float(classifier["learning_rate"])
    penalty = float(classifier["l2_penalty"])
    for _iteration in range(int(classifier["maximum_iterations"])):
        gradients = [0.0] * len(weights)
        for row, target in zip(normalized_train, train_y, strict=True):
            linear = weights[0] + sum(weight * value for weight, value in zip(weights[1:], row, strict=True))
            probability = 1 / (1 + math.exp(-max(-30.0, min(30.0, linear))))
            sample_weight = positive_weight if target else negative_weight
            error = (probability - float(target)) * sample_weight
            gradients[0] += error
            for index, value in enumerate(row, start=1):
                gradients[index] += error * value
        for index in range(len(weights)):
            regularization = 0.0 if index == 0 else penalty * weights[index]
            weights[index] -= learning_rate * (gradients[index] / len(train_y) + regularization)
    return [
        1 / (1 + math.exp(-max(-30.0, min(30.0, weights[0] + sum(weight * value for weight, value in zip(weights[1:], row, strict=True))))))
        for row in normalized_test
    ]


def _auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return _rounded((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _classifier_diagnostic(
    population: Sequence[Mapping[str, Any]],
    comparison_id: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    left, right = _group_masks(population, comparison_id)
    active = [l or r for l, r in zip(left, right, strict=True)]
    selected = [item for item, keep in zip(population, active, strict=True) if keep]
    labels = [flag for flag, keep in zip(left, active, strict=True) if keep]
    matrix = _classifier_matrix(selected, protocol)
    folds = int(protocol["classifier"]["cross_validation_folds"])
    fold_ids = [int(str(item["query_identity"])[:8], 16) % folds for item in selected]
    scores = [0.0] * len(selected)
    for fold in range(folds):
        train_indices = [index for index, value in enumerate(fold_ids) if value != fold]
        test_indices = [index for index, value in enumerate(fold_ids) if value == fold]
        predictions = _fit_logistic(
            [matrix[index] for index in train_indices],
            [labels[index] for index in train_indices],
            [matrix[index] for index in test_indices],
            protocol["classifier"],
        )
        for index, prediction in zip(test_indices, predictions, strict=True):
            scores[index] = prediction
    observed_auc = _auc(labels, scores)
    rng = random.Random(int(protocol["random_seed"]) ^ int(hashlib.sha256((comparison_id + "classifier").encode()).hexdigest()[:8], 16))
    null_values: list[float] = []
    for _iteration in range(int(protocol["classifier"]["permutation_count"])):
        permuted = component_permutation_labels(selected, labels, rng)
        value = _auc(permuted, scores)
        if value is not None:
            null_values.append(value)
    return {
        "auc": observed_auc,
        "cross_validation_folds": folds,
        "feature_count": len(matrix[0]),
        "interpretation": "descriptive distinguishability only; not causal and not a prediction of unfinished retrieval outcomes",
        "permutation_baseline": {
            "count": len(null_values),
            "confidence_interval_95": [_quantile(null_values, 0.025), _quantile(null_values, 0.975)],
            "mean": _rounded(sum(null_values) / len(null_values)) if null_values else None,
            "p_value": (
                _rounded((sum(value >= float(observed_auc) for value in null_values) + 1) / (len(null_values) + 1))
                if null_values and observed_auc is not None
                else None
            ),
        },
    }


def _extrapolation_boundaries(
    order: Mapping[str, Any], components: Mapping[str, Any], comparisons: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    main = order["included_main_analysis"]
    recorded = order["recorded_terminal"]
    main_components = components["included_main_analysis"]
    recorded_components = components["recorded_terminal"]
    flagged: list[str] = []
    if int(main["prefix_length"]) >= int(main["count"]):
        flagged.append("main_analysis_is_a_complete_source_order_prefix")
    if int(recorded["prefix_length"]) >= int(recorded["count"]):
        flagged.append("recorded_population_is_a_complete_source_order_prefix")
    if float(main_components["entirely_missing_component_rate"]) > 0:
        flagged.append("main_analysis_has_entirely_uncovered_query_components")
    if float(recorded_components["entirely_missing_component_rate"]) > 0:
        flagged.append("recorded_population_has_entirely_uncovered_query_components")
    for comparison in comparisons:
        if not comparison["inferential"]:
            continue
        if any(
            item["holm_adjusted_p_value"] is not None
            and float(item["holm_adjusted_p_value"]) <= 0.05
            for family in ("continuous", "categorical")
            for item in comparison[family].values()
        ):
            flagged.append(f"preregistered_feature_imbalance:{comparison['comparison_id']}")
    return {
        "applicable_scope": "frozen 160-query main-analysis population only",
        "flagged_boundaries": sorted(set(flagged)),
        "full1000_representativeness_claim_permitted": False,
        "mandatory_statements": [
            "Record160 coverage, source, ranking, constraint, and delivery findings apply only to the frozen 160-query main-analysis population.",
            "Absence of a detected feature difference would not establish representativeness of all 1000 queries.",
            "No unfinished-query retrieval outcome, relevance, source yield, or quality value is inferred by this audit.",
        ],
    }


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        attempts["network"] += 1
        raise CompletionBiasError("network_attempt_detected")

    with (
        patch.object(socket, "socket", blocked),
        patch.object(socket, "create_connection", blocked),
        patch.object(socket, "getaddrinfo", blocked),
    ):
        yield


def run_completion_bias_audit(
    protocol_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    inputs = protocol["inputs"]
    paths = {name: _repo_path(root, str(spec["path"])) for name, spec in inputs.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise CompletionBiasNotEligible("required_input_missing")
        if sha256_file(path) != str(inputs[name]["sha256"]):
            raise CompletionBiasNotEligible("frozen_input_hash_drift")
    attempts = {"network": 0}
    with _forbid_network(attempts):
        ordered_queries = _read_jsonl(paths["ordered_queries"])
        record_rows = _read_jsonl(paths["record_membership"])
        component_rows = _read_jsonl(paths["component_assignments"])
        population = build_population(ordered_queries, record_rows, component_rows, protocol)
        comparisons = [compare_groups(population, comparison, protocol) for comparison in protocol["comparisons"]]
        _apply_holm(comparisons)
        order = _order_diagnostics(population)
        components = _component_coverage(population)
        classifiers = {
            comparison_id: _classifier_diagnostic(population, comparison_id, protocol)
            for comparison_id in (
                "main160_vs_remaining840",
                "recorded162_vs_unrecorded838",
            )
        }
    if attempts["network"]:
        raise CompletionBiasError("network_attempt_detected")
    query_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "query_identity": str(item["query_identity"]),
            "component_identity": str(item["component_identity"]),
            "component_query_count": int(item["component_query_count"]),
            "completion_status": str(item["completion_status"]),
            "order_index": int(item["order_index"]),
            "features": {
                name: item["features"][name]
                for name in [*protocol["continuous_features"], *protocol["categorical_features"]]
            },
        }
        for item in population
    ]
    status_counts = Counter(str(item["completion_status"]) for item in population)
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "implementation_base_commit": "c4709c5ad849284fd6d00356573a06957038149b",
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "inputs": {
            name: {"path": str(spec["path"]), "sha256": str(spec["sha256"])}
            for name, spec in sorted(inputs.items())
        },
        "protocol_sha256": sha256_file(protocol_file),
        "closure": {
            "query_count": len(population),
            "recorded_query_count": sum(item["is_recorded"] for item in population),
            "included_main_analysis_count": sum(item["is_main"] for item in population),
            "excluded_no_successful_source_count": sum(item["is_excluded"] for item in population),
            "unrecorded_query_count": sum(not item["is_recorded"] for item in population),
            "remaining_not_main_count": sum(not item["is_main"] for item in population),
            "component_count": int(components["component_count"]),
            "status_counts": dict(sorted(status_counts.items())),
            "identity_join_mode": "exact_opaque_identity_only",
            "raw_identity_output_count": 0,
        },
        "comparisons": comparisons,
        "order_concentration": order,
        "component_coverage": components,
        "distinguishability": classifiers,
        "extrapolation_boundaries": _extrapolation_boundaries(order, components, comparisons),
        "execution": dict(protocol["execution"]),
        "interpretation": {
            "causal_claim": False,
            "effectiveness_or_quality_claim": False,
            "official_score_claim": False,
            "unfinished_query_outcome_inference": False,
            "scope": "pre-retrieval observable completion bias and extrapolation boundaries only",
        },
    }
    return query_rows, aggregate


def write_analysis(
    output_dir: str | Path,
    query_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protocol_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(protocol_path)
    contents = {
        "audit.json": canonical_json(aggregate).encode("utf-8"),
        "protocol.json": canonical_json(protocol).encode("utf-8"),
        "queries.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in query_rows
        ).encode("utf-8"),
    }
    for name, content in contents.items():
        (output / name).write_bytes(content)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "status": str(aggregate["status"]),
        "files": {
            name: {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
            for name, content in sorted(contents.items())
        },
        "input_hashes": {
            name: str(value["sha256"])
            for name, value in sorted(aggregate["inputs"].items())
        },
        "aggregate_sha256": hashlib.sha256(contents["audit.json"]).hexdigest(),
        "query_diagnostics_sha256": hashlib.sha256(contents["queries.jsonl"]).hexdigest(),
    }
    (output / "bundle.json").write_text(canonical_json(bundle), encoding="utf-8")
    return bundle


def verify_analysis(
    output_dir: str | Path,
    protocol_path: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    output = Path(output_dir)
    protocol = load_protocol(protocol_path)
    bundle_path = output / "bundle.json"
    if not bundle_path.is_file():
        raise CompletionBiasNotEligible("analysis_bundle_missing")
    bundle = _read_json(bundle_path)
    if bundle.get("analysis") != ANALYSIS_NAME or bundle.get("protocol_version") != PROTOCOL_VERSION:
        raise CompletionBiasError("analysis_bundle_protocol_drift")
    if bundle.get("status") != "completed":
        raise CompletionBiasNotEligible("analysis_not_completed")
    for name, spec in bundle.get("files", {}).items():
        path = output / str(name)
        if not path.is_file():
            raise CompletionBiasNotEligible("analysis_file_missing")
        if path.stat().st_size != int(spec["size"]) or sha256_file(path) != str(spec["sha256"]):
            raise CompletionBiasError("analysis_file_hash_drift")
    root = Path(repository_root).resolve()
    for name, spec in protocol["inputs"].items():
        if bundle["input_hashes"].get(name) != spec["sha256"]:
            raise CompletionBiasError("bundle_input_hash_drift")
        if sha256_file(_repo_path(root, str(spec["path"]))) != str(spec["sha256"]):
            raise CompletionBiasNotEligible("frozen_input_hash_drift")
    audit = _read_json(output / "audit.json")
    closure = audit.get("closure") or {}
    if closure.get("query_count") != 1000 or closure.get("recorded_query_count") != 162 or closure.get("included_main_analysis_count") != 160:
        raise CompletionBiasError("analysis_closure_drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "exit_code": EXIT_COMPLETED,
        "verified_file_count": len(bundle["files"]),
        "bundle_sha256": sha256_file(bundle_path),
        "execution": dict(protocol["execution"]),
    }


def _rounded(value: float) -> float:
    return round(float(value), 12)
