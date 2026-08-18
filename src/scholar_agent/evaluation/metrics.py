"""Single source of truth for search evaluation matching and metrics."""

from __future__ import annotations

import math
import re
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scholar_agent.core.evaluation_schemas import (
    EvalAggregateEfficiency,
    EvalCaseEfficiency,
    EvalGoldPaper,
    EvalMetricVersion,
    EvalMetricSet,
    DEDUPLICATED_GOLD_METRIC_VERSION,
    LEGACY_GOLD_METRIC_VERSION,
)
from scholar_agent.core.identity import (
    IdentityProfile,
    build_identity_profile,
    identity_evidence_from_profiles,
    paper_identifier_set as shared_paper_identifier_set,
    paper_title_year_key as shared_paper_title_year_key,
    normalize_arxiv_id as shared_normalize_arxiv_id,
    normalize_doi as shared_normalize_doi,
    normalize_simple_id as shared_normalize_simple_id,
    normalize_title as shared_normalize_title,
)
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.core.search_schemas import RankedPaper


@dataclass(frozen=True)
class _GoldRecord:
    index: int
    paper: Any
    profile: IdentityProfile
    member_indices: tuple[int, ...]
    identifiers: frozenset[str]
    title_year: str | None
    grade: float


@dataclass(frozen=True)
class _RankedMatch:
    rank: int
    gold_index: int
    grade: float
    match_key: str


DEFAULT_METRIC_VERSION: EvalMetricVersion = DEDUPLICATED_GOLD_METRIC_VERSION


def canonical_paper_id(
    paper: Any = None,
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    openalex_id: str | None = None,
    semantic_scholar_id: str | None = None,
    s2orc_corpus_id: str | int | None = None,
    pubmed_id: str | None = None,
    title: str | None = None,
    year: int | None = None,
) -> str | None:
    """Return one display ID; metric matching always uses every stable ID."""

    values = {
        "doi": doi,
        "arxiv_id": arxiv_id,
        "openalex_id": openalex_id,
        "semantic_scholar_id": semantic_scholar_id,
        "s2orc_corpus_id": s2orc_corpus_id,
        "pubmed_id": pubmed_id,
        "title": title,
        "year": year,
    }
    if paper is None:
        paper = values
    else:
        paper = _unwrap_ranked_paper(paper)
        paper = _overlay_values(paper, values)

    doi_value = _extract_identifier(paper, "doi")
    if doi_value:
        normalized_doi = _normalize_doi(doi_value)
        arxiv_from_doi = _arxiv_id_from_doi(normalized_doi)
        if arxiv_from_doi:
            return f"arxiv:{arxiv_from_doi}"
        if normalized_doi:
            return f"doi:{normalized_doi}"
    identifiers = paper_identifier_set(paper)
    for prefix in ("arxiv:", "openalex:", "s2:", "s2orc:", "pubmed:"):
        matches = sorted(item for item in identifiers if item.startswith(prefix))
        if matches:
            return matches[0]
    return paper_title_year_key(paper)


def paper_identifier_set(paper: Any) -> set[str]:
    """Use the production identity normalizer for evaluator matching."""

    return shared_paper_identifier_set(_unwrap_ranked_paper(paper))


def paper_title_year_key(paper: Any) -> str | None:
    return shared_paper_title_year_key(_unwrap_ranked_paper(paper))


def matched_paper_ids(
    ranked_papers: Sequence[Any],
    gold_papers: Sequence[Any],
    *,
    k: int | None = None,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> list[str]:
    return [
        match.match_key
        for match in _ranked_matches(
            ranked_papers,
            gold_papers,
            k=k,
            metric_version=metric_version,
        )
    ]


def evaluable_gold_count(
    gold_papers: Sequence[Any],
    *,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> int:
    """Count positive gold records that have a usable matching key."""

    return len(_positive_gold_records(gold_papers, metric_version=metric_version))


def gold_deduplication_audit(gold_papers: Sequence[Any]) -> dict[str, Any]:
    """Describe every v2 denominator merge without mutating the input gold."""

    raw = _raw_positive_gold_records(gold_papers)
    deduplicated = _deduplicate_gold_records(raw)
    duplicate_relations: list[dict[str, Any]] = []
    by_index = {record.index: record for record in raw}
    for record in deduplicated:
        retained = by_index[record.index]
        for duplicate_index in record.member_indices:
            if duplicate_index == retained.index:
                continue
            duplicate = by_index[duplicate_index]
            basis = _direct_identity_basis(duplicate, record, by_index)
            evidence = identity_evidence_from_profiles(
                duplicate.profile,
                basis.profile,
            )
            duplicate_relations.append(
                {
                    "duplicate_index": duplicate.index,
                    "retained_index": retained.index,
                    "identity_basis_index": basis.index,
                    "rule": evidence.rule,
                    "shared_identifiers": list(evidence.shared_identifiers),
                    "conflicting_identifiers": list(evidence.conflicting_identifiers),
                    "title": evidence.title,
                    "author_overlap": list(evidence.author_overlap),
                    "year": evidence.year,
                    "retained_identity": _gold_profile_key(record.profile),
                }
            )
    return {
        "metric_version": DEDUPLICATED_GOLD_METRIC_VERSION,
        "legacy_evaluable_gold_count": len(raw),
        "deduplicated_evaluable_gold_count": len(deduplicated),
        "duplicate_relation_count": len(raw) - len(deduplicated),
        "duplicate_relations": sorted(
            duplicate_relations,
            key=lambda item: (item["duplicate_index"], item["retained_index"]),
        ),
    }


def gold_crosswalk_status(paper: Any) -> str | None:
    """Return an evaluator-only crosswalk terminal status when present."""

    unwrapped = _unwrap_ranked_paper(paper)
    metadata = _get_value(unwrapped, "metadata")
    crosswalk = _get_value(metadata, "evaluator_crosswalk")
    status = _get_value(crosswalk, "status")
    normalized = str(status).strip().lower() if status is not None else ""
    return normalized or None


def recall_at_k(
    ranked_papers: Sequence[Any],
    gold_papers: Sequence[Any],
    k: int,
    *,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> float:
    if k <= 0:
        return 0.0
    gold = _positive_gold_records(gold_papers, metric_version=metric_version)
    if not gold:
        return 0.0
    return len(
        _ranked_matches(
            ranked_papers,
            gold_papers,
            k=k,
            metric_version=metric_version,
        )
    ) / len(gold)


def precision_at_k(
    ranked_papers: Sequence[Any],
    gold_papers: Sequence[Any],
    k: int,
    *,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> float:
    """Precision@K keeps K as the denominator for backward compatibility."""

    if k <= 0 or not _positive_gold_records(
        gold_papers, metric_version=metric_version
    ):
        return 0.0
    return len(
        _ranked_matches(
            ranked_papers,
            gold_papers,
            k=k,
            metric_version=metric_version,
        )
    ) / k


def f1_at_k(
    ranked_papers: Sequence[Any],
    gold_papers: Sequence[Any],
    k: int,
    *,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> float:
    precision = precision_at_k(
        ranked_papers,
        gold_papers,
        k,
        metric_version=metric_version,
    )
    recall = recall_at_k(
        ranked_papers,
        gold_papers,
        k,
        metric_version=metric_version,
    )
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mrr(
    ranked_papers: Sequence[Any],
    gold_papers: Sequence[Any],
    *,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> float:
    matches = _ranked_matches(
        ranked_papers,
        gold_papers,
        metric_version=metric_version,
    )
    return 1.0 / matches[0].rank if matches else 0.0


def ndcg_at_k(
    ranked_papers: Sequence[Any],
    gold_papers: Sequence[Any],
    k: int,
    *,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> float:
    if k <= 0:
        return 0.0
    gold = _positive_gold_records(gold_papers, metric_version=metric_version)
    if not gold:
        return 0.0
    gain_by_rank = {
        match.rank: match.grade
        for match in _ranked_matches(
            ranked_papers,
            gold_papers,
            k=k,
            metric_version=metric_version,
        )
    }
    gains = [gain_by_rank.get(rank, 0.0) for rank in range(1, k + 1)]
    ideal_gains = sorted((record.grade for record in gold), reverse=True)[:k]
    idcg = _dcg(ideal_gains)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    ranked_papers: Sequence[Any],
    gold_papers: Sequence[Any],
    k_values: Sequence[int] = (5, 10, 20),
    *,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> EvalMetricSet:
    values = _normalize_k_values(k_values)
    return EvalMetricSet(
        metric_version=metric_version,
        recall_at_k={
            k: recall_at_k(
                ranked_papers,
                gold_papers,
                k,
                metric_version=metric_version,
            )
            for k in values
        },
        precision_at_k={
            k: precision_at_k(
                ranked_papers,
                gold_papers,
                k,
                metric_version=metric_version,
            )
            for k in values
        },
        f1_at_k={
            k: f1_at_k(
                ranked_papers,
                gold_papers,
                k,
                metric_version=metric_version,
            )
            for k in values
        },
        ndcg_at_k={
            k: ndcg_at_k(
                ranked_papers,
                gold_papers,
                k,
                metric_version=metric_version,
            )
            for k in values
        },
        mrr=mrr(ranked_papers, gold_papers, metric_version=metric_version),
    )


def zero_metric_set(
    k_values: Sequence[int] = (5, 10, 20),
    *,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> EvalMetricSet:
    values = _normalize_k_values(k_values)
    return EvalMetricSet(
        metric_version=metric_version,
        recall_at_k={k: 0.0 for k in values},
        precision_at_k={k: 0.0 for k in values},
        f1_at_k={k: 0.0 for k in values},
        ndcg_at_k={k: 0.0 for k in values},
    )


def average_metric_sets(metrics: Sequence[EvalMetricSet]) -> EvalMetricSet:
    if not metrics:
        return zero_metric_set()
    versions = {item.metric_version for item in metrics}
    if len(versions) != 1:
        raise ValueError(f"cannot average mixed metric versions:{sorted(versions)}")
    metric_version = next(iter(versions))
    count = len(metrics)
    raw_count = sum(item.raw_count for item in metrics)
    duplicate_count = sum(item.duplicate_count for item in metrics)
    source_call_count = sum(item.source_call_count for item in metrics)
    source_error_count = sum(item.source_error_count for item in metrics)
    return EvalMetricSet(
        metric_version=metric_version,
        recall_at_k=_average_maps([item.recall_at_k for item in metrics]),
        precision_at_k=_average_maps([item.precision_at_k for item in metrics]),
        f1_at_k=_average_maps([item.f1_at_k for item in metrics]),
        ndcg_at_k=_average_maps([item.ndcg_at_k for item in metrics]),
        mrr=sum(item.mrr for item in metrics) / count,
        raw_count=raw_count,
        deduplicated_count=sum(item.deduplicated_count for item in metrics),
        ranked_count=sum(item.ranked_count for item in metrics),
        duplicate_count=duplicate_count,
        duplicate_ratio=duplicate_count / raw_count if raw_count else 0.0,
        per_source_returned_count=_sum_source_counts(metrics),
        source_call_count=source_call_count,
        source_error_count=source_error_count,
        source_error_rate=(
            source_error_count / source_call_count if source_call_count else 0.0
        ),
        warning_count=sum(item.warning_count for item in metrics),
        query_warning_rate=sum(item.warning_count > 0 for item in metrics) / count,
        failed_case_count=sum(item.failed_case_count for item in metrics),
        failed_case_rate=sum(item.failed_case_count for item in metrics) / count,
    )


def aggregate_efficiency(
    cases: Sequence[EvalCaseEfficiency],
) -> EvalAggregateEfficiency:
    if not cases:
        return EvalAggregateEfficiency()
    count = len(cases)
    warnings = _dedupe_strings(
        warning for case in cases for warning in case.warnings
    )
    return EvalAggregateEfficiency(
        case_count=count,
        average_latency_seconds=sum(item.latency_seconds for item in cases) / count,
        avg_api_call_count=sum(item.api_call_count for item in cases) / count,
        avg_search_api_call_count=(
            sum(item.search_api_call_count for item in cases) / count
        ),
        avg_reference_api_call_count=(
            sum(item.reference_api_call_count for item in cases) / count
        ),
        avg_retry_count=sum(item.retry_count for item in cases) / count,
        avg_error_count=sum(item.error_count for item in cases) / count,
        avg_cache_hit_count=sum(item.cache_hit_count for item in cases) / count,
        avg_rate_limit_wait_seconds=(
            sum(item.rate_limit_wait_seconds for item in cases) / count
        ),
        avg_llm_call_count=sum(item.llm_call_count for item in cases) / count,
        avg_llm_total_tokens=sum(item.llm_total_tokens for item in cases) / count,
        total_llm_call_count=sum(item.llm_call_count for item in cases),
        total_llm_total_tokens=sum(item.llm_total_tokens for item in cases),
        average_search_rounds=sum(item.search_rounds for item in cases) / count,
        total_raw_count=sum(item.raw_count for item in cases),
        total_deduplicated_count=sum(item.deduplicated_count for item in cases),
        total_returned_result_count=sum(item.returned_result_count for item in cases),
        total_cache_hit_count=sum(item.cache_hit_count for item in cases),
        total_source_call_count=sum(item.source_call_count for item in cases),
        total_source_error_count=sum(item.source_error_count for item in cases),
        warnings=warnings,
    )


def candidate_count_metrics(
    raw_count: int,
    deduplicated_count: int,
    *,
    ranked_count: int | None = None,
    source_stats: Sequence[Any] | None = None,
) -> dict[str, Any]:
    raw = max(0, int(raw_count))
    deduplicated = max(0, int(deduplicated_count))
    duplicate_count = max(0, raw - deduplicated)
    per_source: dict[str, int] = {}
    for stat in source_stats or []:
        source = str(_get_value(stat, "source") or "unknown")
        per_source[source] = per_source.get(source, 0) + int(
            _get_value(stat, "returned_count") or 0
        )
    return {
        "raw_count": raw,
        "deduplicated_count": deduplicated,
        "ranked_count": max(0, int(ranked_count or 0)),
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_count / raw if raw else 0.0,
        "per_source_returned_count": per_source,
    }


def error_rate_metrics(
    source_stats: Sequence[Any] | None = None,
    warnings: Sequence[str] | None = None,
    *,
    failed_case_count: int = 0,
    total_case_count: int = 1,
    warning_case_count: int | None = None,
) -> dict[str, float | int]:
    stats = list(source_stats or [])
    source_error_count = sum(
        bool(str(_get_value(stat, "error_message") or "").strip()) for stat in stats
    )
    source_call_count = len(stats)
    warning_count = sum(bool(str(item).strip()) for item in warnings or [])
    total = max(0, int(total_case_count))
    warning_cases = (
        max(0, int(warning_case_count))
        if warning_case_count is not None
        else int(warning_count > 0)
    )
    failed = max(0, int(failed_case_count))
    return {
        "source_call_count": source_call_count,
        "source_error_count": source_error_count,
        "source_error_rate": source_error_count / source_call_count
        if source_call_count
        else 0.0,
        "warning_count": warning_count,
        "query_warning_rate": warning_cases / total if total else 0.0,
        "failed_case_count": failed,
        "failed_case_rate": failed / total if total else 0.0,
    }


def _ranked_matches(
    ranked_papers: Sequence[Any],
    gold_papers: Sequence[Any],
    *,
    k: int | None = None,
    metric_version: EvalMetricVersion = DEFAULT_METRIC_VERSION,
) -> list[_RankedMatch]:
    gold = _positive_gold_records(gold_papers, metric_version=metric_version)
    if not gold:
        return []
    limit = len(ranked_papers) if k is None else max(0, k)
    used_gold: set[int] = set()
    seen_prediction_profiles = []
    matches: list[_RankedMatch] = []
    for rank, paper in enumerate(ranked_papers[:limit], start=1):
        identifiers = paper_identifier_set(paper)
        title_year = paper_title_year_key(paper)
        profile = build_identity_profile(paper)
        if any(
            identity_evidence_from_profiles(profile, prior).equivalent
            for prior in seen_prediction_profiles
        ):
            continue
        seen_prediction_profiles.append(profile)

        for record in gold:
            if record.index in used_gold:
                continue
            evidence = identity_evidence_from_profiles(profile, record.profile)
            match_key = _match_key(evidence, title_year)
            if match_key is None:
                continue
            used_gold.add(record.index)
            matches.append(
                _RankedMatch(
                    rank=rank,
                    gold_index=record.index,
                    grade=record.grade,
                    match_key=match_key,
                )
            )
            break
    return matches


def _positive_gold_records(
    gold_papers: Sequence[Any],
    *,
    metric_version: EvalMetricVersion,
) -> list[_GoldRecord]:
    records = _raw_positive_gold_records(gold_papers)
    if metric_version == LEGACY_GOLD_METRIC_VERSION:
        return records
    if metric_version == DEDUPLICATED_GOLD_METRIC_VERSION:
        return _deduplicate_gold_records(records)
    raise ValueError(f"unsupported metric version:{metric_version}")


def _raw_positive_gold_records(gold_papers: Sequence[Any]) -> list[_GoldRecord]:
    records: list[_GoldRecord] = []
    for index, paper in enumerate(gold_papers):
        grade = _relevance_grade(paper)
        if grade <= 0:
            continue
        if gold_crosswalk_status(paper) in {"unavailable", "failed"}:
            continue
        identifiers = paper_identifier_set(paper)
        title_year = paper_title_year_key(paper) if not identifiers else None
        if not identifiers and title_year is None:
            continue
        profile = build_identity_profile(paper)
        records.append(
            _GoldRecord(
                index=index,
                paper=paper,
                profile=profile,
                member_indices=(index,),
                identifiers=frozenset(identifiers),
                title_year=title_year,
                grade=grade,
            )
        )
    return records


def _deduplicate_gold_records(records: Sequence[_GoldRecord]) -> list[_GoldRecord]:
    clusters: list[list[_GoldRecord]] = []
    for record in sorted(records, key=_gold_record_sort_key):
        matches: list[int] = []
        for cluster_index, cluster in enumerate(clusters):
            evidence = [
                identity_evidence_from_profiles(record.profile, item.profile)
                for item in cluster
            ]
            if any(item.conflicting_identifiers for item in evidence):
                continue
            if any(item.equivalent for item in evidence):
                matches.append(cluster_index)
        if not matches:
            clusters.append([record])
            continue
        target = min(matches, key=lambda index: _cluster_sort_key(clusters[index]))
        combined = [*clusters[target], record]
        merged_indices: list[int] = []
        for cluster_index in sorted(
            (index for index in matches if index != target),
            key=lambda index: _cluster_sort_key(clusters[index]),
        ):
            if _clusters_have_identifier_conflict(combined, clusters[cluster_index]):
                continue
            combined.extend(clusters[cluster_index])
            merged_indices.append(cluster_index)
        clusters[target] = combined
        for cluster_index in sorted(merged_indices, reverse=True):
            del clusters[cluster_index]
    merged = [_merge_gold_cluster(cluster) for cluster in clusters]
    return sorted(merged, key=lambda record: min(record.member_indices))


def _merge_gold_cluster(cluster: Sequence[_GoldRecord]) -> _GoldRecord:
    ordered = sorted(cluster, key=_gold_record_sort_key)
    representative = min(
        ordered,
        key=lambda item: (
            -item.grade,
            -len(item.profile.identifiers),
            _gold_record_sort_key(item),
        ),
    )
    profile = _merge_identity_profiles([item.profile for item in ordered])
    return _GoldRecord(
        index=representative.index,
        paper=representative.paper,
        profile=profile,
        member_indices=tuple(sorted(item.index for item in ordered)),
        identifiers=profile.identifiers,
        title_year=(
            f"title_year:{profile.title}:{profile.year}"
            if not profile.identifiers and profile.title and profile.year is not None
            else None
        ),
        grade=max(item.grade for item in ordered),
    )


def _merge_identity_profiles(profiles: Sequence[IdentityProfile]) -> IdentityProfile:
    field_names = [name for name, _value in profiles[0].field_values]
    field_values: list[tuple[str, str | None]] = []
    for field in field_names:
        values = {
            dict(profile.field_values).get(field)
            for profile in profiles
            if dict(profile.field_values).get(field)
        }
        if len(values) > 1:
            raise ValueError(f"conflicting gold identifiers entered one cluster:{field}")
        field_values.append((field, next(iter(values), None)))
    titles = sorted({profile.title for profile in profiles if profile.title})
    years = sorted({profile.year for profile in profiles if profile.year is not None})
    return IdentityProfile(
        identifiers=frozenset(
            identifier for profile in profiles for identifier in profile.identifiers
        ),
        field_values=tuple(field_values),
        title=titles[0] if titles else "",
        authors=frozenset(author for profile in profiles for author in profile.authors),
        year=years[0] if years else None,
    )


def _direct_identity_basis(
    duplicate: _GoldRecord,
    merged: _GoldRecord,
    by_index: Mapping[int, _GoldRecord],
) -> _GoldRecord:
    candidates = [
        by_index[index]
        for index in merged.member_indices
        if index != duplicate.index
        and identity_evidence_from_profiles(
            duplicate.profile, by_index[index].profile
        ).equivalent
    ]
    if not candidates:
        raise ValueError("deduplicated gold relation has no direct identity basis")
    return min(candidates, key=_gold_record_sort_key)


def _gold_record_sort_key(record: _GoldRecord) -> tuple[Any, ...]:
    return (
        _gold_profile_key(record.profile),
        -record.grade,
    )


def _cluster_sort_key(cluster: Sequence[_GoldRecord]) -> tuple[Any, ...]:
    return min(_gold_record_sort_key(record) for record in cluster)


def _clusters_have_identifier_conflict(
    left: Sequence[_GoldRecord],
    right: Sequence[_GoldRecord],
) -> bool:
    return any(
        identity_evidence_from_profiles(left_item.profile, right_item.profile)
        .conflicting_identifiers
        for left_item in left
        for right_item in right
    )


def _gold_profile_key(profile: IdentityProfile) -> str:
    return json.dumps(
        {
            "identifiers": sorted(profile.identifiers),
            "field_values": list(profile.field_values),
            "title": profile.title,
            "authors": sorted(profile.authors),
            "year": profile.year,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _match_key(
    evidence: Any,
    predicted_title: str | None,
) -> str | None:
    if not evidence.equivalent:
        return None
    if evidence.shared_identifiers:
        return _preferred_identifier(set(evidence.shared_identifiers))
    return predicted_title


def _preferred_identifier(identifiers: set[str]) -> str | None:
    for prefix in (
        "arxiv:",
        "doi:",
        "openalex:",
        "s2:",
        "s2orc:",
        "pubmed:",
    ):
        matches = sorted(item for item in identifiers if item.startswith(prefix))
        if matches:
            return matches[0]
    return sorted(identifiers)[0] if identifiers else None


def _relevance_grade(paper: Any) -> float:
    raw = _get_value(_unwrap_ranked_paper(paper), "relevance_grade")
    if raw is None:
        return 1.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(
        (math.pow(2.0, gain) - 1.0) / math.log2(index + 1)
        for index, gain in enumerate(gains, start=1)
        if gain > 0
    )


def _unwrap_ranked_paper(paper: Any) -> Any:
    if isinstance(paper, RankedPaper):
        return paper.paper
    if isinstance(paper, Mapping) and "paper" in paper:
        return paper["paper"]
    if not isinstance(paper, (Paper, EvalGoldPaper)) and hasattr(paper, "paper"):
        return getattr(paper, "paper")
    return paper


def _extract_identifier(paper: Any, name: str) -> str | None:
    value = _get_value(paper, name)
    if value:
        return str(value)
    value = _get_value(_get_value(paper, "identifiers"), name)
    return str(value) if value else None


def _get_value(item: Any, key: str) -> Any:
    if item is None:
        return None
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _overlay_values(paper: Any, values: Mapping[str, Any]) -> Any:
    if not any(value is not None for value in values.values()):
        return paper
    if isinstance(paper, Mapping):
        base = dict(paper)
    elif hasattr(paper, "model_dump"):
        base = paper.model_dump()
    else:
        base = {
            key: _get_value(paper, key)
            for key in (
                "doi",
                "arxiv_id",
                "openalex_id",
                "semantic_scholar_id",
                "s2orc_corpus_id",
                "pubmed_id",
                "title",
                "year",
                "identifiers",
            )
            if _get_value(paper, key) is not None
        }
    for key, value in values.items():
        if value is not None:
            base[key] = value
    if not base:
        return paper
    return base


def _normalize_doi(value: str) -> str:
    return shared_normalize_doi(value) or ""


def _arxiv_id_from_doi(normalized_doi: str) -> str | None:
    match = re.fullmatch(r"10\.48550/arxiv\.(.+)", normalized_doi)
    return _normalize_arxiv_id(match.group(1)) if match else None


def _normalize_arxiv_id(value: str) -> str:
    return shared_normalize_arxiv_id(value) or ""


def _normalize_openalex_id(value: str) -> str:
    return shared_normalize_simple_id(value) or ""


def _normalize_semantic_scholar_id(value: str) -> str:
    return shared_normalize_simple_id(value) or ""


def _normalize_pubmed_id(value: str) -> str:
    return shared_normalize_simple_id(value) or ""


def _normalize_title(value: str) -> str:
    return shared_normalize_title(value)


def _normalize_k_values(values: Sequence[int]) -> list[int]:
    normalized = sorted({int(value) for value in values if int(value) > 0})
    return normalized or [5, 10, 20]


def _average_maps(values: Sequence[dict[int, float]]) -> dict[int, float]:
    keys = sorted({key for item in values for key in item})
    return {
        key: sum(item.get(key, 0.0) for item in values) / len(values)
        for key in keys
    }


def _sum_source_counts(metrics: Sequence[EvalMetricSet]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for metric in metrics:
        for source, count in metric.per_source_returned_count.items():
            totals[source] = totals.get(source, 0) + count
    return totals


def _dedupe_strings(values: Sequence[str] | Any) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output
