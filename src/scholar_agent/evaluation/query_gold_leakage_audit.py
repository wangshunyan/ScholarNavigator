"""Deterministic offline query-to-gold information-leakage audit.

Gold is loaded only in this evaluator-side module.  Detection rules come from
the versioned protocol and are never influenced by retrieval outcomes.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import socket
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import build_identity_profile
from scholar_agent.evaluation.current_rules_regression import compare_profiles
from scholar_agent.evaluation.datasets.auto_scholar_query import (
    load_auto_scholar_query,
)
from scholar_agent.evaluation.datasets.beir_scifact import (
    load_beir_scifact_enriched,
)
from scholar_agent.evaluation.metrics import canonical_paper_id


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "1"
AUDIT_NAME = "autoscholar_query_gold_leakage_v1"
GATE_NAME = "autoscholar_query_gold_leakage_regression"
LEVELS = (
    "identifier_or_url_exact",
    "quoted_title_exact",
    "normalized_title_full",
    "title_token_high_coverage",
    "no_detected_leakage",
)
DEFAULT_SNAPSHOT_ROOT = REPOSITORY_ROOT / "outputs" / "benchmark_snapshots"
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_PUNCTUATION_RE = re.compile(r"[\W_]+", re.UNICODE)


class QueryGoldLeakageAuditError(RuntimeError):
    """Raised when a frozen leakage-audit contract is invalid."""


def detect_query_gold_leakage(
    query: str,
    gold: EvalGoldPaper | Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one query-gold relation using only preregistered lexical rules."""

    _validate_protocol(protocol)
    paper = gold if isinstance(gold, EvalGoldPaper) else EvalGoldPaper.model_validate(gold)
    normalized_query = normalize_leakage_text(query)
    normalized_title = normalize_leakage_text(paper.title or "")
    thresholds = protocol["thresholds"]
    title_tokens = _tokens(normalized_title)
    title_eligible = (
        len(normalized_title)
        >= int(thresholds["minimum_normalized_title_characters"])
        and len(title_tokens) >= int(thresholds["minimum_normalized_title_tokens"])
    )
    rule_hits: list[dict[str, Any]] = []
    identifier_hit = _identifier_hit(query, paper, protocol)
    if identifier_hit is not None:
        rule_hits.append(identifier_hit)

    quoted_spans = _quoted_spans(query, protocol["evidence"]["quoted_span_pairs"])
    if title_eligible:
        matching_quote = next(
            (
                span
                for span in quoted_spans
                if normalize_leakage_text(span) == normalized_title
            ),
            None,
        )
        if matching_quote is not None:
            rule_hits.append(
                {
                    "rule": "quoted_title_exact",
                    "evidence_snippet": matching_quote,
                    "normalized_title": normalized_title,
                }
            )
        if _contains_token_sequence(normalized_query, normalized_title):
            rule_hits.append(
                {
                    "rule": "normalized_title_full",
                    "evidence_snippet": normalized_title,
                    "normalized_title": normalized_title,
                }
            )

        stopwords = {str(item) for item in protocol["stopwords"]}
        informative = _ordered_unique(
            token for token in title_tokens if token not in stopwords
        )
        query_token_set = set(_tokens(normalized_query))
        matched = [token for token in informative if token in query_token_set]
        coverage = len(matched) / len(informative) if informative else 0.0
        if (
            len(informative)
            >= int(thresholds["high_coverage_minimum_informative_title_tokens"])
            and coverage >= float(thresholds["high_coverage_ratio"])
        ):
            rule_hits.append(
                {
                    "rule": "title_token_high_coverage",
                    "coverage": coverage,
                    "matched_informative_tokens": matched,
                    "informative_title_token_count": len(informative),
                    "normalized_title": normalized_title,
                }
            )

    hit_rules = {str(item["rule"]) for item in rule_hits}
    terminal = next((level for level in LEVELS[:-1] if level in hit_rules), LEVELS[-1])
    return {
        "leakage_level": terminal,
        "rule_hits": rule_hits,
        "title_diagnostics": {
            "normalized_title": normalized_title,
            "normalized_character_count": len(normalized_title),
            "normalized_token_count": len(title_tokens),
            "title_rules_eligible": title_eligible,
            "skip_reason": None if title_eligible else "short_title_protection",
        },
    }


def build_query_gold_leakage_audit(
    manifest: Mapping[str, Any],
    *,
    auto_queries: Sequence[EvalQuery] | None = None,
    scifact_queries: Sequence[EvalQuery] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the complete 1000-query/2403-relation audit and frozen strata."""

    _validate_manifest(manifest, require_baseline=False)
    protocol_path = _repo_path(manifest["protocol"]["path"])
    _validate_hash(protocol_path, manifest["protocol"]["sha256"])
    protocol = _read_json(protocol_path)
    _validate_protocol(protocol)
    snapshot_before = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    attempts = {"network": 0}
    with _forbid_network(attempts):
        loaded_auto = list(auto_queries) if auto_queries is not None else load_auto_scholar_query(
            _repo_path(manifest["dataset"]["path"])
        )
        identity_rows = _read_jsonl(
            _repo_path(manifest["gold_identity_baseline"]["path"])
        )
        relations, queries = _audit_auto_relations(
            loaded_auto, identity_rows, protocol
        )
        loaded_scifact = (
            list(scifact_queries)
            if scifact_queries is not None
            else load_beir_scifact_enriched(
                _repo_path(manifest["scifact"]["dataset_path"]),
                crosswalk_path=_repo_path(manifest["scifact"]["crosswalk_path"]),
            )
        )
        scifact_relations = _classify_external_relations(loaded_scifact, protocol)
        frozen_sets = _frozen_set_diagnostics(
            manifest,
            relations,
            queries,
            scifact_relations,
            loaded_scifact,
        )
    snapshot_after = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    execution = {
        "network_request_count": attempts["network"],
        "llm_request_count": 0,
        "snapshot_write_count": int(snapshot_before != snapshot_after),
        "retrieval_invoked": False,
        "effectiveness_metrics_recomputed": False,
        "gold_scope": "offline_evaluator_validity_audit_only",
    }
    if any(
        int(execution[field])
        for field in ("network_request_count", "llm_request_count", "snapshot_write_count")
    ):
        raise QueryGoldLeakageAuditError(f"offline execution invariant failed:{execution}")
    summary = _summarize(
        relations,
        queries,
        frozen_sets=frozen_sets,
        protocol=protocol,
        execution=execution,
    )
    _validate_closure(relations, queries, summary, manifest)
    return relations, queries, summary


def write_query_gold_leakage_audit(
    output_dir: str | Path,
    relations: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "relations.jsonl", relations)
    _write_jsonl(root / "queries.jsonl", queries)
    _write_json(root / "summary.json", summary)


def check_query_gold_leakage_regression(
    manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Rebuild the audit and report minimal deterministic baseline drift."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    drifts: list[dict[str, Any]] = []
    try:
        _validate_manifest(manifest, require_baseline=True)
        relations, queries, summary = build_query_gold_leakage_audit(manifest)
        observed = {
            "relations": relations,
            "queries": queries,
            "summary": summary,
        }
        baseline = {
            "relations": _read_jsonl(_repo_path(manifest["baseline"]["relations_path"])),
            "queries": _read_jsonl(_repo_path(manifest["baseline"]["queries_path"])),
            "summary": _read_json(_repo_path(manifest["baseline"]["summary_path"])),
        }
        for name in ("relations", "queries", "summary"):
            expected_hash = str(manifest["baseline"][f"{name}_sha256"])
            path = _repo_path(manifest["baseline"][f"{name}_path"])
            if sha256_file(path) != expected_hash:
                drifts.append(
                    {
                        "kind": "baseline_fingerprint_drift",
                        "path": f"$.baseline.{name}",
                        "expected": expected_hash,
                        "observed": sha256_file(path),
                    }
                )
            drifts.extend(
                compare_profiles(
                    {name: baseline[name]},
                    {name: observed[name]},
                    max_diffs=100,
                )
            )
        write_query_gold_leakage_audit(output / "observed", relations, queries, summary)
    except (QueryGoldLeakageAuditError, ValueError) as exc:
        drifts.append(
            {
                "kind": "input_or_protocol_drift",
                "path": "$",
                "expected": "frozen audit contract",
                "observed": str(exc),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "passed": not drifts,
        "drift_count": len(drifts),
        "drifts": drifts,
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
        },
    }
    _write_json(output / "regression_report.json", report)
    return report


def normalize_leakage_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    return " ".join(_PUNCTUATION_RE.sub(" ", text).split())


def _audit_auto_relations(
    queries: Sequence[EvalQuery],
    identity_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identity_by_relation = {str(item["relation_id"]): item for item in identity_rows}
    relations: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for query_order, query in enumerate(queries):
        levels: Counter[str] = Counter()
        relation_ids: list[str] = []
        for gold_index, gold in enumerate(query.gold_papers):
            relation_id = f"{query.query_id}::gold[{gold_index}]"
            identity = identity_by_relation.get(relation_id)
            if identity is None:
                raise QueryGoldLeakageAuditError(f"missing identity baseline:{relation_id}")
            detected = detect_query_gold_leakage(query.query, gold, protocol)
            levels[detected["leakage_level"]] += 1
            relation_ids.append(relation_id)
            relations.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "relation_id": relation_id,
                    "query_order": query_order,
                    "query_id": query.query_id,
                    "gold_index": gold_index,
                    "canonical_identity": canonical_paper_id(gold),
                    "identity_cluster_id": identity["identity_cluster_id"],
                    "duplicate_across_queries": bool(identity["duplicate_across_queries"]),
                    "identity_cluster_query_count": int(identity["identity_cluster_query_count"]),
                    **detected,
                }
            )
        highest = next((level for level in LEVELS[:-1] if levels[level]), LEVELS[-1])
        query_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "query_order": query_order,
                "query_id": query.query_id,
                "gold_relation_count": len(query.gold_papers),
                "relation_ids": relation_ids,
                "leakage_level": highest,
                "leakage_level_counts": {
                    level: levels[level] for level in LEVELS
                },
                "has_detected_leakage": highest != LEVELS[-1],
            }
        )
    if set(identity_by_relation) != {str(item["relation_id"]) for item in relations}:
        raise QueryGoldLeakageAuditError("identity baseline relation set drift")
    return relations, query_rows


def _classify_external_relations(
    queries: Sequence[EvalQuery], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        for gold_index, gold in enumerate(query.gold_papers):
            rows.append(
                {
                    "query_id": query.query_id,
                    "gold_index": gold_index,
                    "canonical_identity": canonical_paper_id(gold),
                    "stable_identifiers": sorted(build_identity_profile(gold).identifiers),
                    **detect_query_gold_leakage(query.query, gold, protocol),
                }
            )
    return rows


def _frozen_set_diagnostics(
    manifest: Mapping[str, Any],
    auto_relations: Sequence[Mapping[str, Any]],
    auto_queries: Sequence[Mapping[str, Any]],
    scifact_relations: Sequence[Mapping[str, Any]],
    scifact_queries: Sequence[EvalQuery],
) -> dict[str, Any]:
    auto_relation_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in auto_relations:
        auto_relation_by_query[str(row["query_id"])].append(row)
    auto_query_by_id = {str(item["query_id"]): item for item in auto_queries}
    small_rows = _read_jsonl(_repo_path(manifest["frozen_replays"]["small_sets"]["path"]))
    record_rows = _read_jsonl(_repo_path(manifest["frozen_replays"]["record160"]["path"]))
    diagnostics: dict[str, Any] = {}
    for label in ("auto_dev", "auto_val"):
        cases = [row for row in small_rows if row["dataset"] == label]
        diagnostics[label] = _summarize_frozen_cases(
            cases, auto_relation_by_query, auto_query_by_id
        )
    record_cases = [
        row for row in record_rows if row.get("included_main_analysis") is True
    ]
    diagnostics["autoscholar_record160"] = _summarize_frozen_cases(
        record_cases, auto_relation_by_query, auto_query_by_id
    )
    diagnostics["autoscholar_record160"]["excluded_no_success_case_count"] = sum(
        row.get("analysis_status") == "excluded_no_successful_source"
        for row in record_rows
    )
    scifact_by_query: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scifact_relations:
        scifact_by_query[str(row["query_id"])].append(row)
    scifact_query_rows = {
        query.query_id: {
            "query_id": query.query_id,
            "has_detected_leakage": any(
                item["leakage_level"] != LEVELS[-1]
                for item in scifact_by_query[query.query_id]
            ),
        }
        for query in scifact_queries
    }
    scifact_cases = [row for row in small_rows if row["dataset"] == "scifact"]
    diagnostics["scifact"] = _summarize_frozen_cases(
        scifact_cases, scifact_by_query, scifact_query_rows
    )
    return diagnostics


def _summarize_frozen_cases(
    cases: Sequence[Mapping[str, Any]],
    relations_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
    query_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    query_counts = Counter()
    matched_relation_levels = Counter()
    unmatched_frozen_ids = 0
    for case in cases:
        query_id = str(case["case_id"])
        query = query_by_id.get(query_id)
        if query is None:
            raise QueryGoldLeakageAuditError(f"frozen case missing leakage row:{query_id}")
        leaked = bool(query["has_detected_leakage"])
        suffix = "leaked" if leaked else "non_leaked"
        query_counts[f"query_{suffix}"] += 1
        if int(case.get("candidate_gold_count") or 0) > 0:
            query_counts[f"candidate_hit_query_{suffix}"] += 1
        matched = [str(item) for item in case.get("baseline", {}).get("matched_gold_ids", [])]
        if matched:
            query_counts[f"final_hit_query_{suffix}"] += 1
        relation_rows = list(relations_by_query.get(query_id, []))
        for matched_id in matched:
            relation = next(
                (row for row in relation_rows if _relation_matches_id(row, matched_id)),
                None,
            )
            if relation is None:
                unmatched_frozen_ids += 1
            else:
                matched_relation_levels[str(relation["leakage_level"])] += 1
    return {
        "case_count": len(cases),
        "query_counts": dict(sorted(query_counts.items())),
        "final_matched_relation_level_counts": {
            level: matched_relation_levels[level] for level in LEVELS
        },
        "final_matched_relation_count": sum(matched_relation_levels.values()),
        "unmatched_frozen_gold_id_count": unmatched_frozen_ids,
        "metric_scope": "descriptive frozen-hit stratification; no Recall/F1 recomputation",
    }


def _relation_matches_id(relation: Mapping[str, Any], matched_id: str) -> bool:
    target = str(matched_id).casefold()
    values = {str(relation.get("canonical_identity") or "").casefold()}
    values.update(str(item).casefold() for item in relation.get("stable_identifiers") or [])
    return target in values


def _summarize(
    relations: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    *,
    frozen_sets: Mapping[str, Any],
    protocol: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    relation_counts = Counter(str(row["leakage_level"]) for row in relations)
    query_counts = Counter(str(row["leakage_level"]) for row in queries)
    leaked_relations = [row for row in relations if row["leakage_level"] != LEVELS[-1]]
    leaked_queries = [row for row in queries if row["has_detected_leakage"]]
    duplicate_rows = [row for row in relations if row["duplicate_across_queries"]]
    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in relations:
        by_cluster[str(row["identity_cluster_id"])].append(row)
    repeated_clusters = [
        rows
        for rows in by_cluster.values()
        if len({row["query_id"] for row in rows}) > 1
    ]
    mixed_clusters = [
        rows
        for rows in repeated_clusters
        if len({row["leakage_level"] != LEVELS[-1] for row in rows}) > 1
    ]
    query_rate = len(leaked_queries) / len(queries) if queries else 0.0
    direct_query_count = sum(
        row["leakage_level"] in LEVELS[:3] for row in queries
    )
    direct_rate = direct_query_count / len(queries) if queries else 0.0
    if direct_rate >= 0.01 or query_rate >= 0.05:
        risk = "high"
    elif query_rate >= 0.01:
        risk = "moderate"
    else:
        risk = "low"
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": AUDIT_NAME,
        "protocol_version": protocol["version"],
        "relation_count": len(relations),
        "query_count": len(queries),
        "relation_level_counts": {level: relation_counts[level] for level in LEVELS},
        "query_level_counts": {level: query_counts[level] for level in LEVELS},
        "leaked_relation_count": len(leaked_relations),
        "leaked_relation_rate": len(leaked_relations) / len(relations) if relations else 0.0,
        "leaked_query_count": len(leaked_queries),
        "leaked_query_rate": query_rate,
        "direct_leak_query_count": direct_query_count,
        "direct_leak_query_rate": direct_rate,
        "validity_risk_band": risk,
        "duplicates": {
            "cross_query_duplicate_relation_count": len(duplicate_rows),
            "cross_query_repeated_identity_count": len(repeated_clusters),
            "mixed_leakage_repeated_identity_count": len(mixed_clusters),
            "leaked_cross_query_duplicate_relation_count": sum(
                row["leakage_level"] != LEVELS[-1] for row in duplicate_rows
            ),
        },
        "frozen_set_diagnostics": dict(frozen_sets),
        "execution": dict(execution),
        "interpretation": {
            "internal_metrics_only": True,
            "non_leak_subset_replaces_full_results": False,
            "automatic_data_removal": False,
            "risk_band_uses_preregistered_thresholds": True,
        },
    }


def _identifier_hit(
    query: str, gold: EvalGoldPaper, protocol: Mapping[str, Any]
) -> dict[str, Any] | None:
    normalized_query = unicodedata.normalize("NFKC", query).casefold()
    profile = build_identity_profile(gold)
    for identifier in sorted(profile.identifiers):
        prefix, value = identifier.split(":", 1)
        forms = _identifier_forms(prefix, value)
        for form in forms:
            match = (
                _bounded_regex_find(
                    normalized_query,
                    re.escape(form) + r"(?:v\d+)?(?:\.pdf)?",
                )
                if prefix == "arxiv"
                else _bounded_find(normalized_query, form)
            )
            if match is not None:
                context = int(protocol["evidence"]["identifier_snippet_context_characters"])
                start, end = match
                return {
                    "rule": "identifier_or_url_exact",
                    "identifier_type": prefix,
                    "matched_form": normalized_query[start:end],
                    "evidence_snippet": normalized_query[
                        max(0, start - context) : min(len(normalized_query), end + context)
                    ],
                }
    return None


def _identifier_forms(prefix: str, value: str) -> tuple[str, ...]:
    if prefix == "arxiv":
        return (
            f"arxiv:{value}",
            f"arxiv {value}",
            f"arxiv.org/abs/{value}",
            f"arxiv.org/pdf/{value}",
            f"10.48550/arxiv.{value}",
            value,
        )
    if prefix == "doi":
        return (f"doi:{value}", f"doi {value}", f"doi.org/{value}", value)
    if prefix == "pubmed":
        return (f"pmid:{value}", f"pmid {value}", f"pubmed/{value}")
    if prefix == "openalex":
        return (f"openalex:{value}", f"openalex.org/{value}")
    if prefix == "s2":
        return (f"s2:{value}", f"semanticscholar.org/paper/{value}")
    if prefix == "s2orc":
        return (f"corpusid:{value}", f"s2orc:{value}")
    return (f"{prefix}:{value}",)


def _bounded_find(text: str, value: str) -> tuple[int, int] | None:
    start = text.find(value)
    while start >= 0:
        end = start + len(value)
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if (not before or not before.isalnum()) and (not after or not after.isalnum()):
            return start, end
        start = text.find(value, start + 1)
    return None


def _bounded_regex_find(text: str, pattern: str) -> tuple[int, int] | None:
    for match in re.finditer(pattern, text):
        start, end = match.span()
        before = text[start - 1] if start else ""
        after = text[end] if end < len(text) else ""
        if (not before or not before.isalnum()) and (not after or not after.isalnum()):
            return start, end
    return None


def _quoted_spans(query: str, pairs: Sequence[str]) -> list[str]:
    spans: list[str] = []
    for pair in pairs:
        if len(pair) != 2:
            raise QueryGoldLeakageAuditError("quote pair must contain two characters")
        start, end = pair
        pattern = re.compile(re.escape(start) + r"(.+?)" + re.escape(end), re.DOTALL)
        spans.extend(match.group(1) for match in pattern.finditer(query))
    return spans


def _contains_token_sequence(text: str, phrase: str) -> bool:
    return bool(phrase and f" {phrase} " in f" {text} ")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _ordered_unique(values: Sequence[str] | Iterator[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("audit") != AUDIT_NAME or protocol.get("version") != "query-gold-leakage-protocol-v1":
        raise QueryGoldLeakageAuditError("unexpected leakage protocol")
    if tuple(protocol.get("classification", {}).get("mutually_exclusive_priority") or []) != LEVELS:
        raise QueryGoldLeakageAuditError("leakage classification priority drift")
    thresholds = protocol.get("thresholds") or {}
    if (
        int(thresholds.get("minimum_normalized_title_characters", 0)) != 24
        or int(thresholds.get("minimum_normalized_title_tokens", 0)) != 3
        or int(thresholds.get("high_coverage_minimum_informative_title_tokens", 0)) != 5
        or float(thresholds.get("high_coverage_ratio", 0.0)) != 0.8
    ):
        raise QueryGoldLeakageAuditError("leakage threshold drift")
    execution = protocol.get("execution") or {}
    if any(
        int(execution.get(field, -1)) != 0
        for field in ("network_request_count", "llm_request_count", "snapshot_write_count")
    ):
        raise QueryGoldLeakageAuditError("leakage protocol must be zero-I/O")


def _validate_manifest(manifest: Mapping[str, Any], *, require_baseline: bool) -> None:
    if manifest.get("audit") != AUDIT_NAME or manifest.get("gate") != GATE_NAME:
        raise QueryGoldLeakageAuditError("unexpected leakage audit manifest")
    if int(manifest.get("dataset", {}).get("case_count", 0)) != 1000:
        raise QueryGoldLeakageAuditError("query count contract drift")
    if int(manifest.get("dataset", {}).get("gold_relation_count", 0)) != 2403:
        raise QueryGoldLeakageAuditError("gold relation count contract drift")
    for spec in (
        manifest["dataset"],
        manifest["gold_identity_baseline"],
        manifest["implementation"],
        manifest["protocol"],
        manifest["frozen_replays"]["small_sets"],
        manifest["frozen_replays"]["record160"],
        manifest["scifact"]["dataset"],
        manifest["scifact"]["crosswalk"],
    ):
        _validate_hash(_repo_path(spec["path"]), spec["sha256"])
    if require_baseline and "baseline" not in manifest:
        raise QueryGoldLeakageAuditError("leakage baseline is missing")


def _validate_closure(
    relations: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if len(relations) != int(manifest["dataset"]["gold_relation_count"]):
        raise QueryGoldLeakageAuditError("gold relation closure failure")
    if len(queries) != int(manifest["dataset"]["case_count"]):
        raise QueryGoldLeakageAuditError("query closure failure")
    if sum(summary["relation_level_counts"].values()) != len(relations):
        raise QueryGoldLeakageAuditError("relation terminal counts do not close")
    if sum(summary["query_level_counts"].values()) != len(queries):
        raise QueryGoldLeakageAuditError("query terminal counts do not close")
    if len({row["relation_id"] for row in relations}) != len(relations):
        raise QueryGoldLeakageAuditError("duplicate relation ID")


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _validate_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != str(expected):
        raise QueryGoldLeakageAuditError(f"frozen input hash drift:{path.name}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_signature(root: Path) -> str | None:
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise QueryGoldLeakageAuditError("network access is forbidden")

    with patch.object(socket, "create_connection", blocked), patch.object(
        socket.socket, "connect", blocked
    ):
        yield


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
