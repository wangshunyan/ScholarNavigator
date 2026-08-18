"""Offline migration audit for deduplicated-gold metric semantics.

This module reads frozen evaluator inputs and retrieval Snapshots only. It never
imports SearchService, issues retrieval/LLM requests, or mutates Snapshot data.
"""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from scholar_agent.core.evaluation_schemas import (
    DEDUPLICATED_GOLD_METRIC_VERSION,
    LEGACY_GOLD_METRIC_VERSION,
    EvalQuery,
)
from scholar_agent.evaluation.current_rules_regression import (
    build_current_rules_profile,
    compare_profiles,
)
from scholar_agent.evaluation.datasets.auto_scholar_query import (
    load_auto_scholar_query,
)
from scholar_agent.evaluation.metrics import gold_deduplication_audit


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "1"
GATE_NAME = "deduplicated_gold_metric_semantics_regression"
TERMINAL_METRICS = ("candidate_recall", "recall_at_20", "f1_at_20")


class GoldMetricSemanticsError(RuntimeError):
    """Raised when the frozen metric migration contract is malformed."""


def build_full_gold_denominator_audit(
    queries: Sequence[EvalQuery],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return per-query denominators and every removed duplicate relation."""

    query_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for query_order, query in enumerate(queries):
        audit = gold_deduplication_audit(query.gold_papers)
        for relation in audit["duplicate_relations"]:
            duplicate_index = int(relation["duplicate_index"])
            retained_index = int(relation["retained_index"])
            basis_index = int(relation["identity_basis_index"])
            duplicate_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "query_order": query_order,
                    "query_id": query.query_id,
                    "duplicate_relation_id": _relation_id(
                        query.query_id, duplicate_index
                    ),
                    "retained_relation_id": _relation_id(
                        query.query_id, retained_index
                    ),
                    "identity_basis_relation_id": _relation_id(
                        query.query_id, basis_index
                    ),
                    "duplicate_input_sha256": _sha256_json(
                        query.gold_papers[duplicate_index].model_dump(mode="json")
                    ),
                    "retained_input_sha256": _sha256_json(
                        query.gold_papers[retained_index].model_dump(mode="json")
                    ),
                    "rule": relation["rule"],
                    "shared_identifiers": relation["shared_identifiers"],
                    "conflicting_identifiers": relation[
                        "conflicting_identifiers"
                    ],
                    "title": relation["title"],
                    "author_overlap": relation["author_overlap"],
                    "year": relation["year"],
                    "retained_identity": relation["retained_identity"],
                    "denominator_delta": -1,
                }
            )
        query_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "query_order": query_order,
                "query_id": query.query_id,
                "legacy_evaluable_gold_count": audit[
                    "legacy_evaluable_gold_count"
                ],
                "deduplicated_evaluable_gold_count": audit[
                    "deduplicated_evaluable_gold_count"
                ],
                "duplicate_relation_count": audit["duplicate_relation_count"],
                "denominator_delta": audit["deduplicated_evaluable_gold_count"]
                - audit["legacy_evaluable_gold_count"],
            }
        )
    legacy_count = sum(row["legacy_evaluable_gold_count"] for row in query_rows)
    deduplicated_count = sum(
        row["deduplicated_evaluable_gold_count"] for row in query_rows
    )
    summary = {
        "query_count": len(query_rows),
        "legacy_evaluable_gold_count": legacy_count,
        "deduplicated_evaluable_gold_count": deduplicated_count,
        "duplicate_relation_count": len(duplicate_rows),
        "impacted_query_count": sum(
            bool(row["duplicate_relation_count"]) for row in query_rows
        ),
        "count_closed": legacy_count - deduplicated_count == len(duplicate_rows),
    }
    if not summary["count_closed"]:
        raise GoldMetricSemanticsError("full gold denominator audit is not closed")
    return query_rows, duplicate_rows, summary


def compare_frozen_metric_profiles(
    legacy: Mapping[str, Any],
    deduplicated: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare profiles while requiring candidates and source terminals to match."""

    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    legacy_datasets = legacy.get("datasets") or {}
    deduplicated_datasets = deduplicated.get("datasets") or {}
    if set(legacy_datasets) != set(deduplicated_datasets):
        raise GoldMetricSemanticsError("frozen dataset set changed between versions")
    for label in legacy_datasets:
        before_dataset = legacy_datasets[label]
        after_dataset = deduplicated_datasets[label]
        before_cases = before_dataset.get("cases") or {}
        after_cases = after_dataset.get("cases") or {}
        if set(before_cases) != set(after_cases):
            raise GoldMetricSemanticsError(f"case set changed:{label}")
        impacted = 0
        candidate_parity = True
        returned_parity = True
        terminal_parity = True
        for case_id in before_cases:
            before = before_cases[case_id]
            after = after_cases[case_id]
            case_candidate_parity = (
                before.get("candidate_identities")
                == after.get("candidate_identities")
            )
            case_returned_parity = (
                before.get("returned_identities") == after.get("returned_identities")
            )
            case_terminal_parity = (
                before.get("source_terminals") == after.get("source_terminals")
                and before.get("required_retrieval_keys")
                == after.get("required_retrieval_keys")
            )
            if not (case_candidate_parity and case_returned_parity and case_terminal_parity):
                raise GoldMetricSemanticsError(
                    f"non-metric frozen state drift:{label}:{case_id}"
                )
            before_metrics = before.get("metrics") or {}
            after_metrics = after.get("metrics") or {}
            denominator_delta = int(
                after_metrics.get("evaluable_gold_count") or 0
            ) - int(before_metrics.get("evaluable_gold_count") or 0)
            impacted += int(denominator_delta != 0)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": label,
                    "case_id": case_id,
                    "status": before.get("status"),
                    "legacy_evaluable_gold_count": before_metrics.get(
                        "evaluable_gold_count"
                    ),
                    "deduplicated_evaluable_gold_count": after_metrics.get(
                        "evaluable_gold_count"
                    ),
                    "duplicate_relation_count": -denominator_delta,
                    "legacy": {
                        name: before_metrics.get(name) for name in TERMINAL_METRICS
                    },
                    "deduplicated": {
                        name: after_metrics.get(name) for name in TERMINAL_METRICS
                    },
                    "delta": {
                        name: _optional_delta(
                            before_metrics.get(name), after_metrics.get(name)
                        )
                        for name in TERMINAL_METRICS
                    },
                    "candidate_identity_parity": case_candidate_parity,
                    "returned_identity_parity": case_returned_parity,
                    "source_terminal_parity": case_terminal_parity,
                }
            )
            candidate_parity &= case_candidate_parity
            returned_parity &= case_returned_parity
            terminal_parity &= case_terminal_parity
        before_summary = before_dataset.get("summary_metrics") or {}
        after_summary = after_dataset.get("summary_metrics") or {}
        summaries[label] = {
            "case_count": before_summary.get("case_count"),
            "evaluable_case_count": before_summary.get("evaluable_case_count"),
            "impacted_query_count": impacted,
            "legacy": {
                name: before_summary.get(name) for name in TERMINAL_METRICS
            },
            "deduplicated": {
                name: after_summary.get(name) for name in TERMINAL_METRICS
            },
            "delta": {
                name: _optional_delta(
                    before_summary.get(name), after_summary.get(name)
                )
                for name in TERMINAL_METRICS
            },
            "candidate_identity_parity": candidate_parity,
            "returned_identity_parity": returned_parity,
            "source_terminal_parity": terminal_parity,
        }
    return rows, summaries


def build_gold_metric_semantics_audit(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the complete migration audit with strict offline guards."""

    _validate_manifest(manifest, require_baseline=False)
    identity_manifest_path = _repo_path(manifest["inputs"]["identity_manifest_path"])
    regression_manifest_path = _repo_path(
        manifest["inputs"]["current_rules_manifest_path"]
    )
    identity_manifest = _read_json(identity_manifest_path)
    regression_manifest = _read_json(regression_manifest_path)
    snapshot_roots = [
        _repo_path(spec["snapshot_dir"])
        for spec in regression_manifest["datasets"]
    ]
    before = {str(path): _tree_sha256(path) for path in snapshot_roots}
    attempts = {"network": 0}
    with _forbid_network(attempts):
        queries = load_auto_scholar_query(
            _repo_path(identity_manifest["dataset"]["path"])
        )
        query_rows, duplicate_rows, full_summary = build_full_gold_denominator_audit(
            queries
        )
        legacy_profile = build_current_rules_profile(
            regression_manifest,
            metric_version=LEGACY_GOLD_METRIC_VERSION,
        )
        deduplicated_profile = build_current_rules_profile(
            regression_manifest,
            metric_version=DEDUPLICATED_GOLD_METRIC_VERSION,
        )
        replay_rows, replay_summaries = compare_frozen_metric_profiles(
            legacy_profile,
            deduplicated_profile,
        )
    after = {str(path): _tree_sha256(path) for path in snapshot_roots}
    snapshot_write_count = sum(before[path] != after[path] for path in before)
    execution = {
        "network_request_count": attempts["network"],
        "llm_request_count": 0,
        "snapshot_write_count": snapshot_write_count,
        "retrieval_invoked": False,
        "snapshot_mode": "read_only",
    }
    if any(
        execution[name]
        for name in (
            "network_request_count",
            "llm_request_count",
            "snapshot_write_count",
        )
    ):
        raise GoldMetricSemanticsError(f"offline invariant failed:{execution}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "metric_versions": {
            "legacy": LEGACY_GOLD_METRIC_VERSION,
            "current_internal": DEDUPLICATED_GOLD_METRIC_VERSION,
        },
        "full_autoscholar": full_summary,
        "frozen_replay": replay_summaries,
        "execution": execution,
        "historical_artifacts_mutated": False,
        "official_scorer": False,
        "input_hashes": _input_hashes(manifest),
    }
    _validate_closure(query_rows, duplicate_rows, replay_rows, summary)
    return query_rows, duplicate_rows, replay_rows, summary


def write_gold_metric_semantics_audit(
    output_dir: Path,
    query_rows: Sequence[Mapping[str, Any]],
    duplicate_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "full1000_query_denominators.jsonl", query_rows)
    _write_jsonl(output_dir / "duplicate_relations.jsonl", duplicate_rows)
    _write_jsonl(output_dir / "frozen_replay_cases.jsonl", replay_rows)
    _write_json(output_dir / "summary.json", summary)


def check_gold_metric_semantics_regression(
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest, require_baseline=True)
    query_rows, duplicate_rows, replay_rows, summary = (
        build_gold_metric_semantics_audit(manifest)
    )
    write_gold_metric_semantics_audit(
        output_dir,
        query_rows,
        duplicate_rows,
        replay_rows,
        summary,
    )
    expected = {
        "full1000_query_denominators": _read_jsonl(
            _repo_path(manifest["baseline"]["query_rows_path"])
        ),
        "duplicate_relations": _read_jsonl(
            _repo_path(manifest["baseline"]["duplicate_rows_path"])
        ),
        "frozen_replay_cases": _read_jsonl(
            _repo_path(manifest["baseline"]["replay_rows_path"])
        ),
        "summary": _read_json(_repo_path(manifest["baseline"]["summary_path"])),
    }
    actual = {
        "full1000_query_denominators": query_rows,
        "duplicate_relations": duplicate_rows,
        "frozen_replay_cases": replay_rows,
        "summary": summary,
    }
    drifts: list[dict[str, Any]] = []
    for item in _input_fingerprint_drifts(manifest):
        drifts.append(item)
    for item in compare_profiles(expected, actual, max_diffs=200):
        drifts.append(item)
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "passed": not drifts,
        "drift_count": len(drifts),
        "drifts": drifts[:200],
        "query_count": len(query_rows),
        "duplicate_relation_count": len(duplicate_rows),
        "frozen_replay_case_count": len(replay_rows),
        "execution": summary["execution"],
        "official_scorer": False,
        "artifact_hashes": {
            name: _sha256_file(output_dir / filename)
            for name, filename in (
                ("query_rows", "full1000_query_denominators.jsonl"),
                ("duplicate_rows", "duplicate_relations.jsonl"),
                ("replay_rows", "frozen_replay_cases.jsonl"),
                ("summary", "summary.json"),
            )
        },
    }
    _write_json(output_dir / "regression_report.json", report)
    return report


def _validate_closure(
    query_rows: Sequence[Mapping[str, Any]],
    duplicate_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    full = summary["full_autoscholar"]
    if len(query_rows) != int(full["query_count"]):
        raise GoldMetricSemanticsError("full query count is not closed")
    if len(duplicate_rows) != int(full["duplicate_relation_count"]):
        raise GoldMetricSemanticsError("duplicate relation count is not closed")
    if len(replay_rows) != sum(
        int(item["case_count"]) for item in summary["frozen_replay"].values()
    ):
        raise GoldMetricSemanticsError("frozen replay case count is not closed")
    if any(
        not (
            row["candidate_identity_parity"]
            and row["returned_identity_parity"]
            and row["source_terminal_parity"]
        )
        for row in replay_rows
    ):
        raise GoldMetricSemanticsError("non-metric frozen replay state changed")


def _validate_manifest(manifest: Mapping[str, Any], *, require_baseline: bool) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise GoldMetricSemanticsError("unsupported metric semantics manifest")
    if manifest.get("gate") != GATE_NAME:
        raise GoldMetricSemanticsError("unexpected metric semantics gate")
    expected_versions = manifest.get("metric_versions") or {}
    if expected_versions != {
        "legacy": LEGACY_GOLD_METRIC_VERSION,
        "current_internal": DEDUPLICATED_GOLD_METRIC_VERSION,
    }:
        raise GoldMetricSemanticsError("metric version contract drifted")
    if require_baseline:
        required = {
            "query_rows_path",
            "query_rows_sha256",
            "duplicate_rows_path",
            "duplicate_rows_sha256",
            "replay_rows_path",
            "replay_rows_sha256",
            "summary_path",
            "summary_sha256",
        }
        if not required.issubset(manifest.get("baseline") or {}):
            raise GoldMetricSemanticsError("metric semantics baseline is incomplete")


def _input_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    inputs = manifest["inputs"]
    output = {
        name: _sha256_file(_repo_path(inputs[path_field]))
        for name, path_field in (
            ("identity_manifest_sha256", "identity_manifest_path"),
            ("current_rules_manifest_sha256", "current_rules_manifest_path"),
            ("identity_implementation_sha256", "identity_implementation_path"),
            ("metrics_implementation_sha256", "metrics_implementation_path"),
            ("evaluation_schema_sha256", "evaluation_schema_path"),
            (
                "current_rules_reconstruction_sha256",
                "current_rules_reconstruction_path",
            ),
            ("audit_implementation_sha256", "audit_implementation_path"),
        )
    }
    for name, spec in sorted((manifest.get("protected_historical_artifacts") or {}).items()):
        output[f"protected:{name}"] = _sha256_file(_repo_path(spec["path"]))
    return output


def _input_fingerprint_drifts(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    actual = _input_hashes(manifest)
    expected = dict(manifest["input_sha256"])
    for name, spec in sorted((manifest.get("protected_historical_artifacts") or {}).items()):
        expected[f"protected:{name}"] = spec["sha256"]
    return compare_profiles(expected, actual, max_diffs=100)


def _relation_id(query_id: str, index: int) -> str:
    return f"{query_id}::gold[{index}]"


def _optional_delta(before: Any, after: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


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
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise GoldMetricSemanticsError("network access is forbidden in metric audit")

    with patch.object(socket, "create_connection", blocked), patch.object(
        socket.socket, "connect", blocked
    ):
        yield
