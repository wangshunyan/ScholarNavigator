"""Deterministic registry and default-on gate for tracked experiment evidence."""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import socket
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, get_args
from unittest.mock import patch

from scholar_agent.core.search_schemas import (
    DEFAULT_SEARCH_SOURCES,
    SUPPORTED_SEARCH_SOURCES,
    JudgementPolicy,
    LexicalNormalizationPolicy,
    QueryEvolutionPolicy,
    QueryPlanningPolicy,
    RankingPolicy,
)
from scholar_agent.agents.judgement_config import CURRENT_RULES_CONFIG
from scholar_agent.evaluation.current_rules_regression import compare_profiles
from scholar_agent.evaluation.selection import DEFAULT_RESULT_POLICY, ResultPolicy
from scholar_agent.retrieval.query_adapter import (
    DEFAULT_QUERY_ADAPTER_POLICY,
    QueryAdapterPolicy,
)
from scholar_agent.services.search_service import SearchService


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_ROOT = REPOSITORY_ROOT / "outputs" / "benchmark_snapshots"
SCHEMA_VERSION = "1"
REGISTRY_NAME = "experiment_evidence_registry_v1"
GATE_NAME = "experiment_evidence_default_policy_gate"
ALLOWED_METRIC_VERSIONS = {
    "legacy_gold_records_v1",
    "deduplicated_gold_identity_v2",
    "not_applicable",
}
ALLOWED_DECISIONS = {
    "validated_default",
    "promising_default_off",
    "negative",
    "inconclusive",
    "blocked",
    "unvalidated",
}
ALLOWED_EVIDENCE_STATUS = {"tracked_machine_evidence", "evidence_unavailable"}
STATIC_STRATEGIES = {
    "current_rules",
    "refchain",
    "semantic_seed_expansion",
    "llm_query_understanding",
    "llm_judgement",
    "local_bm25_original_deepening",
}


class EvidenceRegistryError(RuntimeError):
    """Raised when the tracked registry contract is invalid."""


def implemented_strategy_ids() -> tuple[str, ...]:
    """Enumerate every named, selectable strategy in the frozen registry scope."""

    strategy_ids = set(STATIC_STRATEGIES)
    strategy_ids.update(
        value for value in get_args(QueryPlanningPolicy) if value != "current_rules"
    )
    strategy_ids.update(
        f"query_evolution_{value}"
        for value in get_args(QueryEvolutionPolicy)
        if value != "off"
    )
    strategy_ids.update(
        value for value in get_args(RankingPolicy) if value != "current_rules"
    )
    strategy_ids.update(
        value for value in get_args(JudgementPolicy) if value != "current_rules"
    )
    strategy_ids.update(
        value for value in get_args(LexicalNormalizationPolicy) if value != "off"
    )
    strategy_ids.update(
        f"query_adapter_{value}"
        for value in get_args(QueryAdapterPolicy)
        if value != "adaptive"
    )
    strategy_ids.update(
        f"result_policy_{value}"
        for value in get_args(ResultPolicy)
        if value != "highly_and_partial"
    )
    if (
        "local_bm25" in SUPPORTED_SEARCH_SOURCES
        and "local_bm25" not in DEFAULT_SEARCH_SOURCES
    ):
        strategy_ids.add("local_bm25")
    return tuple(sorted(strategy_ids))


def canonical_default_contract() -> dict[str, Any]:
    """Read the canonical SearchService defaults without loading runtime secrets."""

    run_defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SearchService.run_search).parameters.items()
    }
    service_defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SearchService.__init__).parameters.items()
    }
    return {
        "query_planning_policy": run_defaults["query_planning_policy"],
        "ranking_policy": run_defaults["ranking_policy"],
        "judgement_policy": service_defaults["judgement_policy"],
        "lexical_normalization_policy": CURRENT_RULES_CONFIG.lexical_normalization_policy,
        "query_adapter_policy": DEFAULT_QUERY_ADAPTER_POLICY,
        "result_policy": DEFAULT_RESULT_POLICY,
        "enable_query_evolution": run_defaults["enable_query_evolution"],
        "query_evolution_policy": run_defaults["query_evolution_policy"],
        "enable_refchain": run_defaults["enable_refchain"],
        "enable_semantic_seed_expansion": run_defaults[
            "enable_semantic_seed_expansion"
        ],
        "default_sources": list(DEFAULT_SEARCH_SOURCES),
    }


def canonical_default_strategy_ids() -> tuple[str, ...]:
    contract = canonical_default_contract()
    active: set[str] = set()
    if contract["query_planning_policy"] != "current_rules":
        active.add(str(contract["query_planning_policy"]))
    if contract["ranking_policy"] != "current_rules":
        active.add(str(contract["ranking_policy"]))
    if contract["judgement_policy"] != "current_rules":
        active.add(str(contract["judgement_policy"]))
    if contract["lexical_normalization_policy"] != "off":
        active.add(str(contract["lexical_normalization_policy"]))
    if contract["query_adapter_policy"] != "adaptive":
        active.add(f"query_adapter_{contract['query_adapter_policy']}")
    if contract["result_policy"] != "highly_and_partial":
        active.add(f"result_policy_{contract['result_policy']}")
    if contract["enable_query_evolution"]:
        active.add(f"query_evolution_{contract['query_evolution_policy']}")
    if contract["enable_refchain"]:
        active.add("refchain")
    if contract["enable_semantic_seed_expansion"]:
        active.add("semantic_seed_expansion")
    if "local_bm25" in contract["default_sources"]:
        active.add("local_bm25")
    return tuple(sorted(active or {"current_rules"}))


def build_evidence_registry(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    tracked_paths: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Build the registry, compact evidence matrix, and Markdown summary."""

    _validate_manifest_shape(manifest)
    snapshot_before = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    attempts = {"network": 0}
    with _forbid_network(attempts):
        tracked = tuple(sorted(tracked_paths or _git_tracked_paths(repository_root)))
        scanned = _scan_tracked_evidence(
            tracked,
            manifest["scan_patterns"],
            manifest.get("scan_excludes", []),
            repository_root,
        )
        entries = [
            _materialize_entry(item, repository_root, tracked)
            for item in manifest["strategies"]
        ]
        global_blockers = [
            _materialize_blocker(item, repository_root, tracked)
            for item in manifest.get("global_blockers", [])
        ]
        registry = {
            "schema_version": SCHEMA_VERSION,
            "registry": REGISTRY_NAME,
            "protocol_sha256": _protocol_sha256(manifest),
            "score_scope": "internal_benchmark_evidence_not_official_score",
            "inventory_scope": dict(manifest["inventory_scope"]),
            "implemented_strategy_ids": list(implemented_strategy_ids()),
            "canonical_default_contract": canonical_default_contract(),
            "canonical_default_strategy_ids": list(canonical_default_strategy_ids()),
            "scan": {
                "patterns": list(manifest["scan_patterns"]),
                "excludes": list(manifest.get("scan_excludes", [])),
                "tracked_evidence_file_count": len(scanned),
                "tracked_evidence_files": scanned,
            },
            "strategies": sorted(entries, key=lambda item: item["strategy_id"]),
            "global_blockers": sorted(
                global_blockers, key=lambda item: item["blocker_id"]
            ),
        }
        violations = validate_registry_document(registry, repository_root=repository_root)
        if violations:
            raise EvidenceRegistryError(
                "registry violations:" + json.dumps(violations, ensure_ascii=False)
            )
        matrix = _build_matrix(registry)
        summary = _render_markdown(registry, matrix)
    snapshot_after = _tree_signature(DEFAULT_SNAPSHOT_ROOT)
    execution = {
        "network_request_count": attempts["network"],
        "llm_request_count": 0,
        "snapshot_write_count": int(snapshot_before != snapshot_after),
        "benchmark_run_count": 0,
        "input_mode": "tracked_repository_evidence_only",
    }
    if any(
        execution[field]
        for field in (
            "network_request_count",
            "llm_request_count",
            "snapshot_write_count",
            "benchmark_run_count",
        )
    ):
        raise EvidenceRegistryError(f"offline invariant failed:{execution}")
    registry["execution"] = execution
    matrix["execution"] = execution
    return registry, matrix, summary


def validate_registry_document(
    registry: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    entries = list(registry.get("strategies") or [])
    ids = [str(item.get("strategy_id") or "") for item in entries]
    expected = set(implemented_strategy_ids())
    expected_defaults = set(canonical_default_strategy_ids())
    observed = set(ids)
    for strategy_id in sorted(expected - observed):
        violations.append(_violation("missing_strategy", strategy_id, "present", "missing"))
    for strategy_id in sorted(observed - expected):
        violations.append(_violation("unknown_strategy", strategy_id, "absent", "present"))
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    for strategy_id in duplicates:
        matching = [item for item in entries if item.get("strategy_id") == strategy_id]
        violations.append(
            _violation("duplicate_strategy", strategy_id, 1, ids.count(strategy_id))
        )
        canonical = {
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in matching
        }
        if len(canonical) > 1:
            violations.append(
                _violation(
                    "conflicting_strategy_records",
                    strategy_id,
                    "one canonical record",
                    len(canonical),
                )
            )

    for item in entries:
        strategy_id = str(item.get("strategy_id") or "")
        default_enabled = bool(item.get("default_enabled"))
        decision = item.get("decision")
        evidence_status = item.get("evidence_status")
        metric_version = item.get("metric_version")
        if decision not in ALLOWED_DECISIONS:
            violations.append(_violation("invalid_decision", strategy_id, sorted(ALLOWED_DECISIONS), decision))
        if evidence_status not in ALLOWED_EVIDENCE_STATUS:
            violations.append(_violation("invalid_evidence_status", strategy_id, sorted(ALLOWED_EVIDENCE_STATUS), evidence_status))
        if metric_version not in ALLOWED_METRIC_VERSIONS:
            violations.append(_violation("metric_version_drift", strategy_id, sorted(ALLOWED_METRIC_VERSIONS), metric_version))
        expected_default = strategy_id in expected_defaults
        if default_enabled != expected_default:
            violations.append(
                _violation(
                    "default_switch_drift",
                    strategy_id,
                    expected_default,
                    default_enabled,
                )
            )
        if default_enabled and decision != "validated_default":
            violations.append(_violation("default_without_passing_evidence", strategy_id, "validated_default", decision))
        if not default_enabled and decision == "validated_default":
            violations.append(_violation("conflicting_default_conclusion", strategy_id, "non-default decision", decision))
        if evidence_status == "evidence_unavailable":
            if item.get("core_metrics") is not None or item.get("efficiency_cost") is not None:
                violations.append(_violation("unavailable_evidence_has_metrics", strategy_id, None, "populated"))
            if "tracked_primary_artifact_unavailable" not in item.get("blockers", []):
                violations.append(_violation("unavailable_evidence_missing_blocker", strategy_id, "tracked_primary_artifact_unavailable", item.get("blockers")))
        elif item.get("core_metrics") is None:
            violations.append(_violation("machine_evidence_missing_metrics", strategy_id, "populated", None))
        for source in item.get("evidence_sources", []):
            _validate_evidence_source(source, repository_root, strategy_id, violations)
        for name, value in sorted((item.get("artifact_hashes") or {}).items()):
            if not _is_sha256(value):
                violations.append(
                    _violation(
                        "invalid_artifact_hash",
                        f"{strategy_id}.{name}",
                        "64-hex",
                        value,
                    )
                )
        for commit_field in ("implementation_commit", "evaluation_commit"):
            value = item.get(commit_field)
            if value is not None and not _is_commit(value):
                violations.append(_violation("invalid_commit", f"{strategy_id}.{commit_field}", "40-hex", value))
            elif value is not None and not _git_commit_exists(repository_root, value):
                violations.append(
                    _violation(
                        "unknown_commit",
                        f"{strategy_id}.{commit_field}",
                        "tracked git object",
                        value,
                    )
                )
        if item.get("score_scope") != "internal_not_official":
            violations.append(_violation("official_score_mislabel", strategy_id, "internal_not_official", item.get("score_scope")))
    blocker_ids = [str(item.get("blocker_id") or "") for item in registry.get("global_blockers", [])]
    for blocker_id, count in sorted(Counter(blocker_ids).items()):
        if count > 1:
            violations.append(_violation("duplicate_global_blocker", blocker_id, 1, count))
    for blocker in registry.get("global_blockers", []):
        blocker_id = str(blocker.get("blocker_id") or "")
        for source in blocker.get("evidence_sources", []):
            _validate_evidence_source(source, repository_root, blocker_id, violations)
    return violations


def write_evidence_registry(
    output_dir: str | Path,
    registry: Mapping[str, Any],
    matrix: Mapping[str, Any],
    summary: str,
) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "registry.json", registry)
    _write_json(root / "matrix.json", matrix)
    (root / "summary.md").write_text(summary, encoding="utf-8")


def check_evidence_registry(
    manifest_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    drifts: list[dict[str, Any]] = []
    try:
        _validate_manifest_hashes(manifest)
        values = build_evidence_registry(manifest)
        baseline = manifest["baseline"]
        names = ("registry", "matrix", "summary")
        for name, observed in zip(names, values, strict=True):
            path = _repo_path(baseline[f"{name}_path"])
            actual_hash = sha256_file(path)
            if actual_hash != baseline[f"{name}_sha256"]:
                drifts.append(_violation("baseline_hash_drift", name, baseline[f"{name}_sha256"], actual_hash))
            expected: Any = path.read_text(encoding="utf-8") if name == "summary" else _read_json(path)
            if name == "summary":
                if expected != observed:
                    drifts.append(_violation("summary_drift", name, expected, observed))
            else:
                drifts.extend(compare_profiles({name: expected}, {name: observed}, max_diffs=100))
        write_evidence_registry(output / "observed", *values)
    except (EvidenceRegistryError, KeyError, ValueError) as exc:
        drifts.append(_violation("registry_protocol_error", "$", "valid frozen registry", str(exc)))
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
            "benchmark_run_count": 0,
        },
    }
    _write_json(output / "regression_report.json", report)
    return report


def _materialize_entry(
    item: Mapping[str, Any], repository_root: Path, tracked: Sequence[str]
) -> dict[str, Any]:
    output = dict(item)
    output["evidence_sources"] = [
        _evidence_source(path, repository_root, tracked)
        for path in sorted(set(item.get("evidence_paths") or []))
    ]
    output.pop("evidence_paths", None)
    return output


def _materialize_blocker(
    item: Mapping[str, Any], repository_root: Path, tracked: Sequence[str]
) -> dict[str, Any]:
    output = dict(item)
    output["evidence_sources"] = [
        _evidence_source(path, repository_root, tracked)
        for path in sorted(set(item.get("evidence_paths") or []))
    ]
    output.pop("evidence_paths", None)
    return output


def _evidence_source(path_value: str, repository_root: Path, tracked: Sequence[str]) -> dict[str, str]:
    path = Path(path_value)
    normalized = path.as_posix()
    if normalized not in set(tracked):
        raise EvidenceRegistryError(f"evidence is not tracked:{normalized}")
    absolute = (repository_root / path).resolve()
    if not absolute.is_file():
        raise EvidenceRegistryError(f"tracked evidence missing:{normalized}")
    return {"path": normalized, "sha256": sha256_file(absolute)}


def _scan_tracked_evidence(
    tracked: Sequence[str],
    patterns: Sequence[str],
    excludes: Sequence[str],
    repository_root: Path,
) -> list[dict[str, str]]:
    return [
        {"path": path, "sha256": sha256_file(repository_root / path)}
        for path in tracked
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
        and not any(fnmatch.fnmatch(path, pattern) for pattern in excludes)
        and (repository_root / path).is_file()
    ]


def _build_matrix(registry: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for item in registry["strategies"]:
        rows.append(
            {
                "strategy_id": item["strategy_id"],
                "family": item["family"],
                "default_enabled": item["default_enabled"],
                "evidence_status": item["evidence_status"],
                "decision": item["decision"],
                "metric_version": item["metric_version"],
                "datasets": item["datasets"],
                "core_metrics": item["core_metrics"],
                "efficiency_cost": item["efficiency_cost"],
                "call_completeness": item["call_completeness"],
                "blockers": item["blockers"],
                "conclusion": item["conclusion"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "registry": REGISTRY_NAME,
        "score_scope": "internal_benchmark_evidence_not_official_score",
        "strategy_count": len(rows),
        "decision_counts": dict(sorted(Counter(row["decision"] for row in rows).items())),
        "evidence_status_counts": dict(sorted(Counter(row["evidence_status"] for row in rows).items())),
        "default_enabled_strategy_ids": [row["strategy_id"] for row in rows if row["default_enabled"]],
        "rows": rows,
    }


def _render_markdown(registry: Mapping[str, Any], matrix: Mapping[str, Any]) -> str:
    lines = [
        "# 实验证据矩阵",
        "",
        "> 仅汇总仓库已跟踪的内部 Benchmark/审计证据，不是官方赛题成绩。",
        "",
        f"- 策略数：{matrix['strategy_count']}",
        f"- 默认开启：{', '.join(matrix['default_enabled_strategy_ids'])}",
        f"- 证据状态：{json.dumps(matrix['evidence_status_counts'], ensure_ascii=False, sort_keys=True)}",
        "",
        "| 策略 | 类型 | 默认 | 证据 | 决策 | 指标版本 | 数据范围 | 结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in matrix["rows"]:
        datasets = "; ".join(row["datasets"]) or "未评测"
        conclusion = str(row["conclusion"]).replace("|", "\\|")
        lines.append(
            f"| `{row['strategy_id']}` | {row['family']} | "
            f"{'是' if row['default_enabled'] else '否'} | {row['evidence_status']} | "
            f"{row['decision']} | `{row['metric_version']}` | {datasets} | {conclusion} |"
        )
    lines.extend(["", "## 全局阻断", ""])
    for blocker in registry["global_blockers"]:
        lines.append(f"- `{blocker['blocker_id']}`：{blocker['conclusion']}")
    lines.extend(
        [
            "",
            "## 默认策略门禁",
            "",
            "只有 `current_rules` 可以默认开启；任何实验项若无通过证据、处于负面、阻断、不可判定或证据不可用状态，默认开启都会使门禁失败。",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if manifest.get("registry") != REGISTRY_NAME:
        raise EvidenceRegistryError("unexpected evidence registry manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceRegistryError("registry schema drift")
    expected = list(implemented_strategy_ids())
    declared = sorted(str(item.get("strategy_id") or "") for item in manifest.get("strategies", []))
    if declared != expected:
        raise EvidenceRegistryError(f"strategy inventory drift:expected={expected}:observed={declared}")


def _validate_manifest_hashes(manifest: Mapping[str, Any]) -> None:
    _validate_manifest_shape(manifest)
    implementation = manifest["implementation"]
    _validate_hash(_repo_path(implementation["path"]), implementation["sha256"])
    if "baseline" not in manifest:
        raise EvidenceRegistryError("registry baseline missing")


def _validate_evidence_source(
    source: Mapping[str, Any],
    repository_root: Path,
    strategy_id: str,
    violations: list[dict[str, Any]],
) -> None:
    path = (repository_root / str(source.get("path") or "")).resolve()
    if not path.is_file():
        violations.append(_violation("evidence_missing", strategy_id, "existing tracked file", str(path)))
        return
    actual = sha256_file(path)
    if actual != source.get("sha256"):
        violations.append(_violation("evidence_hash_drift", strategy_id, source.get("sha256"), actual))


def _git_tracked_paths(repository_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _protocol_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "baseline"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_commit(value: Any) -> bool:
    text = str(value)
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _git_commit_exists(repository_root: Path, value: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{value}^{{commit}}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _violation(kind: str, path: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {"kind": kind, "path": path, "expected": expected, "observed": observed}


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise EvidenceRegistryError(f"hash drift:{path}:{expected}:{actual}")


def _tree_signature(root: Path) -> str | None:
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


@contextmanager
def _forbid_network(attempts: dict[str, int]) -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> None:
        attempts["network"] += 1
        raise EvidenceRegistryError("network access forbidden")

    with patch.object(socket, "create_connection", blocked), patch.object(socket.socket, "connect", blocked):
        yield


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
