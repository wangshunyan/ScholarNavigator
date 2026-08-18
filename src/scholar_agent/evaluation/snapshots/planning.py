"""动态 Benchmark 快照规划的集中配置与覆盖率产物。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.evaluation.snapshots.store import (
    SnapshotMissingError,
    SnapshotStore,
    utc_now,
)


ABLATION_GROUPS = (
    "baseline",
    "query_evolution_only",
    "refchain_only",
    "query_evolution_plus_refchain",
)


class SnapshotCollectionLimits(BaseModel):
    """公共来源增量采集的保守安全上限。"""

    max_plan_rounds: int = Field(default=4, ge=1)
    max_new_requests: int = Field(default=24, ge=0)
    max_new_failed_entries: int = Field(default=4, ge=0)
    max_collection_seconds: float = Field(default=300.0, ge=0.0)
    source_failure_limit: int = Field(default=2, ge=1)


def plan_group_root(snapshot_dir: Path | str, group: str) -> Path:
    return Path(snapshot_dir).expanduser().resolve() / "plans" / group


def plan_round_root(
    snapshot_dir: Path | str,
    group: str,
    round_index: int,
) -> Path:
    return plan_group_root(snapshot_dir, group) / f"round_{round_index}"


def write_coverage_artifacts(
    snapshot_dir: Path | str,
    *,
    group: str,
    round_index: int,
) -> dict[str, Any]:
    """写入单轮覆盖和四组全局汇总；不访问网络。"""

    root = Path(snapshot_dir).expanduser().resolve()
    report = SnapshotStore(root).inspect()
    coverage = _all_group_coverage(report.get("groups") or {})
    round_root = plan_round_root(root, group, round_index)
    _atomic_write_json(round_root / "coverage_after_round.json", coverage[group])
    plans_root = root / "plans"
    _atomic_write_json(plans_root / "group_coverage.json", coverage)
    _atomic_write_json(
        plans_root / "collection_summary.json",
        {
            "snapshot_name": root.name,
            "updated_at": utc_now(),
            "dynamic_collection": _dynamic_collection_totals(
                SnapshotStore(root),
                report,
            ),
            "groups": coverage,
        },
    )
    return coverage


def _dynamic_collection_totals(
    store: SnapshotStore,
    report: dict[str, Any],
) -> dict[str, Any]:
    """按非 baseline 计划键汇总真实记录成本，避免重复轮次重复计数。"""

    baseline = (report.get("groups") or {}).get("baseline") or {}
    baseline_keys = set(baseline.get("required_retrieval_keys") or []) | set(
        baseline.get("required_reference_keys") or []
    )
    planned: dict[str, tuple[str, str]] = {}
    for path in sorted((store.root / "plans").rglob("plan_round_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in payload.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key") or "")
            entry_type = str(entry.get("entry_type") or "")
            source = str(entry.get("source") or "unknown")
            if (
                len(key) == 64
                and key not in baseline_keys
                and entry_type in {"retrieval", "reference"}
            ):
                planned[key] = (entry_type, source)
    request_count = 0
    retry_count = 0
    successful = 0
    failed = 0
    present = 0
    by_source: dict[str, dict[str, int]] = {}
    for key, (entry_type, source) in planned.items():
        try:
            entry = (
                store.read_retrieval(key)
                if entry_type == "retrieval"
                else store.read_reference(key)
            )
        except SnapshotMissingError:
            continue
        present += 1
        successful += entry.status == "success"
        failed += entry.status == "failed"
        request_count += entry.diagnostics.request_count
        retry_count += entry.diagnostics.retry_count
        row = by_source.setdefault(
            source,
            {"entries": 0, "success": 0, "failed": 0, "requests": 0},
        )
        row["entries"] += 1
        row["success"] += entry.status == "success"
        row["failed"] += entry.status == "failed"
        row["requests"] += entry.diagnostics.request_count
    return {
        "planned_key_count": len(planned),
        "present_entry_count": present,
        "missing_entry_count": len(planned) - present,
        "success_entry_count": successful,
        "failed_entry_count": failed,
        "actual_request_count": request_count,
        "retry_count": retry_count,
        "by_source": dict(sorted(by_source.items())),
    }


def mark_group_stop_reason(
    snapshot_dir: Path | str,
    *,
    group: str,
    stop_reason: str,
) -> None:
    store = SnapshotStore(snapshot_dir)
    manifest = store.read_manifest()
    observation = manifest.groups.get(group)
    if observation is None:
        return
    store.update_group(
        group,
        observation.model_copy(
            update={
                "collection_completed": False,
                "replay_ready": False,
                "replay_verified": False,
                "stop_reason": stop_reason,
                "completed": False,
                "updated_at": utc_now(),
            }
        ),
    )


def _all_group_coverage(observed: dict[str, Any]) -> dict[str, Any]:
    empty = {
        "completed": False,
        "collection_started": False,
        "collection_completed": False,
        "replay_ready": False,
        "replay_verified": False,
        "retrieval_key_count": 0,
        "reference_key_count": 0,
        "required_retrieval_keys": [],
        "required_reference_keys": [],
        "required_key_count": 0,
        "present_success_entries": 0,
        "present_failed_entries": 0,
        "missing_entries": 0,
        "missing_retrieval_keys": [],
        "missing_reference_keys": [],
        "plan_rounds": 0,
        "last_plan_round": 0,
        "stop_reason": "not_planned",
        "missing_keys_by_source": {},
        "failed_keys_by_source": {},
    }
    groups = list(ABLATION_GROUPS)
    groups.extend(group for group in observed if group not in groups)
    return {
        group: dict(observed.get(group) or empty)
        for group in groups
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
