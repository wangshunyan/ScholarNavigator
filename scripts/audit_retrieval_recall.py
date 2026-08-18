#!/usr/bin/env python3
"""为固定 holdout30 规划、采集或回放 Retrieval 召回审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.datasets import load_dataset  # noqa: E402
from scholar_agent.evaluation.holdout_comparison import (  # noqa: E402
    HOLDOUT_LIMIT,
    HOLDOUT_OFFSET,
)
from scholar_agent.evaluation.retrieval_recall_audit import (  # noqa: E402
    AuditSnapshotStore,
    analyze_retrieval_recall,
    build_audit_requests,
    collect_audit_requests,
    write_audit_outputs,
    write_audit_plan,
)
from scholar_agent.evaluation.snapshots import SnapshotStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="固定 AutoScholarQuery offset=20/limit=30 的事后召回审计。"
    )
    parser.add_argument("--mode", choices=["plan", "record-missing", "replay"], required=True)
    parser.add_argument(
        "--holdout-run",
        default="outputs/benchmark_runs/holdout30_baseline/H-current",
    )
    parser.add_argument(
        "--retrieval-snapshot",
        default="outputs/benchmark_snapshots/autoscholar_holdout30_20260720",
    )
    parser.add_argument(
        "--audit-snapshot",
        default="outputs/benchmark_snapshots/retrieval_recall_audit_holdout30_20260720",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark_runs/retrieval_recall_audit",
    )
    parser.add_argument("--max-new-requests", type=int)
    parser.add_argument("--source-failure-limit", type=int, default=3)
    args = parser.parse_args(argv)
    if args.max_new_requests is not None and args.max_new_requests <= 0:
        parser.error("--max-new-requests must be positive")
    if args.source_failure_limit <= 0:
        parser.error("--source-failure-limit must be positive")

    holdout_run = Path(args.holdout_run).expanduser().resolve()
    retrieval_snapshot = Path(args.retrieval_snapshot).expanduser().resolve()
    audit_store = AuditSnapshotStore(args.audit_snapshot)
    queries = load_dataset("auto_scholar_query")[
        HOLDOUT_OFFSET : HOLDOUT_OFFSET + HOLDOUT_LIMIT
    ]
    result_rows = _rows_by_case(holdout_run / "results.jsonl")
    requests, request_index = build_audit_requests(
        queries,
        result_rows,
        SnapshotStore(retrieval_snapshot),
    )
    input_metadata = {
        "holdout_run": str(holdout_run),
        "holdout_results_sha256": _sha256(holdout_run / "results.jsonl"),
        "retrieval_snapshot": str(retrieval_snapshot),
        "retrieval_manifest_sha256": _sha256(retrieval_snapshot / "manifest.json"),
        "request_index_sha256": _stable_hash(request_index),
    }
    manifest = write_audit_plan(
        audit_store,
        requests,
        input_metadata=input_metadata,
    )
    if args.mode == "plan":
        print(
            json.dumps(
                {key: value for key, value in manifest.items() if key != "request_keys"},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "record-missing":
        result = collect_audit_requests(
            audit_store,
            requests,
            max_new_requests=args.max_new_requests,
            source_failure_limit=args.source_failure_limit,
            progress=_print_progress,
        )
        write_audit_plan(audit_store, requests, input_metadata=input_metadata)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key != "remaining_missing_keys"
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result["replay_ready"] else 2

    gold_rows, query_rows, aggregate = analyze_retrieval_recall(
        queries,
        result_rows,
        requests,
        request_index,
        audit_store,
    )
    write_audit_outputs(args.output, gold_rows, query_rows, aggregate)
    print(Path(args.output).expanduser().resolve())
    return 0


def _rows_by_case(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["case_id"]): row for row in rows}


def _print_progress(
    index: int,
    total: int,
    request: Any,
    entry: Any,
) -> None:
    print(
        f"[{index}/{total}] {request.kind} {entry.status} "
        f"papers={len(entry.papers)} key={request.key[:12]}",
        flush=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
