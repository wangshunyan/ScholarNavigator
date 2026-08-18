#!/usr/bin/env python3
"""Build the gold-blind AutoScholarQuery full-1000 resume audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for root in (REPO_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scholar_agent.evaluation.snapshot_resume import (  # noqa: E402
    SnapshotResumeError,
    build_resume_audit,
    runtime_config_from_record_config,
    write_resume_audit,
)


DEFAULT_SNAPSHOT_DIR = Path(
    "outputs/benchmark_snapshots/autoscholar_current_rules_full1000_3cd47c1"
)
DEFAULT_RECORD_DIR = Path(
    "outputs/benchmark_runs/autoscholar_current_rules_full1000_3cd47c1_record_r1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gold-blind deterministic missing-key and resume audit"
    )
    parser.add_argument(
        "--plan-round",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR / "plans/baseline/plan_round_2.json",
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--record-results",
        type=Path,
        default=DEFAULT_RECORD_DIR / "results.jsonl",
    )
    parser.add_argument(
        "--record-config",
        type=Path,
        default=DEFAULT_RECORD_DIR / "config.json",
    )
    parser.add_argument(
        "--query-input",
        type=Path,
        default=Path("benchmark/autoscholar_query_planning_input.jsonl"),
    )
    parser.add_argument(
        "--planning-baseline",
        type=Path,
        default=Path("benchmark/autoscholar_query_planning_baseline.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        runtime_config = runtime_config_from_record_config(args.record_config)
        summary, resume_manifest, rows = build_resume_audit(
            plan_round_path=args.plan_round,
            snapshot_dir=args.snapshot_dir,
            record_results_path=args.record_results,
            record_config_path=args.record_config,
            query_input_path=args.query_input,
            planning_baseline_path=args.planning_baseline,
            runtime_config=runtime_config,
            source_order=runtime_config.sources,
        )
        hashes = write_resume_audit(
            args.output_dir,
            summary,
            resume_manifest,
            rows,
        )
    except SnapshotResumeError as exc:
        print(f"snapshot_resume_audit_error:{exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "required_key_count": summary["required_key_count"],
                "classification_counts": summary["classification_counts"],
                "resume_key_count": summary["resume_key_count"],
                "network_request_count": summary["execution"][
                    "network_request_count"
                ],
                "snapshot_write_count": summary["execution"][
                    "snapshot_write_count"
                ],
                "artifact_sha256": hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
