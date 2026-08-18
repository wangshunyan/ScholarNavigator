#!/usr/bin/env python3
"""Audit and gate the deduplicated-gold internal metric semantics offline."""

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

from scholar_agent.evaluation.gold_metric_semantics import (  # noqa: E402
    GoldMetricSemanticsError,
    build_gold_metric_semantics_audit,
    check_gold_metric_semantics_regression,
    write_gold_metric_semantics_audit,
)


DEFAULT_MANIFEST = Path("benchmark/gold_metric_semantics_manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit versioned gold denominator semantics without retrieval."
    )
    parser.add_argument("command", choices=("audit", "check"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            query_rows, duplicate_rows, replay_rows, summary = (
                build_gold_metric_semantics_audit(manifest)
            )
            write_gold_metric_semantics_audit(
                args.output_dir,
                query_rows,
                duplicate_rows,
                replay_rows,
                summary,
            )
            print(
                "gold_metric_semantics_audit "
                f"queries={len(query_rows)} duplicates={len(duplicate_rows)} "
                f"replay_cases={len(replay_rows)} network=0 llm=0 snapshot_writes=0"
            )
            return 0
        report = check_gold_metric_semantics_regression(
            args.manifest,
            args.output_dir,
        )
        print(
            "gold_metric_semantics_regression "
            f"passed={str(report['passed']).lower()} "
            f"queries={report['query_count']} "
            f"duplicates={report['duplicate_relation_count']} "
            f"replay_cases={report['frozen_replay_case_count']} "
            f"drifts={report['drift_count']} network=0 llm=0 snapshot_writes=0"
        )
        return 0 if report["passed"] else 1
    except GoldMetricSemanticsError as exc:
        print(f"gold_metric_semantics_error:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
