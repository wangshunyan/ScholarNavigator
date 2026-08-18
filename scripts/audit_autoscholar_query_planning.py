#!/usr/bin/env python3
"""Run the gold-blind AutoScholarQuery planning audit."""

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

from scholar_agent.evaluation.query_planning_regression import (  # noqa: E402
    BASELINE_APPROVAL_TOKEN,
    QueryPlanningAuditError,
    check_planning_regression,
    project_query_only_manifest,
    propose_planning_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gold-blind offline query-planning regression audit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    project = subparsers.add_parser(
        "project-input",
        help="project qid/question only; all other dataset fields are ignored",
    )
    project.add_argument("--source", type=Path, required=True)
    project.add_argument("--output", type=Path, required=True)

    check = subparsers.add_parser("check", help="run the frozen offline gate")
    check.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark/autoscholar_query_planning_manifest.json"),
    )
    check.add_argument("--output-dir", type=Path, required=True)

    propose = subparsers.add_parser(
        "propose-baseline",
        help="write review-only baseline artifacts without mutating tracked files",
    )
    propose.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark/autoscholar_query_planning_manifest.json"),
    )
    propose.add_argument("--output-dir", type=Path, required=True)
    propose.add_argument("--approval-token", required=True)
    propose.add_argument("--reason", required=True)

    args = parser.parse_args()
    try:
        if args.command == "project-input":
            result = project_query_only_manifest(args.source, args.output)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "propose-baseline":
            result = propose_planning_baseline(
                args.manifest,
                args.output_dir,
                approval_token=args.approval_token,
                reason=args.reason,
            )
            print(
                "planning_baseline_proposal "
                f"cases={result['query_plan_hash_count']} "
                f"token={BASELINE_APPROVAL_TOKEN} tracked_mutation=0"
            )
            return 0
        report = check_planning_regression(args.manifest, args.output_dir)
        print(
            "query_planning_regression "
            f"passed={str(report['passed']).lower()} "
            f"cases={report['case_count']} "
            f"success={report['success_count']} errors={report['error_count']} "
            f"drifts={report['drift_count']} "
            f"network={report['execution']['network_request_count']} "
            f"llm={report['execution']['llm_request_count']} "
            f"snapshot_writes={report['execution']['snapshot_write_count']}"
        )
        for drift in report["drifts"][:10]:
            print(f"drift {drift['path']} {drift['kind']}")
        return 0 if report["passed"] else 1
    except QueryPlanningAuditError as exc:
        print(f"query_planning_audit_error:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
