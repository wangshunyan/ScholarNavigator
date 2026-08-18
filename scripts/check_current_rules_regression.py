#!/usr/bin/env python3
"""Check or deliberately propose an update to the frozen current-rules gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scholar_agent.evaluation.current_rules_regression import (
    BASELINE_APPROVAL_TOKEN,
    build_baseline_proposal,
    check_current_rules_regression,
    write_baseline_proposal,
    write_gate_artifacts,
)


DEFAULT_MANIFEST = Path("benchmark/current_rules_regression_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only current_rules regression gate over frozen Replay."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="run the read-only regression gate")
    check.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    check.add_argument("--output-dir", type=Path, required=True)

    proposal = subparsers.add_parser(
        "propose-baseline",
        help="write a review-only baseline proposal without mutating tracked files",
    )
    proposal.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    proposal.add_argument("--output-dir", type=Path, required=True)
    proposal.add_argument("--approval-token", required=True)
    proposal.add_argument("--reason", required=True)

    args = parser.parse_args()
    if args.command == "propose-baseline":
        proposed, audit = build_baseline_proposal(
            args.manifest,
            approval_token=args.approval_token,
            reason=args.reason,
        )
        write_baseline_proposal(args.output_dir, proposed=proposed, audit=audit)
        print(
            "baseline_proposal_complete "
            f"token={BASELINE_APPROVAL_TOKEN} "
            f"drifts={audit['drift_count']} tracked_files_modified=false"
        )
        return

    observed, report = check_current_rules_regression(args.manifest)
    write_gate_artifacts(args.output_dir, observed=observed, report=report)
    print(
        "current_rules_regression "
        f"passed={str(report['passed']).lower()} "
        f"cases={report['case_count']} drifts={report['drift_count']} "
        "network=0 llm=0 snapshot_writes=0"
    )
    if not report["passed"]:
        for drift in report["drifts"][:10]:
            print(
                f"drift path={drift['path']} kind={drift['kind']}",
                file=sys.stderr,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
