#!/usr/bin/env python3
"""Run the offline AutoScholarQuery gold identity audit and regression gate."""

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

from scholar_agent.evaluation.autoscholar_gold_identity import (  # noqa: E402
    BASELINE_APPROVAL_TOKEN,
    GoldIdentityAuditError,
    build_gold_identity_audit,
    check_gold_identity_regression,
    propose_gold_identity_baseline,
    write_gold_identity_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen AutoScholarQuery gold identities offline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "check"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--manifest",
            type=Path,
            default=Path("benchmark/autoscholar_gold_identity_manifest.json"),
        )
        command.add_argument("--output-dir", type=Path, required=True)
    propose = subparsers.add_parser("propose-baseline")
    propose.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark/autoscholar_gold_identity_manifest.json"),
    )
    propose.add_argument("--output-dir", type=Path, required=True)
    propose.add_argument("--approval-token", required=True)
    propose.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            gold, queries, summary = build_gold_identity_audit(manifest)
            write_gold_identity_audit(args.output_dir, gold, queries, summary)
            print(
                "gold_identity_audit "
                f"cases={len(queries)} gold={len(gold)} "
                f"evaluable={summary['safe_evaluable_relation_count']} "
                "network=0 llm=0 snapshot_writes=0"
            )
            return 0
        if args.command == "propose-baseline":
            report = propose_gold_identity_baseline(
                args.manifest,
                args.output_dir,
                approval_token=args.approval_token,
                reason=args.reason,
            )
            print(
                "gold_identity_baseline_proposal "
                f"gold={report['gold_relation_count']} "
                f"token={BASELINE_APPROVAL_TOKEN} tracked_mutation=0"
            )
            return 0
        report = check_gold_identity_regression(args.manifest, args.output_dir)
        print(
            "gold_identity_regression "
            f"passed={str(report['passed']).lower()} "
            f"cases={report['case_count']} gold={report['gold_relation_count']} "
            f"drifts={report['drift_count']} network=0 llm=0 snapshot_writes=0"
        )
        for drift in report["drifts"][:10]:
            print(f"drift {drift['path']} {drift['kind']}")
        return 0 if report["passed"] else 1
    except GoldIdentityAuditError as exc:
        print(f"gold_identity_audit_error:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
