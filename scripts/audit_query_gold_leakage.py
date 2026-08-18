#!/usr/bin/env python3
"""Run or regression-check the frozen query-to-gold leakage audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.query_gold_leakage_audit import (  # noqa: E402
    build_query_gold_leakage_audit,
    check_query_gold_leakage_regression,
    write_query_gold_leakage_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit deterministic query-to-gold information leakage."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--manifest",
        default="benchmark/autoscholar_query_gold_leakage_manifest.json",
    )
    run.add_argument("--output", required=True)
    check = subparsers.add_parser("check")
    check.add_argument(
        "--manifest",
        default="benchmark/autoscholar_query_gold_leakage_manifest.json",
    )
    check.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        manifest = json.loads(
            Path(args.manifest).expanduser().resolve().read_text(encoding="utf-8")
        )
        relations, queries, summary = build_query_gold_leakage_audit(manifest)
        write_query_gold_leakage_audit(args.output, relations, queries, summary)
        print(
            json.dumps(
                {
                    "query_count": len(queries),
                    "relation_count": len(relations),
                    "validity_risk_band": summary["validity_risk_band"],
                    "network_request_count": 0,
                    "llm_request_count": 0,
                    "snapshot_write_count": 0,
                    "output": str(Path(args.output).expanduser().resolve()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    report = check_query_gold_leakage_regression(args.manifest, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
