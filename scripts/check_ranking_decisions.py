#!/usr/bin/env python3
"""Run or verify the frozen Judgement/rerank decision audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.evaluation.ranking_decision_audit import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_COMPLETED,
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    EXIT_VIOLATION,
    RankingDecisionAuditError,
    RankingDecisionNotEligible,
    run_ranking_decision_audit,
    verify_analysis,
    write_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线审计 current_rules Judgement、rerank 与 Top-20 决策。"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行冻结 Record160 排序审计")
    run.add_argument("--protocol", required=True)
    run.add_argument("--output", required=True)
    verify = commands.add_parser("verify", help="验证既有排序审计产物")
    verify.add_argument("--output", required=True)
    return parser


def _emit(value: dict[str, object]) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command == "run":
            cases, decisions, aggregate = run_ranking_decision_audit(args.protocol)
            manifest = write_analysis(
                args.output, cases, decisions, aggregate, args.protocol
            )
            _emit(
                {
                    "analysis": CONTRACT_VERSION,
                    "candidate_decision_count": len(decisions),
                    "exit_code": EXIT_COMPLETED,
                    "manifest_file_count": len(manifest["files"]),
                    "status": "completed",
                }
            )
            return EXIT_COMPLETED
        if args.command == "verify":
            _emit(verify_analysis(args.output))
            return EXIT_COMPLETED
        return EXIT_USAGE_ERROR
    except RankingDecisionNotEligible as exc:
        _emit(
            {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_NOT_ELIGIBLE,
                "reason": str(exc),
                "status": "not_eligible",
            }
        )
        return EXIT_NOT_ELIGIBLE
    except RankingDecisionAuditError as exc:
        _emit(
            {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_VIOLATION,
                "reason": str(exc),
                "status": "reconstruction_or_ranking_violation",
            }
        )
        return EXIT_VIOLATION
    except (OSError, TypeError, ValueError, KeyError) as exc:
        _emit(
            {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_USAGE_ERROR,
                "reason": type(exc).__name__,
                "status": "usage_error",
            }
        )
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
