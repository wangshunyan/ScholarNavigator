#!/usr/bin/env python3
"""Run or verify deterministic_tiebreak_qualification_v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.evaluation.tiebreak_qualification import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_NOT_QUALIFIED,
    EXIT_QUALIFIED,
    EXIT_USAGE_ERROR,
    EXIT_VIOLATION,
    TieBreakNotEligible,
    TieBreakQualificationError,
    run_tiebreak_qualification,
    verify_analysis,
    write_analysis,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线资格审计 deterministic_tiebreak_v2，不改变生产默认行为。"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行冻结 Record160 资格审计")
    run.add_argument("--protocol", required=True)
    run.add_argument("--output", required=True)
    verify = commands.add_parser("verify", help="验证既有资格审计产物")
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
            cases, candidates, ties, aggregate = run_tiebreak_qualification(
                args.protocol
            )
            manifest = write_analysis(
                args.output,
                cases,
                candidates,
                ties,
                aggregate,
                args.protocol,
            )
            exit_code = int(aggregate["exit_code"])
            _emit(
                {
                    "analysis": CONTRACT_VERSION,
                    "candidate_count": len(candidates),
                    "exit_code": exit_code,
                    "manifest_file_count": len(manifest["files"]),
                    "status": str(aggregate["status"]),
                    "tie_group_count": len(ties),
                }
            )
            return exit_code
        if args.command == "verify":
            result = verify_analysis(args.output)
            _emit(result)
            return int(result["exit_code"])
        return EXIT_USAGE_ERROR
    except TieBreakNotEligible as exc:
        _emit(
            {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_NOT_QUALIFIED,
                "reason": str(exc),
                "status": "not_eligible",
            }
        )
        return EXIT_NOT_QUALIFIED
    except TieBreakQualificationError as exc:
        _emit(
            {
                "analysis": CONTRACT_VERSION,
                "exit_code": EXIT_VIOLATION,
                "reason": str(exc),
                "status": "semantic_or_stability_violation",
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
