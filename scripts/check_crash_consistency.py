#!/usr/bin/env python3
"""Run the offline crash_consistency_v1 persistence gate."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.crash_consistency import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    GATE_NAME,
    SCHEMA_VERSION,
    CrashConsistencyError,
    CrashNotEligible,
    audit_frozen_baseline_eligibility,
    load_protocol,
    run_crash_consistency,
    sanitize_error,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "crash_consistency_v1_protocol.json"


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CrashConsistencyError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Validate Benchmark crash consistency without network access."
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check")
    check.add_argument("--output")
    check.add_argument(
        "--fault",
        choices=["non_atomic_writer"],
        help="Deterministic implementation fault for exit-code verification.",
    )

    frozen = commands.add_parser("audit-frozen")
    frozen.add_argument("--output")
    return parser


def _emit(report: dict[str, Any], output: str | None) -> None:
    if output:
        write_json(Path(output), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def _error_report(status: str, exit_code: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": exit_code,
        "score_scope": "persistence_only_not_quality_or_official_score",
        "violation_count": 0,
        "reason": sanitize_error(reason),
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
            "real_process_kill_count": 0,
            "real_disk_fill_count": 0,
            "sleep_race_count": 0,
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except CrashConsistencyError as exc:
        report = _error_report("usage_error", EXIT_USAGE_ERROR, str(exc))
        _emit(report, None)
        return EXIT_USAGE_ERROR
    root = Path(args.repository_root).resolve()
    output = getattr(args, "output", None)
    try:
        protocol = load_protocol(Path(args.protocol), repository_root=root)
        if args.command == "audit-frozen":
            report = audit_frozen_baseline_eligibility(
                protocol, repository_root=root
            )
        else:
            with tempfile.TemporaryDirectory(prefix="spar-crash-gate-") as value:
                report = run_crash_consistency(
                    protocol,
                    work_root=Path(value),
                    controlled_fault=args.fault,
                )
        _emit(report, output)
        return int(report["exit_code"])
    except CrashNotEligible as exc:
        report = _error_report("not_eligible", EXIT_NOT_ELIGIBLE, str(exc))
        _emit(report, output)
        return EXIT_NOT_ELIGIBLE
    except (CrashConsistencyError, OSError, ValueError) as exc:
        report = _error_report("usage_error", EXIT_USAGE_ERROR, str(exc))
        _emit(report, output)
        return EXIT_USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
