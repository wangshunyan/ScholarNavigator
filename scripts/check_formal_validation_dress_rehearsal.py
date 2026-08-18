#!/usr/bin/env python3
"""Run and verify the synthetic-only formal validation dress rehearsal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_validation_dress_rehearsal import (  # noqa: E402
    EXIT_BLOCKED,
    EXIT_COMPLETED,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    DressRehearsalBlocked,
    DressRehearsalError,
    audit_readiness,
    build_handoff_checklist,
    canonical_json,
    load_protocol,
    read_json,
    run_rehearsal,
    simulate_failures,
    verify_rehearsal_report,
    write_json,
)


DEFAULT_PROTOCOL = (
    ROOT / "benchmark/formal_validation_dress_rehearsal_v1_protocol.json"
)
DEFAULT_REPORT = (
    ROOT
    / "benchmark/formal_validation_dress_rehearsal_v1_evidence/rehearsal.json"
)


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def parser() -> argparse.ArgumentParser:
    value = Parser(
        description="Exercise the formal validation chain with synthetic evidence only."
    )
    value.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    value.add_argument("--repository-root", default=str(ROOT))
    commands = value.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output")
    run.add_argument("--handoff-output")
    verify = commands.add_parser("verify")
    verify.add_argument("--report", default=str(DEFAULT_REPORT))
    failure = commands.add_parser("simulate-failure")
    failure.add_argument("--report", default=str(DEFAULT_REPORT))
    failure.add_argument("--output")
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--report", default=str(DEFAULT_REPORT))
    audit.add_argument("--output")
    return value


def emit(value: dict[str, object], output: str | None = None) -> None:
    if output:
        write_json(Path(output), value)
    sys.stdout.buffer.write(canonical_json(value))


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        protocol = load_protocol(Path(args.protocol), repository_root=root)
        if args.command == "run":
            report = run_rehearsal(root, protocol)
            if args.handoff_output:
                write_json(Path(args.handoff_output), build_handoff_checklist(protocol))
            emit(report, args.output)
            return EXIT_COMPLETED
        rehearsal = read_json(Path(args.report))
        if args.command == "verify":
            emit(verify_rehearsal_report(rehearsal, protocol))
            return EXIT_COMPLETED
        if args.command == "simulate-failure":
            report = simulate_failures(root, protocol, rehearsal)
            emit(report, args.output)
            return EXIT_COMPLETED
        report = audit_readiness(root, protocol, rehearsal)
        emit(report, args.output)
        return EXIT_BLOCKED
    except UsageError:
        emit(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "status": "usage_error",
                "exit_code": EXIT_USAGE,
            }
        )
        return EXIT_USAGE
    except DressRehearsalBlocked:
        emit(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "status": "real_external_evidence_still_blocked",
                "exit_code": EXIT_BLOCKED,
                "formal_validation_complete": False,
            }
        )
        return EXIT_BLOCKED
    except (
        DressRehearsalError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        emit(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "status": "integration_or_isolation_violation",
                "exit_code": EXIT_VIOLATION,
                "error_code": "rehearsal_input_or_invariant_invalid",
                "formal_validation_complete": False,
            }
        )
        return EXIT_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
