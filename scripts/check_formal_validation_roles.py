#!/usr/bin/env python3
"""Offline formal-validation separation-of-duties gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_validation_roles import (  # noqa: E402
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    RoleControlError,
    RoleControlNotReady,
    audit_current,
    canonical_json,
    load_protocol,
    read_json,
    read_json_sequence,
    simulate_matrix,
    verify_ceremony,
)


DEFAULT_PROTOCOL = (
    ROOT / "benchmark/formal_validation_separation_of_duties_v1_protocol.json"
)
DEFAULT_ASSIGNMENTS = (
    ROOT / "benchmark/formal_validation_separation_of_duties_v1_assignments.json"
)


class UsageError(RuntimeError):
    """CLI arguments do not match the public contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _result(
    status: str,
    exit_code: int,
    reason_code: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
    }
    if reason_code is not None:
        value["reason_code"] = reason_code
    return value


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-policy")
    commands.add_parser("simulate-ceremony")
    verify = commands.add_parser("verify-authorization")
    verify.add_argument("--assignments", type=Path, required=True)
    verify.add_argument("--authorizations", type=Path, required=True)
    verify.add_argument("--events", type=Path, required=True)
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    return parser


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    protocol = load_protocol(args.protocol)
    if args.command == "verify-policy":
        return (
            {
                **_result("separation_controls_ready", EXIT_READY),
                "protocol_sha256": protocol["protocol_sha256"],
                "role_count": len(protocol["roles"]),
            },
            EXIT_READY,
        )
    if args.command == "simulate-ceremony":
        report = simulate_matrix()
        report["exit_code"] = EXIT_READY
        return report, EXIT_READY
    if args.command == "verify-authorization":
        assignments = read_json(args.assignments)
        authorizations = read_json_sequence(args.authorizations)
        events = read_json_sequence(args.events)
        report = verify_ceremony(
            assignments=assignments,
            authorizations=authorizations,
            events=events,
        )
        report.update(_result("separation_controls_ready", EXIT_READY))
        return report, EXIT_READY
    if args.command == "audit-readiness":
        report = audit_current(protocol, read_json(args.assignments))
        return report, int(report["exit_code"])
    raise UsageError("unsupported_command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report, exit_code = _run(_parser().parse_args(argv))
    except UsageError:
        report = _result("usage_error", EXIT_USAGE, "invalid_arguments")
        exit_code = EXIT_USAGE
    except RoleControlNotReady as exc:
        report = _result(
            "not_ready_missing_real_role_assignments",
            EXIT_NOT_READY,
            str(exc),
        )
        exit_code = EXIT_NOT_READY
    except (RoleControlError, OSError, UnicodeError, TypeError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, RoleControlError) else "input_invalid"
        report = _result("role_or_approval_violation", EXIT_VIOLATION, reason)
        exit_code = EXIT_VIOLATION
    _emit(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
