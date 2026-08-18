#!/usr/bin/env python3
"""Audit Full1000 provider pacing without network or formal execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_provider_pacing import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    ProviderPacingError,
    ProviderPacingNotReady,
    audit_readiness,
    build_launch_addendum,
    canonical_json,
    load_protocol,
    simulate_capacity,
    verify_policy,
    verify_resume,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_provider_pacing_v1_protocol.json"


class UsageError(RuntimeError):
    """CLI arguments violate the frozen command contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "verify-policy",
        "simulate-capacity",
        "verify-resume",
        "audit-readiness",
    ):
        child = commands.add_parser(command)
        child.add_argument("--output")
        if command == "verify-policy":
            child.add_argument("--launch-addendum")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _status(status: str, exit_code: int, **values: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
        **values,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(_resolve(root, args.protocol), repository_root=root)
    if args.command == "verify-policy":
        report = verify_policy(root, protocol)
        if args.launch_addendum:
            write_json(
                _resolve(root, args.launch_addendum),
                build_launch_addendum(root, protocol),
            )
    elif args.command == "simulate-capacity":
        report = simulate_capacity(root, protocol)
    elif args.command == "verify-resume":
        report = verify_resume(root, protocol)
    elif args.command == "audit-readiness":
        report = audit_readiness(root, protocol)
    else:
        raise UsageError("unsupported_command")
    if args.output:
        write_json(_resolve(root, args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = _run(_parser().parse_args(argv))
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except ProviderPacingNotReady as exc:
        report = _status(
            "not_ready_missing_provider_capacity_declarations",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        ProviderPacingError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "pacing_or_budget_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, ProviderPacingError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        report = _status(
            "pacing_or_budget_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
