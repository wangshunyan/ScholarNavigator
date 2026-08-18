#!/usr/bin/env python3
"""Audit Full1000 storage quotas, capacity, and retention without network I/O."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_run_storage_governance import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    CapacityObservation,
    StorageCapacityNotReady,
    StorageGovernanceError,
    audit_readiness,
    build_launch_addendum,
    build_storage_plan,
    canonical_json,
    load_protocol,
    observe_capacity,
    simulate_pressure,
    verify_capacity,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_run_storage_governance_v1_protocol.json"


class UsageError(RuntimeError):
    """Arguments do not satisfy the public CLI contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-plan")
    build.add_argument("--output")
    build.add_argument("--addendum-output")

    capacity = commands.add_parser("verify-capacity")
    capacity.add_argument("--primary-root", required=True)
    capacity.add_argument("--backup-root", required=True)
    capacity.add_argument("--output")

    simulate = commands.add_parser("simulate-pressure")
    simulate.add_argument("--output")

    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _emit(value: dict[str, Any], output: str | None = None) -> None:
    if output:
        write_json(Path(output), value)
    sys.stdout.buffer.write(canonical_json(value))


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
    if args.command == "build-plan":
        plan = build_storage_plan(root, protocol)
        addendum = build_launch_addendum(root, protocol, plan)
        if args.output:
            write_json(Path(args.output), plan)
        if args.addendum_output:
            write_json(Path(args.addendum_output), addendum)
        return _status(
            "storage_controls_ready",
            EXIT_READY,
            storage_plan_sha256=plan["plan_sha256"],
            launch_addendum_sha256=addendum["addendum_sha256"],
        )
    if args.command == "verify-capacity":
        plan = build_storage_plan(root, protocol)
        primary = observe_capacity(
            _resolve(root, args.primary_root), "primary"
        )
        backup = observe_capacity(_resolve(root, args.backup_root), "backup")
        report = verify_capacity(plan, primary=primary, backup=backup)
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "simulate-pressure":
        report = simulate_pressure(root, protocol)
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "audit-readiness":
        report = audit_readiness(root, protocol)
        if args.output:
            write_json(Path(args.output), report)
        return report
    raise UsageError("unsupported command")


def main(argv: Sequence[str] | None = None) -> int:
    output: str | None = None
    try:
        args = _parser().parse_args(argv)
        output = getattr(args, "output", None)
        report = _run(args)
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except StorageCapacityNotReady as exc:
        report = _status(
            "not_ready_capacity_unverified",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        StorageGovernanceError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "quota_or_retention_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, StorageGovernanceError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        _emit(report, None)
    except (OSError, UnicodeError, TypeError, ValueError):
        fallback = _status(
            "quota_or_retention_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(fallback))
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
