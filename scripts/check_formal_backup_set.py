#!/usr/bin/env python3
"""Audit Full1000 multi-target backup-set topology offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_backup_set_topology import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    BackupSetError,
    BackupSetNotReady,
    audit_readiness,
    build_topology,
    calculate_capacity,
    canonical_json,
    load_protocol,
    simulate_set,
    verify_set,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_backup_set_topology_v1_protocol.json"


class UsageError(RuntimeError):
    """Arguments do not satisfy the public CLI contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError("invalid_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    topology = commands.add_parser("build-topology")
    topology.add_argument("--members", type=int, choices=(2, 3, 4), default=4)
    topology.add_argument("--output")
    capacity = commands.add_parser("calculate-capacity")
    capacity.add_argument("--output")
    verify = commands.add_parser("verify-set")
    verify.add_argument("--output")
    simulation = commands.add_parser("simulate-set")
    simulation.add_argument("--members", type=int, choices=(2, 3, 4), default=4)
    simulation.add_argument("--output")
    readiness = commands.add_parser("audit-readiness")
    readiness.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _status(
    status: str, exit_code: int, *, reason_code: str | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    if reason_code:
        value["reason_code"] = reason_code.split(":", 1)[0]
    return value


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(_resolve(root, args.protocol), repository_root=root)
    if args.command == "build-topology":
        report = {
            **_status("backup_set_ready", EXIT_READY),
            "topology": build_topology(protocol, member_count=args.members),
        }
    elif args.command == "calculate-capacity":
        report = calculate_capacity(protocol)
    elif args.command == "verify-set":
        report = verify_set(protocol)
    elif args.command == "simulate-set":
        report = simulate_set(protocol, member_count=args.members)
    elif args.command == "audit-readiness":
        report = audit_readiness(protocol)
    else:
        raise UsageError("unsupported_command")
    if args.output:
        write_json(Path(args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = _run(_parser().parse_args(argv))
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="invalid_arguments")
    except BackupSetNotReady as exc:
        report = _status(
            "not_ready_missing_qualified_members",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        BackupSetError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "topology_or_recovery_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, BackupSetError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        report = _status(
            "topology_or_recovery_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(report))
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
