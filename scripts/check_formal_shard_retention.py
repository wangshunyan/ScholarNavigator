#!/usr/bin/env python3
"""Audit Full1000 shard streaming retention without network execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_shard_streaming_retention import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    ShardRetentionError,
    ShardRetentionNotReady,
    audit_readiness,
    build_addendum,
    canonical_json,
    load_protocol,
    read_object,
    simulate_streaming,
    verify_capacity,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_shard_streaming_retention_v1_protocol.json"


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

    build = commands.add_parser("build-addendum")
    build.add_argument("--active-shard-window", type=int, default=4)
    build.add_argument("--output")

    capacity = commands.add_parser("verify-capacity")
    capacity.add_argument("--addendum", required=True)
    capacity.add_argument("--primary-available-bytes", type=int, required=True)
    capacity.add_argument("--primary-available-inodes", required=True)
    capacity.add_argument("--primary-quota-bytes", required=True)
    capacity.add_argument("--backup-available-bytes", required=True)
    capacity.add_argument("--backup-available-inodes", required=True)
    capacity.add_argument("--backup-quota-bytes", required=True)
    capacity.add_argument("--backup-failure-domain-independent", required=True)
    capacity.add_argument("--output")

    simulate = commands.add_parser("simulate-streaming")
    simulate.add_argument("--output")

    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _status(status: str, exit_code: int, reason_code: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "reason_code": reason_code.split(":", 1)[0],
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }


def _number_or_unknown(value: str) -> int | str:
    if value == "not_available":
        return value
    number = int(value)
    if number < 0:
        raise UsageError("negative observation")
    return number


def _boolean_or_unknown(value: str) -> bool | str:
    if value == "not_available":
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise UsageError("invalid boolean observation")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(
        _resolve(root, args.protocol), repository_root=root
    )
    if args.command == "build-addendum":
        report = build_addendum(
            protocol, window=args.active_shard_window
        )
        if args.output:
            write_json(Path(args.output), report)
        return {
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "status": "streaming_retention_ready",
            "exit_code": EXIT_READY,
            "addendum_sha256": report["addendum_sha256"],
            "active_shard_window": report["active_shard_window"],
            "default_enabled": report["default_enabled"],
            "formal_validation_complete": False,
            "execution": dict(EXECUTION_ZERO),
        }
    if args.command == "verify-capacity":
        addendum = read_object(_resolve(root, args.addendum))
        report = verify_capacity(
            addendum,
            primary_available_bytes=args.primary_available_bytes,
            primary_available_inodes=_number_or_unknown(
                args.primary_available_inodes
            ),
            primary_quota_bytes=_number_or_unknown(
                args.primary_quota_bytes
            ),
            backup_available_bytes=_number_or_unknown(
                args.backup_available_bytes
            ),
            backup_available_inodes=_number_or_unknown(
                args.backup_available_inodes
            ),
            backup_quota_bytes=_number_or_unknown(args.backup_quota_bytes),
            backup_failure_domain_independent=_boolean_or_unknown(
                args.backup_failure_domain_independent
            ),
        )
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "simulate-streaming":
        report = simulate_streaming(protocol)
        if args.output:
            write_json(Path(args.output), report)
        return report
    if args.command == "audit-readiness":
        report = audit_readiness(protocol)
        if args.output:
            write_json(Path(args.output), report)
        return report
    raise UsageError("unsupported command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = _run(args)
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, "invalid_arguments")
    except ShardRetentionNotReady as exc:
        report = _status(
            "not_ready_missing_qualified_backup", EXIT_NOT_READY, str(exc)
        )
    except (
        ShardRetentionError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "retention_or_recovery_violation",
            EXIT_VIOLATION,
            str(exc)
            if isinstance(exc, ShardRetentionError)
            else "controlled_schema_or_filesystem_violation",
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        fallback = _status(
            "retention_or_recovery_violation",
            EXIT_VIOLATION,
            "output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(fallback))
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
