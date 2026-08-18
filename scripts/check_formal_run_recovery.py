#!/usr/bin/env python3
"""Audit deterministic Full1000 backup and disaster-recovery controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_run_disaster_recovery import (  # noqa: E402
    EXIT_BLOCKED,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    EXECUTION_ZERO,
    PROTOCOL,
    SCHEMA_VERSION,
    DisasterRecoveryError,
    DisasterRecoveryNotEligible,
    audit_readiness,
    canonical_json,
    create_backup,
    load_protocol,
    restore_backup,
    simulate_disaster,
    verify_backup,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_run_disaster_recovery_v1_protocol.json"


class UsageError(RuntimeError):
    """The command line is incomplete or inconsistent."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Check offline Full1000 disaster recovery.")
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    backup = commands.add_parser("backup")
    backup.add_argument("--run-root", required=True)
    backup.add_argument("--backup-root", required=True)
    backup.add_argument("--parent-backup-id")
    backup.add_argument("--output")

    verify = commands.add_parser("verify-backup")
    verify.add_argument("--backup-root", required=True)
    verify.add_argument("--backup-id")
    verify.add_argument("--output")

    restore = commands.add_parser("restore")
    restore.add_argument("--backup-root", required=True)
    restore.add_argument("--target", required=True)
    restore.add_argument("--backup-id")
    restore.add_argument("--output")

    simulate = commands.add_parser("simulate-disaster")
    simulate.add_argument("--output")

    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _emit(value: dict[str, object], output: str | None = None) -> None:
    if output:
        write_json(Path(output), value)
    sys.stdout.buffer.write(canonical_json(value))


def _status(status: str, exit_code: int, **values: object) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": EXECUTION_ZERO,
        **values,
    }


def main(argv: list[str] | None = None) -> int:
    output: str | None = None
    try:
        args = _parser().parse_args(argv)
        output = getattr(args, "output", None)
        root = Path(args.repository_root).resolve()
        protocol_path = _resolve(root, args.protocol)
        protocol = load_protocol(protocol_path, repository_root=root)
        if args.command == "backup":
            report = create_backup(
                _resolve(root, args.run_root),
                _resolve(root, args.backup_root),
                repository_root=root,
                protocol=protocol,
                parent_backup_id=args.parent_backup_id,
            )
        elif args.command == "verify-backup":
            report = verify_backup(
                _resolve(root, args.backup_root),
                repository_root=root,
                protocol=protocol,
                backup_id=args.backup_id,
            )
        elif args.command == "restore":
            report = restore_backup(
                _resolve(root, args.backup_root),
                _resolve(root, args.target),
                repository_root=root,
                protocol=protocol,
                backup_id=args.backup_id,
            )
        elif args.command == "simulate-disaster":
            report = simulate_disaster(root, protocol)
        else:
            report = audit_readiness(root, protocol)
        _emit(report, output)
        return int(report["exit_code"])
    except UsageError:
        report = _status("usage_error", EXIT_USAGE, reason_code="usage_error")
    except DisasterRecoveryNotEligible as exc:
        report = _status(
            "external_run_not_started",
            EXIT_BLOCKED,
            reason_code=str(exc),
        )
    except (
        DisasterRecoveryError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        ValidationError,
    ) as exc:
        report = _status(
            "backup_or_recovery_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, DisasterRecoveryError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    _emit(report, output)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
