#!/usr/bin/env python3
"""Register and preflight explicitly listed local Full1000 backup targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_backup_target_registration import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    BackupTargetRegistrationError,
    audit_readiness,
    build_registration_manifest,
    canonical_json,
    load_private_registration,
    load_protocol,
    read_object,
    simulate_profiles,
    validate_manifest,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_backup_target_registration_v1_protocol.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError("invalid_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register-dry-run")
    register.add_argument("--registration", required=True)
    register.add_argument("--output")
    verify = commands.add_parser("verify-registration")
    verify.add_argument("--registration", required=True)
    verify.add_argument("--manifest", required=True)
    probe = commands.add_parser("probe-target")
    probe.add_argument("--registration", required=True)
    probe.add_argument("--alias", required=True)
    probe.add_argument("--output")
    simulation = commands.add_parser("simulate-profiles")
    simulation.add_argument("--output")
    readiness = commands.add_parser("audit-readiness")
    readiness.add_argument("--registration")
    readiness.add_argument("--output")
    return parser


def _status(status: str, exit_code: int, reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "execution": dict(EXECUTION_ZERO),
    }
    if reason:
        result["reason_code"] = reason.split(":", 1)[0]
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol_path = Path(args.protocol)
    if not protocol_path.is_absolute():
        protocol_path = root / protocol_path
    protocol = load_protocol(protocol_path, repository_root=root)
    if args.command == "simulate-profiles":
        report = simulate_profiles(protocol, repository_root=root)
    elif args.command == "audit-readiness" and not args.registration:
        report = audit_readiness(protocol)
    else:
        registration = load_private_registration(
            Path(args.registration), protocol=protocol
        )
        if args.command == "verify-registration":
            manifest = validate_manifest(
                protocol,
                registration,
                read_object(Path(args.manifest)),
                repository_root=root,
            )
            exit_code = (
                EXIT_READY
                if manifest["registered_candidate_count"]
                else EXIT_NOT_READY
            )
            report = {
                **_status(manifest["status"], exit_code),
                "manifest": manifest,
            }
        else:
            if args.command == "probe-target":
                targets = [row for row in registration["targets"] if row["alias"] == args.alias]
                if len(targets) != 1:
                    raise BackupTargetRegistrationError("registered_alias_unknown")
                registration = {**registration, "targets": targets, "revoked_aliases": [alias for alias in registration["revoked_aliases"] if alias == args.alias]}
            manifest = build_registration_manifest(
                protocol, registration, repository_root=root
            )
            report = {
                **_status(
                    manifest["status"],
                    EXIT_READY if manifest["registered_candidate_count"] else EXIT_NOT_READY,
                ),
                "manifest": manifest,
            }
    if getattr(args, "output", None):
        write_json(Path(args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = _run(_parser().parse_args(argv))
    except UsageError as exc:
        report = _status("usage_error", EXIT_USAGE, str(exc))
    except (
        BackupTargetRegistrationError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "registration_or_probe_violation",
            EXIT_VIOLATION,
            str(exc) if isinstance(exc, BackupTargetRegistrationError) else "controlled_input_violation",
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
