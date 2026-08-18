#!/usr/bin/env python3
"""Build and verify Full1000 backup-member enrollment packages offline."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_backup_member_enrollment import (  # noqa: E402
    EXIT_NOT_READY, EXIT_READY, EXIT_USAGE, EXIT_VIOLATION, PROTOCOL,
    BackupMemberEnrollmentError, audit_readiness, build_kit, canonical_json,
    contract_from_kit, load_protocol, read_object, run_enrollment,
    simulate_matrix, verify_member_package, write_json,
)

DEFAULT_PROTOCOL = "benchmark/formal_backup_member_enrollment_v1_protocol.json"


class UsageError(RuntimeError): pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None: raise UsageError("invalid_arguments")


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-kit")
    build.add_argument("--members", type=int, choices=(2,3,4), required=True)
    build.add_argument("--slot", type=int, required=True); build.add_argument("--challenge", required=True)
    build.add_argument("--issued-epoch", type=int, default=10_000); build.add_argument("--output", required=True)
    run = commands.add_parser("run-enrollment-dry-run")
    run.add_argument("--kit", required=True); run.add_argument("--target", required=True)
    run.add_argument("--evidence", required=True); run.add_argument("--observation-epoch", type=int, required=True)
    run.add_argument("--output", required=True)
    verify = commands.add_parser("verify-member-package")
    verify.add_argument("--kit", required=True); verify.add_argument("--package", required=True)
    verify.add_argument("--observation-epoch", type=int, required=True)
    matrix = commands.add_parser("simulate-matrix"); matrix.add_argument("--output")
    readiness = commands.add_parser("audit-readiness"); readiness.add_argument("--output")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value); return path.resolve() if path.is_absolute() else (root / path).resolve()


def _status(status: str, code: int, reason: str | None = None) -> dict[str, Any]:
    value = {"protocol": PROTOCOL, "status": status, "exit_code": code,
             "formal_validation_complete": False,
             "execution": {"network_request_count": 0, "llm_request_count": 0,
                           "snapshot_write_count": 0}}
    if reason: value["reason_code"] = reason.split(":", 1)[0]
    return value


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(_resolve(root, args.protocol), repository_root=root)
    if args.command == "build-kit":
        return build_kit(root, protocol, member_count=args.members, slot=args.slot,
                         challenge_id=args.challenge, issued_epoch=args.issued_epoch,
                         output=Path(args.output))
    if args.command == "run-enrollment-dry-run":
        contract = contract_from_kit(Path(args.kit), protocol, repository_root=root)
        package = run_enrollment(root, contract, Path(args.target), read_object(Path(args.evidence)),
                                 observation_epoch=args.observation_epoch, synthetic_only=False)
        write_json(Path(args.output), package); return package
    if args.command == "verify-member-package":
        contract = contract_from_kit(Path(args.kit), protocol, repository_root=root)
        package = verify_member_package(root, contract, read_object(Path(args.package)),
                                        observation_epoch=args.observation_epoch,
                                        require_real=False)
        return {**_status("member_package_ready", EXIT_READY),
                "member_count": package["member_count"], "slot": package["slot"],
                "package_sha256": package["package_sha256"]}
    if args.command == "simulate-matrix":
        with tempfile.TemporaryDirectory() as raw:
            report = simulate_matrix(protocol, repository_root=root, temporary_root=Path(raw))
    elif args.command == "audit-readiness": report = audit_readiness()
    else: raise UsageError("unsupported_command")
    if args.output: write_json(Path(args.output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try: report = _run(_parser().parse_args(argv))
    except UsageError: report = _status("usage_error", EXIT_USAGE, "invalid_arguments")
    except (BackupMemberEnrollmentError, KeyError, OSError, TypeError, UnicodeError,
            ValueError, json.JSONDecodeError) as exc:
        report = _status("enrollment_or_attestation_violation", EXIT_VIOLATION, str(exc))
    sys.stdout.buffer.write(canonical_json(report)); return int(report["exit_code"])


if __name__ == "__main__": raise SystemExit(main())
