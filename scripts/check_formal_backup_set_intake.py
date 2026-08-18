#!/usr/bin/env python3
"""Audit Full1000 backup-set member intake and activation offline."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_backup_set_member_intake import (  # noqa: E402
    EXECUTION_ZERO,
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    BackupSetIntakeError,
    BackupSetIntakeNotReady,
    activate_set,
    audit_readiness,
    build_slot_contract,
    build_slot_kit,
    canonical_json,
    import_member,
    load_protocol,
    read_object,
    read_kit,
    simulate_matrix,
    validate_member,
    verify_activation,
    verify_slot_kit,
    write_json,
)


DEFAULT_PROTOCOL = "benchmark/formal_backup_set_member_intake_v1_protocol.json"


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

    build = commands.add_parser("build-slot-kit")
    build.add_argument("--members", type=int, choices=(2, 3, 4), required=True)
    build.add_argument("--slot", type=int, required=True)
    build.add_argument("--challenge", required=True)
    build.add_argument("--issued-epoch", type=int, default=10_000)
    build.add_argument("--output", required=True)

    verify = commands.add_parser("verify-member")
    verify.add_argument("--kit", required=True)
    verify.add_argument("--attestation", required=True)
    verify.add_argument("--observation-epoch", type=int, required=True)

    import_dry = commands.add_parser("import-member-dry-run")
    import_dry.add_argument("--kit", required=True)
    import_dry.add_argument("--attestation", required=True)
    import_dry.add_argument("--observation-epoch", type=int, required=True)

    activate = commands.add_parser("activate-set-dry-run")
    activate.add_argument("--members", type=int, choices=(2, 3, 4), required=True)
    activate.add_argument("--bundle", required=True)
    activate.add_argument("--observation-epoch", type=int, required=True)

    simulation = commands.add_parser("simulate-matrix")
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


def _contract_from_kit(
    root: Path, protocol: dict[str, Any], kit: Path
) -> dict[str, Any]:
    verified = verify_slot_kit(
        kit,
        protocol,
        repository_root=root,
    )
    _manifest, files = read_kit(kit)
    try:
        contract = json.loads(files["slot_contract.json"].decode("utf-8"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupSetIntakeError("slot_contract_invalid") from exc
    if (
        not isinstance(contract, dict)
        or contract.get("contract_sha256") != verified["contract_sha256"]
    ):
        raise BackupSetIntakeError("slot_contract_invalid")
    return contract


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    protocol = load_protocol(_resolve(root, args.protocol), repository_root=root)
    if args.command == "build-slot-kit":
        return build_slot_kit(
            root,
            protocol,
            member_count=args.members,
            slot=args.slot,
            challenge_id=args.challenge,
            issued_epoch=args.issued_epoch,
            output=Path(args.output),
        )
    if args.command in {"verify-member", "import-member-dry-run"}:
        kit = Path(args.kit)
        contract = _contract_from_kit(root, protocol, kit)
        attestation = read_object(Path(args.attestation))
        validated = validate_member(
            root,
            contract,
            attestation,
            observation_epoch=args.observation_epoch,
            require_real=False,
        )
        if args.command == "verify-member":
            return {
                **_status("backup_set_member_qualified", EXIT_READY),
                "attestation_sha256": validated["attestation_sha256"],
                "member_count": contract["member_count"],
                "slot": contract["slot"],
            }
        events = import_member(
            root,
            [],
            contract,
            validated,
            observation_epoch=args.observation_epoch,
            require_real=False,
        )
        return {
            **_status("backup_set_member_qualified", EXIT_READY),
            "member_count": contract["member_count"],
            "slot": contract["slot"],
            "registry_event_count": len(events),
            "registry_tip_sha256": events[-1]["event_sha256"],
            "synthetic_only": validated["synthetic_only"],
        }
    if args.command == "activate-set-dry-run":
        bundle = read_object(Path(args.bundle))
        if set(bundle) != {"attestations", "contracts", "events"}:
            raise BackupSetIntakeError("activation_bundle_schema_invalid")
        events, receipt = activate_set(
            root,
            protocol,
            bundle["events"],
            bundle["contracts"],
            bundle["attestations"],
            member_count=args.members,
            observation_epoch=args.observation_epoch,
            require_real=False,
        )
        verify_activation(
            root,
            protocol,
            receipt,
            bundle["contracts"],
            bundle["attestations"],
            observation_epoch=args.observation_epoch,
            require_real=False,
        )
        return {
            **_status("backup_set_activated", EXIT_READY),
            "receipt": receipt,
            "registry_event_count": len(events),
        }
    if args.command == "simulate-matrix":
        report = simulate_matrix(root, protocol)
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
    except BackupSetIntakeNotReady as exc:
        report = _status(
            "not_ready_missing_real_members",
            EXIT_NOT_READY,
            reason_code=str(exc),
        )
    except (
        BackupSetIntakeError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        report = _status(
            "member_or_activation_violation",
            EXIT_VIOLATION,
            reason_code=(
                str(exc)
                if isinstance(exc, BackupSetIntakeError)
                else "controlled_schema_or_filesystem_violation"
            ),
        )
    try:
        sys.stdout.buffer.write(canonical_json(report))
    except (OSError, TypeError, UnicodeError, ValueError):
        report = _status(
            "member_or_activation_violation",
            EXIT_VIOLATION,
            reason_code="output_unavailable",
        )
        sys.stdout.buffer.write(canonical_json(report))
        return EXIT_VIOLATION
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
