#!/usr/bin/env python3
"""Build and audit offline human-annotator qualification intake."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.human_annotator_qualification_intake import (  # noqa: E402
    EXIT_NOT_READY,
    EXIT_QUALIFIED,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    ROLES,
    SCHEMA_VERSION,
    HumanAnnotatorQualificationError,
    HumanAnnotatorQualificationNotReady,
    audit_readiness,
    build_kit,
    canonical_json,
    import_dry_run,
    load_protocol,
    simulate_matrix,
    verify_submission,
)


DEFAULT_PROTOCOL = (
    ROOT / "benchmark/human_annotator_qualification_intake_v1_protocol.json"
)


class UsageError(RuntimeError):
    """CLI arguments do not match the public contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _result(
    status: str, exit_code: int, reason_code: str | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
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
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-kit")
    build.add_argument("--role", choices=ROLES, required=True)
    build.add_argument("--challenge", required=True)
    build.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-submission")
    verify.add_argument("--kit", type=Path, required=True)
    verify.add_argument("--submission", type=Path, required=True)
    verify.add_argument("--allow-synthetic", action="store_true")

    intake = commands.add_parser("import-dry-run")
    for role in ROLES:
        intake.add_argument(f"--kit-{role.replace('_', '-')}", type=Path, required=True)
        intake.add_argument(
            f"--submission-{role.replace('_', '-')}", type=Path, required=True
        )
    intake.add_argument("--ledger", type=Path, required=True)
    intake.add_argument("--proposal", type=Path)
    intake.add_argument("--allow-synthetic", action="store_true")

    commands.add_parser("simulate-matrix")
    commands.add_parser("audit-readiness")
    return parser


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = args.repository_root.resolve()
    protocol = load_protocol(args.protocol, repository_root=root)
    if args.command == "build-kit":
        report = build_kit(
            root,
            protocol,
            challenge=args.challenge,
            role=args.role,
            output=args.output,
        )
        return report, EXIT_QUALIFIED
    if args.command == "verify-submission":
        submission = verify_submission(
            args.kit,
            args.submission,
            protocol,
            repository_root=root,
            allow_synthetic=args.allow_synthetic,
        )
        return (
            {
                **_result("annotator_roles_qualified", EXIT_QUALIFIED),
                "qualification_sha256": submission["qualification_sha256"],
                "role": submission["role"],
            },
            EXIT_QUALIFIED,
        )
    if args.command == "import-dry-run":
        pairs = [
            (
                getattr(args, f"kit_{role}"),
                getattr(args, f"submission_{role}"),
            )
            for role in ROLES
        ]
        report = import_dry_run(
            pairs,
            args.ledger,
            protocol,
            repository_root=root,
            proposal_path=args.proposal,
            allow_synthetic=args.allow_synthetic,
        )
        return report, EXIT_QUALIFIED
    if args.command == "simulate-matrix":
        return simulate_matrix(root, protocol), EXIT_QUALIFIED
    if args.command == "audit-readiness":
        report = audit_readiness(protocol)
        return report, int(report["exit_code"])
    raise UsageError("unsupported_command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report, exit_code = _run(_parser().parse_args(argv))
    except UsageError:
        report = _result("usage_error", EXIT_USAGE, "invalid_arguments")
        exit_code = EXIT_USAGE
    except HumanAnnotatorQualificationNotReady as exc:
        report = _result(
            "not_ready_missing_real_qualified_principals",
            EXIT_NOT_READY,
            str(exc),
        )
        exit_code = EXIT_NOT_READY
    except (
        HumanAnnotatorQualificationError,
        OSError,
        UnicodeError,
        TypeError,
        ValueError,
    ) as exc:
        reason = (
            str(exc)
            if isinstance(exc, HumanAnnotatorQualificationError)
            else "input_invalid"
        )
        report = _result(
            "qualification_or_role_violation", EXIT_VIOLATION, reason
        )
        exit_code = EXIT_VIOLATION
    _emit(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
