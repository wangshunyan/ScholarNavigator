#!/usr/bin/env python3
"""CLI for human_annotation_assignment_activation_v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholar_agent.evaluation.human_annotation_assignment_activation import (  # noqa: E402
    EXIT_NOT_READY,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    HumanAnnotationAssignmentError,
    HumanAnnotationAssignmentNotReady,
    PROTOCOL,
    ROLES,
    SOURCE_COMMIT,
    audit_readiness,
    canonical_json,
    issue_assignments,
    load_protocol,
    read_object,
    simulate_matrix,
    verify_acknowledgements,
)
from scholar_agent.evaluation.human_annotator_qualification_intake import (  # noqa: E402
    load_protocol as load_qualification_protocol,
    verify_submission,
)


DEFAULT_PROTOCOL = (
    ROOT / "benchmark/human_annotation_assignment_activation_v1_protocol.json"
)


class UsageError(RuntimeError):
    """CLI arguments are incomplete."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def _result(status: str, code: int, reason: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "exit_code": code,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "schema_version": "1",
        "status": status,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)

    issue = commands.add_parser("issue-dry-run")
    for role in ROLES:
        key = role.replace("_", "-")
        issue.add_argument(f"--{key}-kit", type=Path, required=True)
        issue.add_argument(f"--{key}-submission", type=Path, required=True)
        issue.add_argument(f"--{key}-challenge", required=True)
    issue.add_argument("--output", type=Path, required=True)
    issue.add_argument("--ledger", type=Path, required=True)

    receipt = commands.add_parser("verify-receipt")
    for role in ROLES:
        key = role.replace("_", "-")
        receipt.add_argument(f"--{key}-bundle", type=Path, required=True)
        receipt.add_argument(f"--{key}-receipt", type=Path, required=True)
    receipt.add_argument("--ledger", type=Path, required=True)

    commands.add_parser("simulate-matrix")
    commands.add_parser("audit-readiness")
    return parser


def _qualification_protocol(
    repository_root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    return load_qualification_protocol(
        repository_root / protocol["bindings"]["qualification"]["path"],
        repository_root=repository_root,
    )


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    repository_root = args.repository_root.resolve()
    protocol = load_protocol(args.protocol, repository_root=repository_root)
    if args.command == "prepare":
        value = {
            "assignment_protocol_sha256": protocol["protocol_sha256"],
            "delivery_bundle_sha256": protocol["bindings"]["delivery_bundle"][
                "sha256"
            ],
            "formal_validation_complete": False,
            "item_count_per_annotator": 471,
            "label_intake_allowed": False,
            "protocol": PROTOCOL,
            "roles": list(ROLES),
            "schema_version": "1",
            "source_commit": SOURCE_COMMIT,
            "state": "prepared",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(value))
        result = dict(value)
        result.update({"exit_code": EXIT_READY, "status": "assignment_chain_ready"})
        return result, EXIT_READY
    if args.command == "simulate-matrix":
        value = simulate_matrix(
            repository_root,
            protocol,
            _qualification_protocol(repository_root, protocol),
        )
        return value, EXIT_READY
    if args.command == "audit-readiness":
        value = audit_readiness(protocol)
        return value, EXIT_NOT_READY
    if args.command == "issue-dry-run":
        qualification_protocol = _qualification_protocol(
            repository_root, protocol
        )
        qualifications = []
        challenges = {}
        for role in ROLES:
            prefix = role
            kit = getattr(args, f"{prefix}_kit")
            submission = getattr(args, f"{prefix}_submission")
            qualifications.append(
                verify_submission(
                    kit,
                    submission,
                    qualification_protocol,
                    repository_root=repository_root,
                    allow_synthetic=False,
                )
            )
            challenges[role] = getattr(args, f"{prefix}_challenge")
        value = issue_assignments(
            repository_root,
            protocol,
            qualifications,
            challenges=challenges,
            output_root=args.output,
            ledger_path=args.ledger,
            allow_synthetic=False,
        )
        return value, EXIT_READY
    if args.command == "verify-receipt":
        bundles = {
            role: getattr(args, f"{role}_bundle") for role in ROLES
        }
        receipts = [getattr(args, f"{role}_receipt") for role in ROLES]
        value = verify_acknowledgements(
            bundles,
            receipts,
            args.ledger,
            protocol,
            repository_root=repository_root,
        )
        return value, EXIT_READY
    raise UsageError("unknown command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        value, code = _run(args)
    except UsageError:
        value, code = _result("usage_error", EXIT_USAGE, "usage_error"), EXIT_USAGE
    except HumanAnnotationAssignmentNotReady as exc:
        value, code = (
            _result(
                "not_ready_missing_real_qualified_principals",
                EXIT_NOT_READY,
                str(exc),
            ),
            EXIT_NOT_READY,
        )
    except (
        HumanAnnotationAssignmentError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        reason = (
            str(exc)
            if isinstance(exc, HumanAnnotationAssignmentError)
            else "input_unavailable_or_invalid"
        )
        value, code = (
            _result("assignment_or_blinding_violation", EXIT_VIOLATION, reason),
            EXIT_VIOLATION,
        )
    _emit(value)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
