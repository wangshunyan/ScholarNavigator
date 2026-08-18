#!/usr/bin/env python3
"""Plan and verify deterministic change-risk validation scope."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.change_risk_validation import (  # noqa: E402
    EXIT_INCOMPLETE,
    EXIT_SATISFIED,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    ValidationScopeError,
    ValidationScopeIncomplete,
    audit_current,
    build_plan,
    canonical_json,
    commit_changes,
    git_head,
    load_protocol,
    read_json,
    verify_execution,
    worktree_changes,
    worktree_digest,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark/change_risk_validation_v1_protocol.json"
DEFAULT_FRESHNESS = (
    ROOT / "benchmark/validation_evidence_freshness_v1_contract.json"
)
DEFAULT_READINESS = (
    ROOT / "benchmark/validation_readiness_bundle_v1_contract.json"
)


class UsageError(RuntimeError):
    """CLI arguments do not match the public contract."""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--freshness", type=Path, default=DEFAULT_FRESHNESS)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commit_plan = commands.add_parser("plan")
    commit_plan.add_argument("--from", dest="from_commit", required=True)
    commit_plan.add_argument("--to", dest="to_commit", required=True)
    commit_plan.add_argument("--output", type=Path)
    worktree = commands.add_parser("plan-worktree")
    worktree.add_argument("--output", type=Path)
    verify = commands.add_parser("verify-execution")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--attestation", type=Path, required=True)
    commands.add_parser("audit-current")
    return parser


def _result(status: str, exit_code: int, reason: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "exit_code": exit_code,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": status,
    }
    if reason is not None:
        value["reason_code"] = reason
    return value


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    root = args.repository_root.resolve()
    protocol = load_protocol(args.protocol)
    freshness = read_json(args.freshness)
    readiness = read_json(args.readiness)
    if args.command == "plan":
        changes = commit_changes(root, args.from_commit, args.to_commit)
        plan = build_plan(
            changes=changes,
            protocol=protocol,
            freshness=freshness,
            readiness=readiness,
            target={
                "from_commit": args.from_commit,
                "mode": "commit",
                "target_commit": args.to_commit,
                "worktree_sha256": None,
            },
        )
        if args.output:
            write_json(args.output, plan)
        return plan, EXIT_SATISFIED
    if args.command == "plan-worktree":
        changes = worktree_changes(root)
        plan = build_plan(
            changes=changes,
            protocol=protocol,
            freshness=freshness,
            readiness=readiness,
            target={
                "from_commit": git_head(root),
                "mode": "worktree",
                "target_commit": git_head(root),
                "worktree_sha256": worktree_digest(root, changes),
            },
        )
        if args.output:
            write_json(args.output, plan)
        return plan, EXIT_SATISFIED
    if args.command == "verify-execution":
        report = verify_execution(
            read_json(args.plan), read_json(args.attestation)
        )
        return report, EXIT_SATISFIED
    if args.command == "audit-current":
        return audit_current(protocol, freshness), EXIT_SATISFIED
    raise UsageError("unsupported_command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report, exit_code = _run(_parser().parse_args(argv))
    except UsageError:
        report = _result("usage_error", EXIT_USAGE, "invalid_arguments")
        exit_code = EXIT_USAGE
    except ValidationScopeIncomplete as exc:
        report = _result("validation_incomplete", EXIT_INCOMPLETE, str(exc))
        exit_code = EXIT_INCOMPLETE
    except (ValidationScopeError, OSError, UnicodeError, TypeError, ValueError) as exc:
        reason = str(exc) if isinstance(exc, ValidationScopeError) else "input_invalid"
        report = _result("scope_or_execution_violation", EXIT_VIOLATION, reason)
        exit_code = EXIT_VIOLATION
    _emit(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
