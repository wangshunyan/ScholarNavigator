#!/usr/bin/env python3
"""Audit or exercise formal_validation_clearance_v1 offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_validation_clearance import (  # noqa: E402
    EXIT_BLOCKED,
    EXIT_USAGE,
    EXIT_VALID,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    ClearanceBlocked,
    ClearanceError,
    build_current_evidence,
    canonical_json,
    evaluate,
    issue_receipt,
    load_protocol,
    verify_receipt,
    write_json,
)
from scholar_agent.evaluation.evidence_revocation import (  # noqa: E402
    ActiveIncident,
    RevocationError,
    assert_no_active_incident,
)
from scholar_agent.evaluation.formal_validation_preregistration import (  # noqa: E402
    PreregistrationError,
    assert_current_preregistration,
)


DEFAULT_PROTOCOL = ROOT / "benchmark/formal_validation_clearance_v1_protocol.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Fail-closed formal validation clearance gate.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--repository-root", default=str(ROOT))
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-current")
    audit.add_argument("--output")
    evaluate_cmd = commands.add_parser("evaluate")
    evaluate_cmd.add_argument("--evidence", required=True)
    evaluate_cmd.add_argument("--output")
    issue = commands.add_parser("issue-receipt")
    issue.add_argument("--evidence", required=True)
    issue.add_argument("--receipt", required=True)
    verify = commands.add_parser("verify-receipt")
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--output")
    return parser


def _read(path: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClearanceError("cli_json_input_invalid") from exc
    if not isinstance(value, dict):
        raise ClearanceError("cli_json_root_not_object")
    return value


def _emit(value: dict[str, object], output: str | None = None) -> None:
    if output:
        write_json(Path(output), value)
    sys.stdout.buffer.write(canonical_json(value))


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        assert_no_active_incident(root, target="clearance_receipt")
        assert_current_preregistration(root)
        protocol = load_protocol(Path(args.protocol))
        if args.command == "audit-current":
            report = evaluate(build_current_evidence(protocol, repository_root=root))
            _emit(report, args.output)
            return int(report["exit_code"])
        evidence = _read(args.evidence)
        if args.command == "evaluate":
            report = evaluate(evidence)
            _emit(report, args.output)
            return int(report["exit_code"])
        if args.command == "issue-receipt":
            receipt = issue_receipt(evidence, protocol)
            write_json(Path(args.receipt), receipt, exclusive=True)
            _emit(receipt)
            return EXIT_VALID
        receipt = _read(args.receipt)
        report = verify_receipt(receipt, evidence, protocol)
        _emit(report, args.output)
        return int(report["exit_code"])
    except UsageError:
        report = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "status": "usage_error", "exit_code": EXIT_USAGE}
        _emit(report)
        return EXIT_USAGE
    except (ClearanceBlocked, ActiveIncident) as exc:
        report = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "status": "blocked", "exit_code": EXIT_BLOCKED, "error_code": str(exc), "formal_validation_complete": False}
        _emit(report)
        return EXIT_BLOCKED
    except (ClearanceError, RevocationError, PreregistrationError) as exc:
        report = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "status": "invalid", "exit_code": EXIT_VIOLATION, "error_code": str(exc), "formal_validation_complete": False}
        _emit(report)
        return EXIT_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
