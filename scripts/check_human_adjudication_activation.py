#!/usr/bin/env python3
"""CLI for human_adjudication_activation_v1."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_PROTOCOL = ROOT / "benchmark/human_adjudication_activation_v1_protocol.json"
PROTOCOL = "human_adjudication_activation_v1"
SOURCE_COMMIT = "e8add06c729156223466d51e8f718cdfb59e9035"
FROZEN_PROTOCOL_SHA256 = (
    "5a792f16ebd0a1b193ab29bcfdc808bc2620c307cec8de5f813ab56988c9cc24"
)
ZERO_SHA256 = "0" * 64
EXIT_READY, EXIT_VIOLATION, EXIT_NOT_READY, EXIT_USAGE = 0, 2, 3, 4
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "official_scorer_call_count": 0,
    "quality_metric_count": 0,
    "real_label_count": 0,
    "snapshot_write_count": 0,
}
FORMAL_BLOCKERS = [
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
]


class UsageError(RuntimeError):
    pass


class Violation(RuntimeError):
    pass


class NotReady(Violation):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def emit(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def _audit_protocol(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise Violation("protocol_unavailable_or_invalid") from exc
    if not isinstance(value, dict):
        raise Violation("protocol_schema_invalid")
    payload = dict(value)
    payload["protocol_sha256"] = ZERO_SHA256
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    if (
        value.get("protocol") != PROTOCOL
        or value.get("source_commit") != SOURCE_COMMIT
        or value.get("schema_version") != "1"
        or value.get("execution") != EXECUTION_ZERO
        or value.get("formal_validation_complete") is not False
        or value.get("protocol_sha256") != FROZEN_PROTOCOL_SHA256
        or digest != FROZEN_PROTOCOL_SHA256
    ):
        raise Violation("protocol_binding_invalid")
    return value


def audit_readiness() -> dict[str, Any]:
    return {
        "blocked_reasons": [
            "annotator_a_real_qualification_missing",
            "annotator_b_real_qualification_missing",
            "adjudicator_real_qualification_missing",
            "annotator_a_assignment_acknowledgement_missing",
            "annotator_b_assignment_acknowledgement_missing",
            "adjudicator_assignment_acknowledgement_missing",
            "annotator_a_locked_submission_missing",
            "annotator_b_locked_submission_missing",
            "real_adjudication_queue_missing",
            "real_adjudication_submission_missing",
        ],
        "execution": EXECUTION_ZERO,
        "exit_code": EXIT_NOT_READY,
        "formal_blockers": FORMAL_BLOCKERS,
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "schema_version": "1",
        "state": "not_ready_missing_real_labels_or_adjudication",
        "statistics": None,
        "status": "not_ready_missing_real_labels_or_adjudication",
    }


def _paths(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Path], dict[str, tuple[Path, Path]]]:
    bundles = {
        role: getattr(args, f"{role}_bundle")
        for role in ("annotator_a", "annotator_b", "adjudicator")
    }
    receipts = {
        role: getattr(args, f"{role}_assignment_receipt")
        for role in ("annotator_a", "annotator_b", "adjudicator")
    }
    submissions = {
        role: (
            getattr(args, f"{role}_submission"),
            getattr(args, f"{role}_submission_receipt"),
        )
        for role in ("annotator_a", "annotator_b")
    }
    return bundles, receipts, submissions


def _add_upstream(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assignment-ledger", type=Path, required=True)
    for role in ("annotator-a", "annotator-b", "adjudicator"):
        parser.add_argument(f"--{role}-bundle", type=Path, required=True)
        parser.add_argument(f"--{role}-assignment-receipt", type=Path, required=True)
    for role in ("annotator-a", "annotator-b"):
        parser.add_argument(f"--{role}-submission", type=Path, required=True)
        parser.add_argument(f"--{role}-submission-receipt", type=Path, required=True)
    parser.add_argument("--submission-ledger", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--operator-mapping", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    value = Parser(description=__doc__)
    value.add_argument("--repository-root", type=Path, default=ROOT)
    value.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = value.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue-dry-run")
    _add_upstream(issue)
    issue.add_argument("--challenge", required=True)
    issue.add_argument("--package-output", type=Path, required=True)
    issue.add_argument("--ledger", type=Path, required=True)
    for name in ("verify-adjudication", "score-dry-run"):
        command = commands.add_parser(name)
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--acknowledgement", type=Path, required=True)
        command.add_argument("--result", type=Path, required=True)
        command.add_argument("--ledger", type=Path, required=True)
        command.add_argument("--submission-ledger", type=Path, required=True)
        if name == "score-dry-run":
            for role in ("annotator-a", "annotator-b"):
                command.add_argument(f"--{role}-submission", type=Path, required=True)
                command.add_argument(
                    f"--{role}-submission-receipt", type=Path, required=True
                )
            command.add_argument("--operator-mapping", type=Path, required=True)
    commands.add_parser("simulate-matrix")
    commands.add_parser("audit-readiness")
    return value


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        if args.command == "audit-readiness":
            _audit_protocol(args.protocol)
            result = audit_readiness()
            emit(result)
            return result["exit_code"]
        runtime = importlib.import_module(
            "scholar_agent.evaluation.human_adjudication_activation"
        )
        protocol = runtime.load_protocol(
            args.protocol, repository_root=args.repository_root
        )
        if args.command == "simulate-matrix":
            result = runtime.simulate_matrix(args.repository_root, protocol)
        elif args.command == "issue-dry-run":
            submission_runtime = importlib.import_module(
                "scholar_agent.evaluation.human_annotation_submission_intake"
            )
            submission_protocol = submission_runtime.load_protocol(
                args.repository_root
                / protocol["bindings"]["submission_intake"]["path"],
                repository_root=args.repository_root,
            )
            bundles, receipts, submissions = _paths(args)
            context = submission_runtime.validate_assignment_context(
                args.repository_root,
                submission_protocol,
                bundle_paths=bundles,
                assignment_receipt_paths=receipts,
                assignment_ledger_path=args.assignment_ledger,
            )
            result = runtime.issue_package(
                args.repository_root,
                protocol,
                assignment_context=context,
                submissions=submissions,
                submission_ledger_path=args.submission_ledger,
                queue_path=args.queue,
                operator_mapping_path=args.operator_mapping,
                challenge=args.challenge,
                package_path=args.package_output,
                ledger_path=args.ledger,
            )
        elif args.command == "verify-adjudication":
            result = runtime.verify_result(
                args.package,
                args.acknowledgement,
                args.result,
                args.ledger,
                args.submission_ledger,
                protocol,
            )
        else:
            submissions = {
                role: (
                    getattr(args, f"{role}_submission"),
                    getattr(args, f"{role}_submission_receipt"),
                )
                for role in ("annotator_a", "annotator_b")
            }
            result = runtime.unlock_statistics(
                args.repository_root,
                protocol,
                package_path=args.package,
                acknowledgement_path=args.acknowledgement,
                result_path=args.result,
                ledger_path=args.ledger,
                submission_ledger_path=args.submission_ledger,
                operator_mapping_path=args.operator_mapping,
                submissions=submissions,
            )
        emit(result)
        return int(result["exit_code"])
    except UsageError as exc:
        emit({"exit_code": EXIT_USAGE, "protocol": PROTOCOL, "reason": "usage_error", "statistics": None, "status": "usage_error"})
        return EXIT_USAGE
    except Exception as exc:  # normalized CLI boundary; never emits traceback/raw input
        name = type(exc).__name__
        if name.endswith("NotReady"):
            code, status = EXIT_NOT_READY, "not_ready_missing_real_labels_or_adjudication"
        else:
            code, status = EXIT_VIOLATION, "adjudication_or_statistics_violation"
        emit(
            {
                "exit_code": code,
                "formal_validation_complete": False,
                "protocol": PROTOCOL,
                "reason": str(exc).split(":", 1)[0] or "validation_failed",
                "statistics": None,
                "status": status,
            }
        )
        return code


if __name__ == "__main__":
    raise SystemExit(main())
