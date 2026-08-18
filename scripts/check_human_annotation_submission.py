#!/usr/bin/env python3
"""CLI for human_annotation_submission_intake_v1."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_PROTOCOL = (
    ROOT / "benchmark/human_annotation_submission_intake_v1_protocol.json"
)
PROTOCOL = "human_annotation_submission_intake_v1"
SCHEMA_VERSION = "1"
SOURCE_COMMIT = "dcbede0928d6c15d0c9a68333e53837537fe1249"
FROZEN_PROTOCOL_SHA256 = (
    "d211a40c3e59a21e5cfdf6d73aa9938e2137c963da7832d3420ea4c8613e16d7"
)
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3
EXIT_USAGE = 4
ANNOTATOR_ROLES = ("annotator_a", "annotator_b")
ZERO_SHA256 = "0" * 64
MAX_PROTOCOL_BYTES = 1024 * 1024
EXECUTION_ZERO = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "real_label_count": 0,
    "snapshot_write_count": 0,
}
FORMAL_BLOCKERS = (
    "full1000_incomplete",
    "human_precision_missing",
    "official_scorer_schema_missing",
)
PROTOCOL_FIELDS = {
    "bindings",
    "execution",
    "formal_validation_complete",
    "intake",
    "protocol",
    "protocol_sha256",
    "schema_version",
    "source_commit",
    "state_machine",
    "synthetic_scenarios",
}


class UsageError(RuntimeError):
    """CLI usage is incomplete or contradictory."""


class SubmissionViolation(RuntimeError):
    """Production submission validation rejected an input."""


class SubmissionNotReady(SubmissionViolation):
    """Both complete real submissions are not yet available."""


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
    ).encode("utf-8")


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def _result(status: str, code: int, reason: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "exit_code": code,
        "formal_validation_complete": False,
        "protocol": PROTOCOL,
        "schema_version": "1",
        "statistics": None,
        "status": status,
    }
    if reason is not None:
        value["reason"] = reason
    return value


def _load_audit_protocol(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SubmissionViolation("protocol_unavailable") from exc
    if len(raw) > MAX_PROTOCOL_BYTES:
        raise SubmissionViolation("protocol_schema_invalid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid_constant:{token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise SubmissionViolation("protocol_schema_invalid") from exc
    if not isinstance(value, dict) or set(value) != PROTOCOL_FIELDS:
        raise SubmissionViolation("protocol_schema_invalid")
    payload = dict(value)
    payload["protocol_sha256"] = ZERO_SHA256
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    if (
        value["protocol"] != PROTOCOL
        or value["schema_version"] != SCHEMA_VERSION
        or value["source_commit"] != SOURCE_COMMIT
        or value["execution"] != EXECUTION_ZERO
        or value["formal_validation_complete"] is not False
        or value["protocol_sha256"] != FROZEN_PROTOCOL_SHA256
        or digest != FROZEN_PROTOCOL_SHA256
    ):
        raise SubmissionViolation("protocol_binding_invalid")
    return value


def _audit_readiness(_protocol: dict[str, Any]) -> dict[str, Any]:
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
        ],
        "execution": dict(EXECUTION_ZERO),
        "exit_code": EXIT_NOT_READY,
        "formal_blockers": list(FORMAL_BLOCKERS),
        "formal_validation_complete": False,
        "human_precision_verified": False,
        "protocol": PROTOCOL,
        "real_label_count": 0,
        "schema_version": SCHEMA_VERSION,
        "state": "not_ready_missing_real_submissions",
        "statistics": None,
        "status": "not_ready_missing_real_submissions",
    }


def _load_runtime() -> tuple[ModuleType, ModuleType]:
    submission = importlib.import_module(
        "scholar_agent.evaluation.human_annotation_submission_intake"
    )
    assignment = importlib.import_module(
        "scholar_agent.evaluation.human_annotation_assignment_activation"
    )
    return submission, assignment


def _add_assignment_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assignment-ledger", type=Path, required=True)
    for role in ("annotator-a", "annotator-b", "adjudicator"):
        parser.add_argument(f"--{role}-bundle", type=Path, required=True)
        parser.add_argument(
            f"--{role}-assignment-receipt", type=Path, required=True
        )


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-submission")
    _add_assignment_inputs(verify)
    verify.add_argument(
        "--role", choices=list(ANNOTATOR_ROLES), required=True
    )
    verify.add_argument("--submission", type=Path, required=True)
    verify.add_argument("--submission-receipt", type=Path, required=True)

    ingest = commands.add_parser("import-dry-run")
    _add_assignment_inputs(ingest)
    for role in ("annotator-a", "annotator-b"):
        ingest.add_argument(f"--{role}-submission", type=Path)
        ingest.add_argument(f"--{role}-submission-receipt", type=Path)
    ingest.add_argument("--ledger", type=Path, required=True)

    queue = commands.add_parser("build-adjudication-queue")
    _add_assignment_inputs(queue)
    for role in ("annotator-a", "annotator-b"):
        queue.add_argument(f"--{role}-submission", type=Path, required=True)
        queue.add_argument(
            f"--{role}-submission-receipt", type=Path, required=True
        )
    queue.add_argument("--intake-ledger", type=Path, required=True)
    queue.add_argument("--queue-output", type=Path, required=True)
    queue.add_argument("--operator-mapping-output", type=Path, required=True)

    commands.add_parser("simulate-matrix")
    commands.add_parser("audit-readiness")
    return parser


def _assignment_inputs(args: argparse.Namespace) -> tuple[
    dict[str, Path], dict[str, Path], Path
]:
    bundles = {
        "annotator_a": args.annotator_a_bundle,
        "annotator_b": args.annotator_b_bundle,
        "adjudicator": args.adjudicator_bundle,
    }
    receipts = {
        "annotator_a": args.annotator_a_assignment_receipt,
        "annotator_b": args.annotator_b_assignment_receipt,
        "adjudicator": args.adjudicator_assignment_receipt,
    }
    return bundles, receipts, args.assignment_ledger


def _submission_inputs(
    args: argparse.Namespace,
) -> dict[str, tuple[Path, Path]]:
    values: dict[str, tuple[Path, Path]] = {}
    for role in ANNOTATOR_ROLES:
        submission = getattr(args, f"{role}_submission")
        receipt = getattr(args, f"{role}_submission_receipt")
        if (submission is None) != (receipt is None):
            raise UsageError("submission_and_receipt_must_be_paired")
        if submission is not None and receipt is not None:
            values[role] = (submission, receipt)
    return values


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    repository_root = args.repository_root.resolve()
    if args.command == "audit-readiness":
        protocol = _load_audit_protocol(args.protocol)
        return _audit_readiness(protocol), EXIT_NOT_READY

    runtime, assignment = _load_runtime()
    try:
        protocol = runtime.load_protocol(
            args.protocol, repository_root=repository_root
        )
        if args.command == "simulate-matrix":
            return runtime.simulate_matrix(repository_root, protocol), EXIT_READY

        bundles, receipts, assignment_ledger = _assignment_inputs(args)
        context = runtime.validate_assignment_context(
            repository_root,
            protocol,
            bundle_paths=bundles,
            assignment_receipt_paths=receipts,
            assignment_ledger_path=assignment_ledger,
        )
        if args.command == "verify-submission":
            submission, receipt = runtime.verify_locked_submission(
                args.submission,
                args.submission_receipt,
                role=args.role,
                assignment_context=context,
                protocol=protocol,
                repository_root=repository_root,
                allow_synthetic=False,
            )
            return (
                {
                    "execution": protocol["execution"],
                    "exit_code": EXIT_READY,
                    "formal_validation_complete": False,
                    "human_precision_verified": False,
                    "item_count": len(submission.labels),
                    "locked_labels_sha256": submission.labels_sha256,
                    "principal_identity_sha256": receipt["principal_id"],
                    "protocol": PROTOCOL,
                    "real_label_count": 0,
                    "role": args.role,
                    "schema_version": SCHEMA_VERSION,
                    "statistics": None,
                    "status": "submission_chain_ready",
                },
                EXIT_READY,
            )
        submissions = _submission_inputs(args)
        if args.command == "import-dry-run":
            if not submissions:
                raise UsageError("at_least_one_submission_required")
            value = runtime.intake_submissions(
                repository_root,
                protocol,
                assignment_context=context,
                submissions=submissions,
                ledger_path=args.ledger,
                allow_synthetic=False,
            )
            return value, int(value["exit_code"])
        if args.command == "build-adjudication-queue":
            if set(submissions) != set(ANNOTATOR_ROLES):
                raise UsageError("two_complete_submissions_required")
            value = runtime.build_adjudication_queue(
                repository_root,
                protocol,
                assignment_context=context,
                submissions=submissions,
                ledger_path=args.intake_ledger,
                queue_path=args.queue_output,
                operator_mapping_path=args.operator_mapping_output,
                allow_synthetic=False,
            )
            return value, EXIT_READY
        raise UsageError("unknown_command")
    except runtime.HumanAnnotationSubmissionNotReady as exc:
        raise SubmissionNotReady(str(exc)) from exc
    except (
        runtime.HumanAnnotationSubmissionError,
        assignment.HumanAnnotationAssignmentError,
    ) as exc:
        raise SubmissionViolation(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        value, code = _run(args)
    except UsageError:
        value, code = _result("usage_error", EXIT_USAGE, "usage_error"), EXIT_USAGE
    except SubmissionNotReady as exc:
        value, code = (
            _result(
                "not_ready_missing_real_submissions",
                EXIT_NOT_READY,
                str(exc),
            ),
            EXIT_NOT_READY,
        )
    except (
        SubmissionViolation,
        KeyError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        reason = (
            str(exc)
            if isinstance(
                exc,
                SubmissionViolation,
            )
            else "input_unavailable_or_invalid"
        )
        value, code = (
            _result(
                "submission_or_blinding_violation",
                EXIT_VIOLATION,
                reason,
            ),
            EXIT_VIOLATION,
        )
    _emit(value)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
