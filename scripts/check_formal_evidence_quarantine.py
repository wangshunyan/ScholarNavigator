#!/usr/bin/env python3
"""Verify formal evidence quarantine boundaries without loading real evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.formal_evidence_quarantine import (  # noqa: E402
    EXIT_BLOCKED,
    EXIT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    SCHEMA_VERSION,
    QuarantineError,
    audit_contamination,
    build_intake_manifest,
    canonical_json,
    current_readiness,
    load_intake_manifest,
    load_protocol,
    verify_boundaries,
    verify_intake_manifest,
)
from scholar_agent.evaluation.formal_validation_preregistration import (  # noqa: E402
    PreregistrationError,
    assert_current_preregistration,
)


DEFAULT_PROTOCOL = ROOT / "benchmark/formal_evidence_quarantine_v1_protocol.json"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Fail-closed formal evidence quarantine gate.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--repository-root", default=str(ROOT))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-boundaries")
    intake = commands.add_parser("intake-dry-run")
    intake.add_argument("--artifact", required=True)
    intake.add_argument("--evidence-root", required=True)
    intake.add_argument("--evidence-type", required=True)
    intake.add_argument("--binding", required=True)
    intake.add_argument("--chronology", required=True)
    intake.add_argument("--output")
    contamination = commands.add_parser("audit-contamination")
    contamination.add_argument("--manifest", required=True)
    contamination.add_argument("--evidence-root", required=True)
    contamination.add_argument("--changes", required=True)
    commands.add_parser("audit-readiness")
    return parser


def _read_object(path: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuarantineError("cli_json_input_invalid") from exc
    if not isinstance(value, dict):
        raise QuarantineError("cli_json_root_not_object")
    return value


def _emit(value: dict[str, object]) -> None:
    sys.stdout.buffer.write(canonical_json(value))


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        assert_current_preregistration(root)
        protocol = load_protocol(Path(args.protocol))
        if args.command == "verify-boundaries":
            report = verify_boundaries(root, protocol)
        elif args.command == "audit-readiness":
            report = current_readiness(root, protocol)
        elif args.command == "intake-dry-run":
            manifest = build_intake_manifest(
                evidence_path=Path(args.artifact),
                evidence_root=Path(args.evidence_root),
                evidence_type=args.evidence_type,
                evidence_protocol_version="dry-run-v1",
                input_binding=_read_object(args.binding),
                chronology=_read_object(args.chronology),
                protocol=protocol,
                synthetic_only=True,
            )
            verify_intake_manifest(manifest, evidence_root=Path(args.evidence_root), protocol=protocol)
            payload = manifest.model_dump(mode="json")
            if args.output:
                try:
                    Path(args.output).write_bytes(canonical_json(payload))
                except (OSError, UnicodeError) as exc:
                    raise QuarantineError("cli_output_unavailable") from exc
            report = {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "status": "quarantine_controls_ready",
                "exit_code": EXIT_READY,
                "intake_id": manifest.intake_id,
                "synthetic_only": True,
            }
        else:
            manifest = load_intake_manifest(Path(args.manifest))
            verify_intake_manifest(
                manifest,
                evidence_root=Path(args.evidence_root),
                protocol=protocol,
                repository_root=None if manifest.synthetic_only else root,
            )
            changes = _read_object(args.changes)
            paths = changes.get("paths")
            if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
                raise QuarantineError("change_paths_invalid")
            report = audit_contamination(manifest, paths, protocol)
        _emit(report)
        return int(report["exit_code"])
    except UsageError:
        report = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "status": "usage_error", "exit_code": EXIT_USAGE}
        _emit(report)
        return EXIT_USAGE
    except (QuarantineError, PreregistrationError) as exc:
        report = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "status": "violation", "exit_code": EXIT_VIOLATION, "error_code": str(exc), "formal_validation_complete": False}
        _emit(report)
        return EXIT_VIOLATION
    except (KeyError, OSError, UnicodeError, ValueError, TypeError):
        report = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "status": "violation", "exit_code": EXIT_VIOLATION, "error_code": "cli_filesystem_or_input_error", "formal_validation_complete": False}
        _emit(report)
        return EXIT_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
