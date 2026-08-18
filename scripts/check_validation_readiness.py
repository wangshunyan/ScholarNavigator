#!/usr/bin/env python3
"""Generate or verify validation_readiness_bundle_v1 offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.validation_readiness import (  # noqa: E402
    EXIT_EVIDENCE_OR_CLAIM_VIOLATION,
    EXIT_NOT_READY_MISSING_REQUIRED_EVIDENCE,
    EXIT_READY_WITH_DECLARED_BLOCKERS,
    EXIT_USAGE_ERROR,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    ValidationReadinessError,
    ValidationReadinessNotReady,
    build_release_files,
    canonical_json,
    load_contract,
    stable_tree_sha256,
    verify_release_files,
    write_release_files,
)
from scholar_agent.evaluation.evidence_revocation import (  # noqa: E402
    ActiveIncident,
    RevocationError,
    assert_no_active_incident,
)


DEFAULT_CONTRACT = "benchmark/validation_readiness_bundle_v1_contract.json"
DEFAULT_BUNDLE = "benchmark/validation_readiness_bundle_v1_release"


class UsageError(RuntimeError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = Parser(description="Generate or verify the offline readiness bundle.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--contract", default=DEFAULT_CONTRACT)
        command.add_argument("--bundle", default=DEFAULT_BUNDLE)
        command.add_argument("--repository-root", default=str(ROOT))
    return parser


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(canonical_json(value))


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        root = Path(args.repository_root).resolve()
        assert_no_active_incident(root, target="validation_readiness_bundle")
        contract = load_contract(root / args.contract)
        bundle = root / args.bundle
        if args.command == "generate":
            files = build_release_files(contract, repository_root=root)
            tree_sha256 = write_release_files(bundle, files)
            report = verify_release_files(contract, bundle, repository_root=root)
            if report["bundle_tree_sha256"] != tree_sha256:
                raise ValidationReadinessError("post_write_tree_hash_drift")
            _emit(report)
            return EXIT_READY_WITH_DECLARED_BLOCKERS
        report = verify_release_files(contract, bundle, repository_root=root)
        _emit(report)
        return EXIT_READY_WITH_DECLARED_BLOCKERS
    except UsageError:
        _emit(
            {
                "exit_code": EXIT_USAGE_ERROR,
                "protocol": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
                "status": "usage_error",
            }
        )
        return EXIT_USAGE_ERROR
    except (ValidationReadinessNotReady, ActiveIncident) as exc:
        _emit(
            {
                "error_code": str(exc),
                "exit_code": EXIT_NOT_READY_MISSING_REQUIRED_EVIDENCE,
                "protocol": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
                "status": "not_ready_missing_required_evidence",
            }
        )
        return EXIT_NOT_READY_MISSING_REQUIRED_EVIDENCE
    except (ValidationReadinessError, RevocationError) as exc:
        _emit(
            {
                "error_code": str(exc),
                "exit_code": EXIT_EVIDENCE_OR_CLAIM_VIOLATION,
                "protocol": PROTOCOL_VERSION,
                "schema_version": SCHEMA_VERSION,
                "status": "evidence_or_claim_violation",
            }
        )
        return EXIT_EVIDENCE_OR_CLAIM_VIOLATION


if __name__ == "__main__":
    raise SystemExit(main())
