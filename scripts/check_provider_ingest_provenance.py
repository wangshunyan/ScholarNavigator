#!/usr/bin/env python3
"""Verify provider_ingest_provenance_v1 without network access."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.provider_ingest_provenance import (  # noqa: E402
    EXIT_NOT_ELIGIBLE,
    EXIT_USAGE_ERROR,
    ProviderIngestError,
    ProviderIngestNotEligible,
    audit_frozen_record162,
    deterministic_fixture_matrix,
    replay_capture_bundle,
    verify_capture_bundle,
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ProviderIngestError(f"usage_error:{message}")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Audit parser-pre provider response provenance.")
    parser.add_argument("--repository-root", default=str(ROOT))
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--raw-archive", required=True)
    verify.add_argument("--resource-ledger", required=True)
    replay = commands.add_parser("replay-parser")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--raw-archive", required=True)
    commands.add_parser("check-fixtures")
    commands.add_parser("audit-frozen")
    return parser


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def main() -> int:
    try:
        args = _parser().parse_args()
        root = Path(args.repository_root).resolve()
        if args.command == "verify":
            report = verify_capture_bundle(
                Path(args.bundle),
                Path(args.raw_archive),
                resource_ledger_path=(
                    Path(args.resource_ledger)
                ),
            )
        elif args.command == "replay-parser":
            report = replay_capture_bundle(Path(args.bundle), Path(args.raw_archive))
        elif args.command == "check-fixtures":
            with tempfile.TemporaryDirectory(prefix="spar-provider-ingest-") as value:
                report = deterministic_fixture_matrix(Path(value))
        elif args.command == "audit-frozen":
            report = audit_frozen_record162(root)
        else:  # pragma: no cover - argparse closes this branch
            raise ProviderIngestError("usage_error:unknown_command")
        _emit(report)
        return int(report["exit_code"])
    except ProviderIngestNotEligible as exc:
        _emit(
            {
                "protocol": "provider_ingest_provenance_v1",
                "schema_version": "1",
                "status": "not_eligible",
                "exit_code": EXIT_NOT_ELIGIBLE,
                "reason_code": str(exc),
            }
        )
        return EXIT_NOT_ELIGIBLE
    except (ProviderIngestError, OSError, UnicodeError, ValueError, TypeError) as exc:
        reason = "usage_error" if str(exc).startswith("usage_error:") else "provenance_input_invalid"
        exit_code = EXIT_USAGE_ERROR if reason == "usage_error" else 2
        _emit(
            {
                "protocol": "provider_ingest_provenance_v1",
                "schema_version": "1",
                "status": reason,
                "exit_code": exit_code,
                "reason_code": reason,
            }
        )
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
