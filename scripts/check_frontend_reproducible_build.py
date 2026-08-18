#!/usr/bin/env python3
"""Run or verify frontend_reproducible_build_v1 without network access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.frontend_reproducible_build import (  # noqa: E402
    EXIT_UPSTREAM,
    EXIT_USAGE,
    EXIT_VIOLATION,
    PROTOCOL,
    ReleaseCandidateError,
    ReleaseCandidateNotReady,
    canonical_json,
    run_qualification,
    verify_evidence,
    write_json,
)


DEFAULT_PROTOCOL = ROOT / "benchmark/frontend_reproducible_build_v1_protocol.json"
DEFAULT_CONTRACT = ROOT / "benchmark/frontend_reproducible_build_v1_release_contract.json"
DEFAULT_EVIDENCE = ROOT / "benchmark/frontend_reproducible_build_v1_evidence/current.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output", required=True)
    run.add_argument("--report")
    verify = commands.add_parser("verify")
    verify.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    return parser


def _error(status: str, exit_code: int, reason: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol": PROTOCOL,
        "status": status,
        "exit_code": exit_code,
        "reason": reason,
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
        "formal_validation_complete": False,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        if args.command == "run":
            report = run_qualification(ROOT, protocol, contract, Path(args.output))
            if args.report:
                write_json(Path(args.report), report)
        else:
            evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
            report = verify_evidence(evidence, protocol, contract)
    except ReleaseCandidateNotReady as exc:
        report = _error("not_qualified_upstream_limitation", EXIT_UPSTREAM, str(exc))
    except (ReleaseCandidateError, OSError, ValueError, json.JSONDecodeError) as exc:
        reason = str(exc) if isinstance(exc, ReleaseCandidateError) else "controlled_input_or_io_failure"
        report = _error("reproducibility_or_runtime_violation", EXIT_VIOLATION, reason)
    except SystemExit:
        report = _error("usage_error", EXIT_USAGE, "invalid_arguments")
    sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
