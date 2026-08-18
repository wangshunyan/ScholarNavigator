#!/usr/bin/env python3
"""Build and verify deterministic offline SPAR software release candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.evaluation.release_candidate_reproducibility import (  # noqa: E402
    EXIT_NOT_READY,
    EXIT_USAGE,
    EXIT_VIOLATION,
    ReleaseCandidateError,
    ReleaseCandidateNotReady,
    audit_readiness,
    canonical_json,
    compare_outputs,
    double_build,
    load_contract,
    verify_output,
    write_json,
)
from scholar_agent.evaluation.evidence_revocation import (  # noqa: E402
    ActiveIncident,
    RevocationError,
    assert_no_active_incident,
)


DEFAULT_CONTRACT = ROOT / "benchmark/release_candidate_reproducibility_v1_contract.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    commands = value.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--release", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--first", required=True)
    compare.add_argument("--second", required=True)
    audit = commands.add_parser("audit-readiness")
    audit.add_argument("--evidence")
    for command in (build, verify, compare, audit):
        command.add_argument("--report")
    return value


def error(status: str, code: int, reason: str) -> dict[str, object]:
    return {
        "schema_version": "1",
        "protocol": "release_candidate_reproducibility_v1",
        "status": status,
        "exit_code": code,
        "reason": reason,
        "execution": {"gold_or_qrels_loaded": False, "llm_request_count": 0, "network_request_count": 0, "quality_metric_count": 0, "snapshot_write_count": 0},
        "formal_validation_complete": False,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        assert_no_active_incident(ROOT, target="release_candidate")
        contract = load_contract(Path(args.contract), ROOT)
        if args.command == "build":
            report = double_build(ROOT, contract, Path(args.output))
        elif args.command == "verify":
            report = verify_output(Path(args.release), contract)
        elif args.command == "compare":
            report = compare_outputs(Path(args.first), Path(args.second))
        else:
            report = audit_readiness(ROOT, contract, Path(args.evidence) if args.evidence else None)
        if args.report:
            write_json(Path(args.report), report)
    except (ReleaseCandidateNotReady, ActiveIncident) as exc:
        report = error("not_ready_missing_offline_dependency_or_input", EXIT_NOT_READY, str(exc))
    except (ReleaseCandidateError, RevocationError) as exc:
        report = error("build_or_supply_chain_violation", EXIT_VIOLATION, str(exc))
    except (OSError, ValueError, json.JSONDecodeError):
        report = error("build_or_supply_chain_violation", EXIT_VIOLATION, "controlled_input_or_io_failure")
    except SystemExit:
        report = error("usage_error", EXIT_USAGE, "invalid_arguments")
    sys.stdout.buffer.write(canonical_json(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
