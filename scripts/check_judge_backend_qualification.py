#!/usr/bin/env python3
"""Analyze and qualify strict structured-output judge backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.core.env_loader import load_project_env  # noqa: E402
from scholar_agent.evaluation.execution_determinism import (  # noqa: E402
    tree_signature,
)
from scholar_agent.evaluation.judge_backend_qualification import (  # noqa: E402
    CONTRACT_VERSION,
    EXIT_NOT_ELIGIBLE,
    EXIT_QUALIFIED,
    EXIT_USAGE_ERROR,
    EXIT_VIOLATION,
    GATE_NAME,
    SCHEMA_VERSION,
    SCORE_SCOPE,
    QualificationError,
    QualificationNotEligible,
    analyze_frozen_evidence,
    candidate_from_runtime,
    load_protocol,
    qualify_run,
    run_probe,
    verify_published,
    write_frozen_analysis,
)
from scholar_agent.llm.provider import (  # noqa: E402
    OpenAICompatibleLLMClient,
    get_llm_request_options,
    get_llm_runtime_config,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "judge_backend_qualification_v1_protocol.json"
DEFAULT_RUN_DIR = ROOT / "outputs" / "benchmark_runs" / "judge_backend_qualification_v1"
DEFAULT_PUBLISH_DIR = ROOT / "benchmark" / "judge_backend_qualification_v1_result"
DEFAULT_SNAPSHOT_ROOT = ROOT / "outputs" / "benchmark_snapshots"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise QualificationError("usage_error")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description=(
            "Qualify strict judge-backend protocol conformance. This does not "
            "judge relevance or compute a quality score."
        )
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--publish-dir", default=str(DEFAULT_PUBLISH_DIR))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("analyze-frozen")
    commands.add_parser("probe")
    commands.add_parser("qualify")
    commands.add_parser("verify")
    return parser


def _emit(report: dict[str, Any]) -> None:
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _failure(*, status: str, exit_code: int, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": exit_code,
        "reason": reason,
        "score_scope": SCORE_SCOPE,
        "labels_persisted": False,
        "quality_metrics_computed": False,
        "execution": {
            "academic_api_request_count": 0,
            "other_network_request_count": 0,
            "snapshot_write_count": 0,
        },
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    publish_dir = Path(args.publish_dir).resolve()
    protocol = load_protocol(Path(args.protocol).resolve(), repository_root=root)
    if args.command == "analyze-frozen":
        report = analyze_frozen_evidence(protocol, repository_root=root)
        write_frozen_analysis(run_dir, report)
        return report
    if args.command == "verify":
        return verify_published(protocol, publish_dir=publish_dir)
    if args.command == "qualify":
        return qualify_run(
            protocol,
            repository_root=root,
            run_dir=run_dir,
            publish_dir=publish_dir,
        )

    if not (run_dir / "frozen_analysis.json").is_file():
        raise QualificationNotEligible("frozen_analysis_required_before_probe")
    # This is the only command that loads project runtime configuration. The
    # resulting descriptor intentionally omits endpoint, host, headers, and key.
    load_project_env(root)
    runtime = get_llm_runtime_config()
    request_options = get_llm_request_options()
    candidate = candidate_from_runtime(
        provider=runtime.provider,
        model=runtime.model,
        available=runtime.available,
        reason=runtime.reason,
        request_options=request_options,
    )
    return run_probe(
        protocol,
        repository_root=root,
        run_dir=run_dir,
        candidates=[candidate],
        client_factory=lambda _candidate: OpenAICompatibleLLMClient.from_env(),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except QualificationError:
        report = _failure(
            status="usage_error",
            exit_code=EXIT_USAGE_ERROR,
            reason="invalid_arguments",
        )
        _emit(report)
        return EXIT_USAGE_ERROR

    snapshot_before = tree_signature(DEFAULT_SNAPSHOT_ROOT)
    try:
        report = _execute(args)
    except QualificationNotEligible as exc:
        report = _failure(
            status="not_eligible",
            exit_code=EXIT_NOT_ELIGIBLE,
            reason=str(exc),
        )
    except QualificationError as exc:
        report = _failure(
            status="integrity_or_conformance_violation",
            exit_code=EXIT_VIOLATION,
            reason=str(exc),
        )
    except (OSError, ValueError, json.JSONDecodeError):
        report = _failure(
            status="usage_error",
            exit_code=EXIT_USAGE_ERROR,
            reason="input_or_runtime_error",
        )
    if snapshot_before != tree_signature(DEFAULT_SNAPSHOT_ROOT):
        report = _failure(
            status="integrity_or_conformance_violation",
            exit_code=EXIT_VIOLATION,
            reason="snapshot_tree_modified",
        )
    _emit(report)
    return int(report.get("exit_code", EXIT_QUALIFIED))


if __name__ == "__main__":
    raise SystemExit(main())
