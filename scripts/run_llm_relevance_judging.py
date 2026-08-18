#!/usr/bin/env python3
"""Run the blinded Record160 LLM relevance proxy workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Callable
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scholar_agent.core.env_loader import load_project_env  # noqa: E402
from scholar_agent.evaluation.execution_determinism import (  # noqa: E402
    tree_signature,
)
from scholar_agent.evaluation.llm_relevance_judging import (  # noqa: E402
    DEFAULT_HARDENED_RUN_DIR,
    DEFAULT_SNAPSHOT_ROOT,
    EXIT_INCOMPLETE,
    EXIT_INTEGRITY_VIOLATION,
    EXIT_USAGE_ERROR,
    HARDENED_CONTRACT_VERSION,
    LLMRelevanceJudgingError,
    LLMRelevanceJudgingIncomplete,
    incomplete_report,
    load_protocol,
    prepare_run,
    publish_incomplete_audit,
    run_adjudication,
    run_judge_round,
    runtime_binding_for_client,
    score_run,
    usage_error_report,
    verify_run,
    violation_report,
)
from scholar_agent.llm.provider import (  # noqa: E402
    OpenAICompatibleLLMClient,
    get_llm_request_options,
    get_llm_runtime_config,
)


DEFAULT_PROTOCOL = ROOT / "benchmark" / "llm_relevance_judging_v1_1_protocol.json"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise LLMRelevanceJudgingError("usage_error")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description=(
            "Run blinded LLM relevance judging. Results are an internal LLM "
            "proxy, not human Precision or an official score."
        )
    )
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-dir", default=str(DEFAULT_HARDENED_RUN_DIR))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    judge = commands.add_parser("judge")
    judge.add_argument(
        "--round",
        choices=("independent_1", "independent_2"),
        required=True,
    )
    judge.add_argument("--max-batches", type=int)
    adjudicate = commands.add_parser("adjudicate")
    adjudicate.add_argument("--max-batches", type=int)
    score = commands.add_parser("score")
    score.add_argument("--publish-dir")
    verify = commands.add_parser("verify")
    verify.add_argument("--publish-incomplete-dir")
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


def _runtime(contract: str) -> tuple[
    OpenAICompatibleLLMClient,
    dict[str, Any],
    Callable[[], OpenAICompatibleLLMClient],
]:
    # This is the only place where the real workflow loads project runtime
    # configuration. No configuration value is printed or persisted.
    load_project_env(ROOT)
    runtime = get_llm_runtime_config()
    if not runtime.available:
        raise LLMRelevanceJudgingIncomplete(
            runtime.reason or "llm_runtime_unavailable"
        )
    client = OpenAICompatibleLLMClient.from_env()
    binding = runtime_binding_for_client(
        client,
        provider=runtime.provider,
        model=str(runtime.model),
        request_options=get_llm_request_options(),
        contract=contract,
    )
    return client, binding, OpenAICompatibleLLMClient.from_env


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repository_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    protocol = load_protocol(Path(args.protocol), repository_root=root)
    if args.command == "prepare":
        return prepare_run(protocol, repository_root=root, run_dir=run_dir)
    if args.command == "verify":
        report = verify_run(protocol, repository_root=root, run_dir=run_dir)
        if args.publish_incomplete_dir:
            return publish_incomplete_audit(
                protocol,
                repository_root=root,
                run_dir=run_dir,
                publish_dir=Path(args.publish_incomplete_dir).resolve(),
            )
        return report
    if args.command == "score":
        result = score_run(
            protocol,
            repository_root=root,
            run_dir=run_dir,
            publish_dir=(
                Path(args.publish_dir).resolve() if args.publish_dir else None
            ),
        )
        return result
    client, binding, client_factory = _runtime(str(protocol["contract"]))
    if args.command == "judge":
        return run_judge_round(
            protocol,
            repository_root=root,
            run_dir=run_dir,
            round_id=args.round,
            client=client,
            runtime_binding=binding,
            max_batches=args.max_batches,
            client_factory=client_factory,
        )
    return run_adjudication(
        protocol,
        repository_root=root,
        run_dir=run_dir,
        client=client,
        runtime_binding=binding,
        max_batches=args.max_batches,
        client_factory=client_factory,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except LLMRelevanceJudgingError:
        report = usage_error_report(contract=HARDENED_CONTRACT_VERSION)
        _emit(report)
        return EXIT_USAGE_ERROR
    if getattr(args, "max_batches", None) is not None and args.max_batches < 1:
        report = usage_error_report(
            "max_batches_must_be_positive",
            contract=HARDENED_CONTRACT_VERSION,
        )
        _emit(report)
        return EXIT_USAGE_ERROR
    snapshot_before = tree_signature(DEFAULT_SNAPSHOT_ROOT)
    try:
        report = _execute(args)
    except LLMRelevanceJudgingIncomplete as exc:
        report = incomplete_report(
            args.command,
            str(exc),
            contract=HARDENED_CONTRACT_VERSION,
        )
    except LLMRelevanceJudgingError as exc:
        report = violation_report(
            args.command,
            str(exc),
            contract=HARDENED_CONTRACT_VERSION,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        report = usage_error_report(
            "input_or_runtime_error",
            contract=HARDENED_CONTRACT_VERSION,
        )
    snapshot_after = tree_signature(DEFAULT_SNAPSHOT_ROOT)
    if snapshot_before != snapshot_after:
        report = violation_report(
            args.command,
            "snapshot_tree_modified",
            contract=HARDENED_CONTRACT_VERSION,
        )
    _emit(report)
    return int(report.get("exit_code", EXIT_INTEGRITY_VIOLATION))


if __name__ == "__main__":
    raise SystemExit(main())
