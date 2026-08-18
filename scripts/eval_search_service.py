#!/usr/bin/env python3
"""Run offline SearchService evaluation with local fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.evaluation.fixture_loader import (  # noqa: E402
    build_fixture_reference_fetcher,
    build_fixture_retriever,
    load_evaluation_fixtures,
)
from scholar_agent.evaluation.offline_evaluator import (  # noqa: E402
    DEFAULT_GROUPS,
    evaluate_search_service_offline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline SearchService evaluation from local fixtures."
    )
    parser.add_argument(
        "--fixtures-dir",
        default="datasets/eval_fixtures/sample",
        help="Fixture directory containing search_cases.jsonl, retrieval_outputs.json, reference_outputs.json.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/eval_runs",
        help="Root directory for evaluation outputs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        choices=list(DEFAULT_GROUPS),
        help="Feature groups to evaluate.",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    fixtures = load_evaluation_fixtures(args.fixtures_dir)
    result = evaluate_search_service_offline(
        fixtures.eval_queries,
        retriever=build_fixture_retriever(fixtures.retrieval_outputs),
        reference_fetcher=build_fixture_reference_fetcher(fixtures.reference_outputs),
        max_workers=args.max_workers,
        groups=[group for group in args.groups],  # type: ignore[list-item]
    )

    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
