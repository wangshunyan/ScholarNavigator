#!/usr/bin/env python3
"""Run and validate the five gold-blind contest demo queries offline.

This is a small presentation smoke, not a quality benchmark.  It delegates
to the same batch CLI used by the documented workflow and only accepts a
complete, successful, zero-network/zero-LLM manifest with visible results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_NOT_READY = 3


def validate_demo_manifest(
    manifest: dict[str, Any], *, expected_case_count: int = 5, minimum_visible_results: int = 1
) -> list[str]:
    reasons: list[str] = []
    if manifest.get("schema_version") != "search-batch-manifest-v1":
        reasons.append("manifest_schema_invalid")
    if manifest.get("case_count") != expected_case_count:
        reasons.append("demo_case_count_mismatch")
    if manifest.get("executed_case_count") != expected_case_count:
        reasons.append("demo_cases_incomplete")
    if manifest.get("partial") is not False:
        reasons.append("demo_manifest_partial")
    if manifest.get("succeeded_count") != expected_case_count:
        reasons.append("demo_case_failed")
    if manifest.get("failed_count") != 0:
        reasons.append("demo_failure_count_nonzero")
    if manifest.get("network_requests") != 0:
        reasons.append("demo_network_requests_nonzero")
    if manifest.get("llm_calls") != 0:
        reasons.append("demo_llm_calls_nonzero")
    if manifest.get("gold_or_qrels_loaded") is not False:
        reasons.append("demo_gold_or_qrels_loaded")
    summaries = manifest.get("case_summaries")
    if not isinstance(summaries, list) or len(summaries) != expected_case_count:
        reasons.append("demo_case_summaries_missing")
    else:
        for row in summaries:
            if not isinstance(row, dict):
                reasons.append("demo_case_summary_invalid")
                continue
            if row.get("status") != "succeeded":
                reasons.append(f"demo_case_not_succeeded:{row.get('case_id', 'unknown')}")
            if int(row.get("visible_result_count", 0) or 0) < minimum_visible_results:
                reasons.append(f"demo_case_no_visible_results:{row.get('case_id', 'unknown')}")
    return sorted(set(reasons))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "docs" / "contest" / "demo-queries.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "demo-batch" / "demo-smoke-results.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "demo-batch" / "demo-smoke-manifest.json",
    )
    parser.add_argument("--sources", default="local_bm25")
    parser.add_argument("--run-profile", default="balanced")
    parser.add_argument("--current-year", type=int, default=2026)
    parser.add_argument("--minimum-visible-results", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.minimum_visible_results < 1:
        print("minimum_visible_results_must_be_positive", file=sys.stderr)
        return EXIT_VIOLATION
    if not args.input.is_file():
        print(f"demo_input_not_found:{args.input}", file=sys.stderr)
        return EXIT_NOT_READY
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_search_batch.py"),
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--manifest",
        str(args.manifest),
        "--sources",
        args.sources,
        "--run-profile",
        args.run_profile,
        "--current-year",
        str(args.current_year),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        return EXIT_NOT_READY
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"demo_manifest_unreadable:{exc}", file=sys.stderr)
        return EXIT_VIOLATION
    reasons = validate_demo_manifest(
        manifest, minimum_visible_results=args.minimum_visible_results
    )
    report = {
        "schema_version": "demo-reproducibility-check-v1",
        "status": "ready" if not reasons else "not_ready",
        "reasons": reasons,
        "manifest": str(args.manifest),
        "source_preferences": args.sources.split(","),
        "minimum_visible_results": args.minimum_visible_results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return EXIT_READY if not reasons else EXIT_NOT_READY


if __name__ == "__main__":
    raise SystemExit(main())

