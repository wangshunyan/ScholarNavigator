#!/usr/bin/env python3
"""Run the pure-offline paired local-BM25 original-query deepening benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.evaluation.local_bm25_original_deepening import (  # noqa: E402
    run_original_deepening_benchmark,
    write_original_deepening_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="纯离线评测 local_bm25 原始查询优先加深策略。"
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/beir_scifact_local_bm25_original_deepening_manifest.json",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases, candidates, gold, aggregate = run_original_deepening_benchmark(
        args.manifest
    )
    hashes = write_original_deepening_artifacts(
        args.output, cases, candidates, gold, aggregate
    )
    print(
        json.dumps(
            {
                "case_count": aggregate["case_count"],
                "evaluable_gold_relation_count": aggregate[
                    "evaluable_gold_relation_count"
                ],
                "variants": aggregate["variants"],
                "deep_gap_gold": aggregate["deep_gap_gold"],
                "execution": aggregate["execution"],
                "artifact_hashes": hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
