#!/usr/bin/env python3
"""Run the pure-offline local BM25 candidate conversion audit."""

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

from scholar_agent.evaluation.local_bm25_conversion_audit import (  # noqa: E402
    run_conversion_audit,
    write_conversion_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pure-offline local_bm25 ranking/filter conversion audit."
    )
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--crosswalk", default="benchmark/beir_scifact_s2_crosswalk.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cases, gold, aggregate = run_conversion_audit(
        baseline_run_dir=args.baseline_run_dir,
        run_dir=args.run_dir,
        snapshot_dir=args.snapshot_dir,
        corpus_path=args.corpus,
        crosswalk_path=args.crosswalk,
    )
    write_conversion_audit(args.output, cases, gold, aggregate)
    print(
        json.dumps(
            {
                "case_count": aggregate["case_count"],
                "candidate_gold_relation_count": aggregate[
                    "candidate_gold_relation_count"
                ],
                "network_request_count": aggregate["execution"][
                    "network_request_count"
                ],
                "llm_request_count": aggregate["execution"]["llm_request_count"],
                "snapshot_write_count": aggregate["execution"][
                    "snapshot_write_count"
                ],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
