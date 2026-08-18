#!/usr/bin/env python3
"""Run the deterministic offline SciFact BM25 upper-bound audit."""

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

from scholar_agent.evaluation.scifact_bm25_audit import (  # noqa: E402
    run_audit,
    write_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 SciFact 官方语料的纯离线 BM25 召回上界审计。"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--crosswalk", required=True)
    parser.add_argument("--external-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config, per_query, aggregate = run_audit(
        dataset_path=args.dataset_path,
        sample_manifest_path=args.sample_manifest,
        crosswalk_path=args.crosswalk,
        external_run_dir=args.external_run_dir,
    )
    hashes = write_artifacts(args.output_dir, config, per_query, aggregate)
    print(
        json.dumps(
            {
                "case_count": aggregate["case_count"],
                "evaluable_gold_count": aggregate["evaluable_gold_count"],
                "depths": aggregate["depths"],
                "mrr": aggregate["mrr"],
                "classification_counts": aggregate["classification_counts"],
                "artifact_hashes": hashes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
