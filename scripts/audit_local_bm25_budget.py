#!/usr/bin/env python3
"""Run the frozen, pure-offline SciFact local-BM25 budget audit."""

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

from scholar_agent.evaluation.local_bm25_budget_audit import (  # noqa: E402
    run_budget_audit,
    write_budget_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="审计 SciFact local_bm25 查询配额与候选截断（纯离线）。"
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/beir_scifact_local_bm25_budget_audit_manifest.json",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases, gold, aggregate = run_budget_audit(args.manifest)
    hashes = write_budget_audit(args.output, cases, gold, aggregate)
    curves = aggregate["depth_curves"]
    print(
        json.dumps(
            {
                "case_count": aggregate["case_count"],
                "gold_chain_count": aggregate["gold_chain_count"],
                "matched_at_200": {
                    scope: values["200"]["matched_gold_relation_count"]
                    for scope, values in curves.items()
                },
                "network_request_count": aggregate["execution"][
                    "network_request_count"
                ],
                "llm_request_count": aggregate["execution"]["llm_request_count"],
                "snapshot_write_count": aggregate["execution"][
                    "snapshot_write_count"
                ],
                "artifact_hashes": hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
