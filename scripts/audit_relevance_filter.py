#!/usr/bin/env python3
"""Run the pure-offline relevance-filter false-negative audit."""

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

from scholar_agent.evaluation.relevance_filter_audit import (  # noqa: E402
    run_relevance_filter_audit,
    write_relevance_filter_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit frozen relevance-filter false negatives without I/O calls."
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/relevance_filter_audit_manifest.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cases, candidates, filtered, aggregate = run_relevance_filter_audit(
        args.manifest
    )
    write_relevance_filter_audit(
        args.output,
        cases,
        candidates,
        filtered,
        aggregate,
    )
    print(
        json.dumps(
            {
                "dataset_count": len(aggregate["datasets"]),
                "filtered_chain_count": aggregate["filtered_chain_count"],
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
