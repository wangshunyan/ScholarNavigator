#!/usr/bin/env python3
"""Run the frozen-snapshot causal audit for constrained LLM query rewriting."""

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

from scholar_agent.evaluation.llm_rewrite_causal_audit import (  # noqa: E402
    AuditPair,
    run_causal_audit,
    write_causal_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pure-Replay causal audit for llm_constrained_rewrite."
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=5,
        metavar=("NAME", "BASELINE_RUN", "REWRITE_RUN", "BASELINE_SNAPSHOT", "REWRITE_SNAPSHOT"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    pairs = [
        AuditPair(
            name=values[0],
            baseline_run=Path(values[1]).expanduser().resolve(),
            rewrite_run=Path(values[2]).expanduser().resolve(),
            baseline_snapshot=Path(values[3]).expanduser().resolve(),
            rewrite_snapshot=Path(values[4]).expanduser().resolve(),
        )
        for values in args.pair
    ]
    case_rows, aggregate = run_causal_audit(pairs)
    write_causal_audit(args.output, case_rows, aggregate)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "case_count": len(case_rows),
                "network_request_count": aggregate["network_request_count"],
                "snapshot_write_count": aggregate["snapshot_write_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
