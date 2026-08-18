#!/usr/bin/env python3
"""Run the frozen-snapshot current_rules source ablation audit."""

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

from scholar_agent.evaluation.current_rules_source_audit import (  # noqa: E402
    AuditDataset,
    run_source_audit,
    write_source_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pure-Replay source contribution audit for current_rules."
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("NAME", "RUN_DIR", "SNAPSHOT_DIR"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    datasets = [
        AuditDataset(
            name=values[0],
            run_dir=Path(values[1]).expanduser().resolve(),
            snapshot_dir=Path(values[2]).expanduser().resolve(),
        )
        for values in args.input
    ]
    rows, aggregate = run_source_audit(datasets)
    write_source_audit(args.output, rows, aggregate)
    print(
        json.dumps(
            {
                "case_count": len(rows),
                "network_request_count": aggregate["network_request_count"],
                "llm_request_count": aggregate["llm_request_count"],
                "snapshot_write_count": aggregate["snapshot_write_count"],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
