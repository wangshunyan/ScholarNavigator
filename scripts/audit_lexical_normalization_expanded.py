#!/usr/bin/env python3
"""Run the 162-case frozen-Record lexical normalization pairing audit."""

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

from scholar_agent.evaluation.lexical_normalization_expanded import (  # noqa: E402
    run_expanded_lexical_audit,
    write_expanded_lexical_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit lexical normalization on the frozen 162-case Record prefix."
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/lexical_normalization_record160_manifest.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cases, candidates, aggregate = run_expanded_lexical_audit(args.manifest)
    write_expanded_lexical_audit(
        args.output,
        cases,
        candidates,
        aggregate,
        args.manifest,
    )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "main_case_count": aggregate["closure"]["status_counts"].get(
                    "included_main_analysis", 0
                ),
                "candidate_pairing_mismatch_count": aggregate["closure"][
                    "candidate_pairing_mismatch_count"
                ],
                "network_request_count": 0,
                "llm_request_count": 0,
                "snapshot_write_count": 0,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
