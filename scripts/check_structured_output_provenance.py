#!/usr/bin/env python3
"""Run the frozen structured-output provenance gate offline."""

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

from scholar_agent.evaluation.structured_output_provenance_gate import (  # noqa: E402
    run_structured_output_provenance_gate,
    write_structured_output_provenance_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate frozen structured search output provenance offline."
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/structured_output_provenance_gate_manifest.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cases, provenance, aggregate = run_structured_output_provenance_gate(
        args.manifest
    )
    write_structured_output_provenance_gate(
        args.output, cases, provenance, aggregate
    )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "gate_passed": aggregate["gate_passed"],
                "terminal_status_counts": aggregate["terminal_status_counts"],
                "network_request_count": 0,
                "llm_request_count": 0,
                "snapshot_write_count": 0,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if aggregate["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
