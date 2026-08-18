#!/usr/bin/env python3
"""Run the frozen-candidate lexical-normalization benchmark offline."""

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

from scholar_agent.evaluation.lexical_normalization_benchmark import (  # noqa: E402
    run_lexical_normalization_benchmark,
    write_lexical_normalization_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current_rules with default-off lexical normalization "
            "using frozen candidates only."
        )
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/lexical_normalization_v1_manifest.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cases, candidates, aggregate = run_lexical_normalization_benchmark(
        args.manifest
    )
    write_lexical_normalization_benchmark(
        args.output,
        cases,
        candidates,
        aggregate,
    )
    print(
        json.dumps(
            {
                "dataset_count": len(aggregate["datasets"]),
                "network_request_count": aggregate["execution"][
                    "network_request_count"
                ],
                "llm_request_count": aggregate["execution"][
                    "llm_request_count"
                ],
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
