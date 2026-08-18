#!/usr/bin/env python3
"""Run the deterministic cross-strategy candidate-union audit."""

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

from scholar_agent.evaluation.cross_strategy_union_audit import (  # noqa: E402
    StrategyArtifact,
    UnionAuditDataset,
    run_cross_strategy_union_audit,
    write_cross_strategy_union_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pure-Replay candidate-union upper-bound audit."
    )
    parser.add_argument(
        "--manifest",
        default="benchmark/cross_strategy_union_audit_manifest.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_root = manifest_path.parent.parent
    datasets = []
    for item in payload["datasets"]:
        datasets.append(
            UnionAuditDataset(
                name=item["name"],
                excluded_strategies=item.get("excluded_strategies") or {},
                strategies=[
                    StrategyArtifact(
                        strategy=strategy["strategy"],
                        run_dir=project_root / strategy["run_dir"],
                        snapshot_dir=project_root / strategy["snapshot_dir"],
                    )
                    for strategy in item["strategies"]
                ],
            )
        )
    rows, aggregate = run_cross_strategy_union_audit(datasets)
    write_cross_strategy_union_audit(args.output, rows, aggregate)
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
