#!/usr/bin/env python3
"""检查 Benchmark 响应快照的完整性、成本与四组覆盖率。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.evaluation.snapshots import SnapshotStore  # noqa: E402


def inspect_snapshot(snapshot_dir: Path | str) -> dict[str, object]:
    return SnapshotStore(snapshot_dir).inspect()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 Benchmark Record/Replay 快照。")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = inspect_snapshot(args.snapshot_dir)
    except Exception as exc:  # noqa: BLE001 - CLI 返回稳定错误
        print(f"snapshot_inspection_failed:{exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    invalid = bool(
        report.get("invalid_entries")
        or report.get("hash_mismatch_entries")
        or report.get("duplicate_keys")
    )
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
