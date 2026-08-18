#!/usr/bin/env python3
"""只读检查已注册 Benchmark 的数据完整性。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.evaluation.datasets import (  # noqa: E402
    inspect_dataset,
    supported_datasets,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查本地 Benchmark 数据完整性。")
    parser.add_argument("--dataset", required=True, choices=supported_datasets())
    parser.add_argument("--path", default=None, help="可选的数据文件覆盖路径。")
    args = parser.parse_args(argv)
    try:
        report = inspect_dataset(args.dataset, path=args.path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
