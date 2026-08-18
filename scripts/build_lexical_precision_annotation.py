#!/usr/bin/env python3
"""Generate or score the lexical-normalization blind annotation package."""

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

from scholar_agent.evaluation.precision_annotation import (  # noqa: E402
    evaluate_annotations,
    generate_precision_annotation_package,
    write_precision_annotation_package,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or score the blinded lexical-normalization annotation package."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--manifest",
        default="benchmark/lexical_normalization_precision_annotation_manifest.json",
    )
    generate.add_argument("--output", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--mapping", required=True)
    score.add_argument("--annotator-one", required=True)
    score.add_argument("--annotator-two", required=True)
    score.add_argument("--adjudication", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        package = generate_precision_annotation_package(args.manifest)
        write_precision_annotation_package(args.output, package)
        print(
            json.dumps(
                {
                    "sample_count": package["summary"]["sample_count"],
                    "annotation_status": package["summary"]["annotation_status"],
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
    mapping = _read_json(args.mapping)
    first = _read_jsonl(args.annotator_one)
    second = _read_jsonl(args.annotator_two)
    adjudication = _read_jsonl(args.adjudication)
    result = evaluate_annotations(mapping, first, second, adjudication)
    Path(args.output).expanduser().resolve().write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "annotation_status": result["annotation_status"],
                "sample_count": result["sample_count"],
                "output": str(Path(args.output).expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _read_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _read_jsonl(path: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
