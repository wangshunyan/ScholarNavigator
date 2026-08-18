#!/usr/bin/env python3
"""Generate or score the exhaustive Record-160 Top-20 change package."""

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

from scholar_agent.evaluation.full_swap_precision_annotation import (  # noqa: E402
    evaluate_full_swap_annotations,
    generate_full_swap_package,
    write_full_swap_package,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or score all lexical-normalization Top-20 changes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument(
        "--manifest",
        default="benchmark/lexical_normalization_record160_precision_manifest.json",
    )
    generate.add_argument("--output", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--mapping", required=True)
    score.add_argument("--annotator-one", required=True)
    score.add_argument("--annotator-two", required=True)
    score.add_argument("--adjudication", required=True)
    score.add_argument("--prior-resolved-labels")
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        package = generate_full_swap_package(args.manifest)
        write_full_swap_package(args.output, package)
        print(
            json.dumps(
                {
                    "annotation_status": package["summary"]["annotation_status"],
                    "public_new_sample_count": package["summary"][
                        "public_new_sample_count"
                    ],
                    "top20_change_relation_count": package["summary"][
                        "top20_change_relation_count"
                    ],
                    "prior_package_overlap_sample_count": package["summary"][
                        "prior_package_overlap_sample_count"
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

    prior = None
    if args.prior_resolved_labels:
        prior = {
            str(row["sample_id"]): str(row["label"])
            for row in _read_jsonl(args.prior_resolved_labels)
        }
    result = evaluate_full_swap_annotations(
        _read_json(args.mapping),
        _read_jsonl(args.annotator_one),
        _read_jsonl(args.annotator_two),
        _read_jsonl(args.adjudication),
        prior_resolved_labels=prior,
    )
    output = Path(args.output).expanduser().resolve()
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "annotation_status": result["annotation_status"],
                "sample_count": result["sample_count"],
                "output": str(output),
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
