#!/usr/bin/env python3
"""Collect or replay the evaluator-only SciFact CorpusId crosswalk."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.core.env_loader import load_project_env  # noqa: E402
from scholar_agent.evaluation.datasets.beir_scifact import (  # noqa: E402
    load_beir_scifact,
)
from scholar_agent.evaluation.datasets.scifact_crosswalk import (  # noqa: E402
    SciFactCrosswalkStore,
    crosswalk_file_sha256,
    fetch_exact_corpus_id,
    record_missing_crosswalk,
    replay_crosswalk,
    write_crosswalk,
)


def _corpus_ids(dataset_path: Path) -> tuple[list[str], int]:
    queries = load_beir_scifact(dataset_path)
    values = [
        str(gold.s2orc_corpus_id)
        for query in queries
        for gold in query.gold_papers
        if gold.s2orc_corpus_id is not None
    ]
    return values, sum(len(query.gold_papers) for query in queries)


def _safe_snapshot_summary(snapshot: object) -> dict[str, object]:
    return {
        "status": getattr(snapshot, "status"),
        "error_type": getattr(snapshot, "error_type"),
        "http_status": getattr(snapshot, "http_status"),
        "request_count": getattr(snapshot, "request_count"),
        "retry_count": getattr(snapshot, "retry_count"),
        "response_fields": [
            "paperId",
            "corpusId",
            "externalIds",
        ],
        "returned_external_id_fields": sorted(
            getattr(snapshot, "external_ids").keys()
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建 SciFact evaluator-only Semantic Scholar CorpusId crosswalk。"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("preflight", "record-missing", "replay"),
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output", default="benchmark/beir_scifact_s2_crosswalk.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    store = SciFactCrosswalkStore(args.snapshot_dir)
    corpus_ids, gold_relation_count = _corpus_ids(dataset_path)
    unique_count = len(set(corpus_ids))

    if args.mode == "preflight":
        load_project_env(REPO_ROOT)
        snapshot = fetch_exact_corpus_id(corpus_ids[0])
        print(json.dumps(_safe_snapshot_summary(snapshot), sort_keys=True))
        return 0 if snapshot.status == "success" else 2

    if args.mode == "record-missing":
        load_project_env(REPO_ROOT)
        counts = record_missing_crosswalk(corpus_ids, store)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "gold_relation_count": gold_relation_count,
                    "unique_request_count": unique_count,
                    **counts,
                },
                sort_keys=True,
            )
        )
        return 0

    artifact = replay_crosswalk(corpus_ids, store)
    output = Path(args.output).expanduser().resolve()
    write_crosswalk(output, artifact)
    statuses = Counter(entry.status for entry in artifact.entries)
    fields = Counter(
        field for entry in artifact.entries for field in entry.external_id_fields
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "gold_relation_count": gold_relation_count,
                "unique_request_count": unique_count,
                "statuses": dict(sorted(statuses.items())),
                "external_id_fields": dict(sorted(fields.items())),
                "artifact_sha256": crosswalk_file_sha256(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
