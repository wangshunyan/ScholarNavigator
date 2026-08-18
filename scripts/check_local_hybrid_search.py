#!/usr/bin/env python3
"""Run a small local_hybrid retrieval sanity check against PaSa queries."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.connectors import (  # noqa: E402
    LocalBM25Config,
    LocalBM25FieldConfig,
    LocalHybridConfig,
    configure_local_hybrid,
    local_hybrid_metadata,
    search_local_hybrid_detailed,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify local_hybrid index loading, RRF fusion and abstracts."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmark/AutoScholarQuery_test.jsonl"),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--bm25-corpus",
        type=Path,
        default=Path("datasets/local_bm25/pasa_papers.jsonl"),
    )
    parser.add_argument(
        "--bm25-cache-dir",
        type=Path,
        default=Path("outputs/benchmark_cache/local_bm25"),
    )
    parser.add_argument(
        "--semantic-corpus",
        type=Path,
        default=Path("datasets/semantic/pasa_papers_with_abstracts.jsonl"),
    )
    parser.add_argument(
        "--semantic-index-dir",
        type=Path,
        default=Path("outputs/benchmark_cache/local_hybrid"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "datasets/semantic/models/models/"
            "AI-ModelScope--bge-small-en-v1.5/snapshots/master"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        configure_local_hybrid(
            LocalHybridConfig(
                bm25_config=LocalBM25Config(
                    corpus_path=args.bm25_corpus,
                    cache_dir=args.bm25_cache_dir,
                    fields=LocalBM25FieldConfig(
                        document_id="_id",
                        title="title",
                        abstract="abstract",
                        document_id_identity="arxiv_id",
                        arxiv_id="arxiv_id",
                    ),
                ),
                semantic_corpus_path=args.semantic_corpus,
                semantic_index_dir=args.semantic_index_dir,
                model_path=args.model,
            )
        )
        rows = _read_queries(args.dataset, args.offset, args.limit)
        report = {
            "metadata": asdict(local_hybrid_metadata()),
            "query_count": len(rows),
            "top_k": args.top_k,
            "queries": [
                _check_query(row, top_k=args.top_k)
                for row in rows
            ],
        }
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _read_queries(path: Path, offset: int, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number < offset or not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"invalid_query_row:{line_number + 1}")
            selected.append(payload)
            if len(selected) >= limit:
                break
    return selected


def _check_query(row: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    result = search_local_hybrid_detailed(question, top_k)
    returned_ids = [
        str(paper.identifiers.arxiv_id or "").strip()
        for paper in result.papers
    ]
    returned_id_set = {_normalize_arxiv_id(value) for value in returned_ids if value}
    gold_ids = [
        str(value)
        for value in row.get("answer_arxiv_id") or []
    ]
    gold_hit_ids = [
        value
        for value in gold_ids
        if _normalize_arxiv_id(value) in returned_id_set
    ]
    return {
        "qid": row.get("qid"),
        "query": question,
        "gold_count": len(gold_ids),
        "gold_hit_count": len(gold_hit_ids),
        "gold_hit_ids": gold_hit_ids,
        "latency_seconds": result.latency_seconds,
        "warnings": result.warnings,
        "abstract_result_count": sum(bool(paper.abstract) for paper in result.papers),
        "top_results": [
            {
                "rank": index,
                "arxiv_id": paper.identifiers.arxiv_id,
                "title": paper.title,
                "sources": paper.sources,
                "has_abstract": bool(paper.abstract),
            }
            for index, paper in enumerate(result.papers[:5], start=1)
        ],
    }


def _normalize_arxiv_id(value: str) -> str:
    return value.strip().casefold().removeprefix("arxiv:").split("v", 1)[0]


if __name__ == "__main__":
    raise SystemExit(main())
