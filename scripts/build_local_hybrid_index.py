#!/usr/bin/env python3
"""Build the local BGE vector index used by the local_hybrid connector."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scholar_agent.connectors import (  # noqa: E402
    LocalBM25Config,
    LocalBM25FieldConfig,
    LocalHybridConfig,
    build_local_hybrid_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local BM25+semantic vector index."
    )
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=80)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--recall-sample-size", type=int, default=100)
    parser.add_argument("--recall-k", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        metadata = build_local_hybrid_index(
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
                hnsw_m=args.hnsw_m,
                hnsw_ef_construction=args.hnsw_ef_construction,
                hnsw_ef_search=args.hnsw_ef_search,
                recall_sample_size=args.recall_sample_size,
                recall_k=args.recall_k,
            ),
            resume=args.resume,
        )
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(asdict(metadata), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
