"""Deterministic offline BM25 upper-bound audit for BEIR SciFact."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from importlib.metadata import version
from inspect import signature
from pathlib import Path
from typing import Any, Iterable, Sequence

from rank_bm25 import BM25Okapi

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.evaluation.datasets.beir_scifact import (
    load_beir_scifact_enriched,
)
from scholar_agent.evaluation.metrics import (
    gold_crosswalk_status,
    matched_paper_ids,
)


AUDIT_VERSION = "scifact-bm25-v1"
DEPTHS = (20, 50, 100, 200)
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(value: str | None) -> list[str]:
    """Case-fold Unicode words without stopwords, stemming, or expansion."""

    return TOKEN_PATTERN.findall(str(value or "").casefold())


def document_text(title: Any, abstract: Any) -> str:
    return " ".join(
        value.strip()
        for value in (str(title or ""), str(abstract or ""))
        if value.strip()
    )


def load_corpus(path: str | Path) -> list[dict[str, str]]:
    """Load the official corpus and reject duplicate document IDs."""

    source = Path(path).expanduser().resolve()
    if source.is_file() and source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            try:
                raw = archive.read("scifact/corpus.jsonl").decode("utf-8")
            except KeyError as exc:
                raise ValueError("SciFact archive missing corpus.jsonl") from exc
    else:
        root = source / "scifact" if (source / "scifact").is_dir() else source
        corpus_path = root / "corpus.jsonl"
        if not corpus_path.is_file():
            raise ValueError("SciFact dataset missing corpus.jsonl")
        raw = corpus_path.read_text(encoding="utf-8")

    documents: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid SciFact corpus row: {line_number}")
        document_id = str(payload.get("_id") or "").strip()
        if not document_id:
            raise ValueError(f"SciFact corpus row missing _id: {line_number}")
        if document_id in documents:
            raise ValueError(f"duplicate SciFact corpus ID: {document_id}")
        documents[document_id] = {
            "corpus_id": document_id,
            "title": str(payload.get("title") or ""),
            "abstract": str(payload.get("text") or ""),
        }
    return [documents[key] for key in sorted(documents)]


def load_sample_manifest(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    query_ids = payload.get("query_ids") if isinstance(payload, dict) else None
    if not isinstance(query_ids, list) or not query_ids:
        raise ValueError("invalid SciFact sample manifest")
    normalized = [str(value).strip() for value in query_ids]
    if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("invalid SciFact sample manifest query IDs")
    return normalized


class DeterministicBM25Index:
    """One immutable BM25Okapi index with deterministic tie-breaking."""

    def __init__(self, documents: Sequence[dict[str, str]]) -> None:
        self.documents = list(documents)
        tokenized = [
            tokenize(document_text(item.get("title"), item.get("abstract")))
            for item in self.documents
        ]
        self.engine = BM25Okapi(tokenized)

    def rank(self, query: str) -> list[tuple[str, float]]:
        scores = self.engine.get_scores(tokenize(query))
        rows = [
            (document["corpus_id"], float(scores[index]))
            for index, document in enumerate(self.documents)
        ]
        return sorted(rows, key=lambda item: (-item[1], item[0]))

    def config(self) -> dict[str, Any]:
        defaults = signature(BM25Okapi).parameters
        return {
            "library": "rank_bm25",
            "library_version": version("rank_bm25"),
            "implementation": "BM25Okapi",
            "parameters": {
                "k1": self.engine.k1,
                "b": self.engine.b,
                "epsilon": self.engine.epsilon,
            },
            "library_defaults": {
                name: defaults[name].default for name in ("k1", "b", "epsilon")
            },
            "tokenizer": "unicode_word_casefold_v1",
            "token_pattern": TOKEN_PATTERN.pattern,
            "stopwords": False,
            "stemming": False,
            "query_expansion": False,
            "document_fields": ["title", "abstract"],
            "tie_break": "score_desc_then_corpus_id_ascending",
        }


def load_external_candidate_hits(
    run_dir: str | Path,
    queries: Sequence[EvalQuery],
) -> dict[tuple[str, int], bool]:
    """Match frozen external candidates to enriched gold by stable IDs only."""

    rows = _read_jsonl(Path(run_dir) / "results.jsonl")
    by_case = {str(row.get("case_id") or ""): row for row in rows}
    if len(by_case) != len(rows):
        raise ValueError("duplicate case ID in external Replay results")
    expected = {query.query_id for query in queries}
    if set(by_case) != expected:
        raise ValueError("external Replay query set does not match SciFact manifest")

    result: dict[tuple[str, int], bool] = {}
    for query in queries:
        row = by_case[query.query_id]
        if row.get("status") != "succeeded":
            raise ValueError("external Replay contains a non-success query terminal")
        diagnostics = row.get("stage_diagnostics")
        snapshots = diagnostics.get("snapshots") if isinstance(diagnostics, dict) else None
        if not isinstance(snapshots, list):
            raise ValueError("external Replay missing stage diagnostics")
        initial = next(
            (
                item
                for item in snapshots
                if isinstance(item, dict) and item.get("stage") == "initial_retrieval"
            ),
            None,
        )
        if not isinstance(initial, dict) or initial.get("status") != "completed":
            raise ValueError("external Replay missing completed initial retrieval")
        candidates = initial.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("external Replay initial candidates are invalid")
        for index, gold in enumerate(query.gold_papers):
            result[(query.query_id, index)] = bool(
                matched_paper_ids(candidates, [gold])
            )
    return result


def run_audit(
    *,
    dataset_path: str | Path,
    sample_manifest_path: str | Path,
    crosswalk_path: str | Path,
    external_run_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    documents = load_corpus(dataset_path)
    queries = load_beir_scifact_enriched(
        dataset_path,
        crosswalk_path=crosswalk_path,
    )
    expected_ids = load_sample_manifest(sample_manifest_path)
    if [query.query_id for query in queries] != expected_ids:
        raise ValueError("SciFact adapter order does not match the fixed manifest")
    corpus_ids = {item["corpus_id"] for item in documents}
    for query in queries:
        for gold in query.gold_papers:
            if str(gold.s2orc_corpus_id) not in corpus_ids:
                raise ValueError("SciFact gold Corpus ID is absent from the corpus")

    external_hits = load_external_candidate_hits(external_run_dir, queries)
    index = DeterministicBM25Index(documents)
    per_query = [
        _audit_query(query, index.rank(query.query), external_hits)
        for query in queries
    ]
    config = {
        "audit_version": AUDIT_VERSION,
        "dataset": "beir_scifact",
        "split": "test",
        "query_input": "original_query_text_only",
        "gold_usage": "offline_evaluator_only",
        "prefix_evaluation": "single_full_corpus_ranking",
        "depths": list(DEPTHS),
        "case_count": len(queries),
        "corpus_count": len(documents),
        "sample_query_ids": expected_ids,
        "inputs": {
            "dataset_sha256": file_sha256(dataset_path),
            "sample_manifest_sha256": file_sha256(sample_manifest_path),
            "crosswalk_sha256": file_sha256(crosswalk_path),
            "external_results_sha256": file_sha256(
                Path(external_run_dir) / "results.jsonl"
            ),
        },
        "bm25": index.config(),
    }
    return config, per_query, aggregate_rows(per_query)


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evaluable_rows = [row for row in rows if row["evaluable_gold_count"] > 0]
    evaluable_gold_count = sum(row["evaluable_gold_count"] for row in rows)
    total_gold_count = sum(row["gold_count"] for row in rows)
    depths: dict[str, dict[str, float | int]] = {}
    for depth in DEPTHS:
        matched = sum(row["depths"][str(depth)]["matched_gold_count"] for row in rows)
        macro_values = [
            row["depths"][str(depth)]["recall"] for row in evaluable_rows
        ]
        depths[str(depth)] = {
            "matched_gold_count": matched,
            "micro_recall": matched / evaluable_gold_count
            if evaluable_gold_count
            else 0.0,
            "macro_recall": sum(macro_values) / len(macro_values)
            if macro_values
            else 0.0,
        }
    reciprocal_ranks = [row["reciprocal_rank"] for row in evaluable_rows]
    classifications = Counter(
        gold["classification"] for row in rows for gold in row["gold"]
    )
    classification_names = (
        "bm25_top_200_only",
        "external_only",
        "both_hit",
        "neither_hit",
        "identity_unresolvable",
    )
    external_count = sum(
        gold["external_candidate_hit"]
        for row in rows
        for gold in row["gold"]
        if gold["evaluable"]
    )
    first_hit_ranks = [
        row["first_hit_rank"]
        for row in evaluable_rows
        if row["first_hit_rank"] is not None
    ]
    return {
        "case_count": len(rows),
        "evaluable_query_count": len(evaluable_rows),
        "gold_count": total_gold_count,
        "evaluable_gold_count": evaluable_gold_count,
        "identity_unresolvable_gold_count": total_gold_count
        - evaluable_gold_count,
        "depths": depths,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0,
        "query_first_hit_distribution": _rank_distribution(first_hit_ranks),
        "gold_first_hit_distribution": _rank_distribution(
            [
                gold["bm25_rank"]
                for row in rows
                for gold in row["gold"]
                if gold["evaluable"] and gold["bm25_rank"] is not None
            ],
            include_miss=evaluable_gold_count,
        ),
        "external_candidate": {
            "matched_gold_count": external_count,
            "micro_recall": external_count / evaluable_gold_count
            if evaluable_gold_count
            else 0.0,
            "macro_recall": sum(
                row["external_candidate_recall"] for row in evaluable_rows
            )
            / len(evaluable_rows)
            if evaluable_rows
            else 0.0,
        },
        "classification_counts": {
            name: classifications[name] for name in classification_names
        },
    }


def _audit_query(
    query: EvalQuery,
    ranking: Sequence[tuple[str, float]],
    external_hits: dict[tuple[str, int], bool],
) -> dict[str, Any]:
    rank_by_id = {
        corpus_id: rank
        for rank, (corpus_id, _score) in enumerate(ranking, start=1)
    }
    gold_rows: list[dict[str, Any]] = []
    for index, gold in enumerate(query.gold_papers):
        status = gold_crosswalk_status(gold)
        evaluable = status == "success"
        rank = rank_by_id.get(str(gold.s2orc_corpus_id)) if evaluable else None
        external_hit = external_hits[(query.query_id, index)] if evaluable else False
        gold_rows.append(
            {
                "gold_index": index,
                "s2orc_corpus_id": str(gold.s2orc_corpus_id),
                "identity_status": status,
                "evaluable": evaluable,
                "bm25_rank": rank,
                "first_hit_depth": _first_depth(rank),
                "external_candidate_hit": external_hit,
                "classification": _classification(
                    evaluable=evaluable,
                    bm25_hit=rank is not None and rank <= max(DEPTHS),
                    external_hit=external_hit,
                ),
            }
        )
    evaluable_gold = [row for row in gold_rows if row["evaluable"]]
    depths = {
        str(depth): _depth_metrics(evaluable_gold, depth) for depth in DEPTHS
    }
    ranks = [row["bm25_rank"] for row in evaluable_gold if row["bm25_rank"]]
    first_rank = min(ranks) if ranks else None
    external_count = sum(row["external_candidate_hit"] for row in evaluable_gold)
    return {
        "query_id": query.query_id,
        "query": query.query,
        "gold_count": len(gold_rows),
        "evaluable_gold_count": len(evaluable_gold),
        "identity_unresolvable_gold_count": len(gold_rows) - len(evaluable_gold),
        "depths": depths,
        "first_hit_rank": first_rank,
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "external_candidate_matched_gold_count": external_count,
        "external_candidate_recall": external_count / len(evaluable_gold)
        if evaluable_gold
        else 0.0,
        "gold": gold_rows,
        "top_200": [
            {"corpus_id": corpus_id, "score": score}
            for corpus_id, score in ranking[: max(DEPTHS)]
        ],
    }


def _depth_metrics(gold_rows: Sequence[dict[str, Any]], depth: int) -> dict[str, Any]:
    matched = sum(
        row["bm25_rank"] is not None and row["bm25_rank"] <= depth
        for row in gold_rows
    )
    return {
        "matched_gold_count": matched,
        "recall": matched / len(gold_rows) if gold_rows else 0.0,
    }


def _first_depth(rank: int | None) -> int | None:
    if rank is None:
        return None
    return next((depth for depth in DEPTHS if rank <= depth), None)


def _classification(
    *,
    evaluable: bool,
    bm25_hit: bool,
    external_hit: bool,
) -> str:
    if not evaluable:
        return "identity_unresolvable"
    if bm25_hit and external_hit:
        return "both_hit"
    if bm25_hit:
        return "bm25_top_200_only"
    if external_hit:
        return "external_only"
    return "neither_hit"


def _rank_distribution(
    ranks: Iterable[int],
    *,
    include_miss: int | None = None,
) -> dict[str, int]:
    values = list(ranks)
    result = {
        "1_20": sum(rank <= 20 for rank in values),
        "21_50": sum(20 < rank <= 50 for rank in values),
        "51_100": sum(50 < rank <= 100 for rank in values),
        "101_200": sum(100 < rank <= 200 for rank in values),
        "beyond_200": sum(rank > 200 for rank in values),
    }
    if include_miss is not None:
        result["not_in_top_200"] = include_miss - sum(
            rank <= 200 for rank in values
        )
    return result


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    if source.is_dir():
        files = sorted(item for item in source.rglob("*") if item.is_file())
        for item in files:
            relative = item.relative_to(source).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(item.read_bytes())
        return digest.hexdigest()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifacts(
    output_dir: str | Path,
    config: dict[str, Any],
    per_query: Sequence[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "config.json", config)
    _write_jsonl(root / "per_query.jsonl", per_query)
    _write_json(root / "aggregate.json", aggregate)
    hashes = {
        name: file_sha256(root / name)
        for name in ("config.json", "per_query.jsonl", "aggregate.json")
    }
    _write_json(root / "artifact_hashes.json", hashes)
    return hashes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid JSONL object at line {line_number}: {path}")
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
