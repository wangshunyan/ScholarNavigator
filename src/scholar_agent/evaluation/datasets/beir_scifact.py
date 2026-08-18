"""Deterministic adapter for the official BEIR SciFact test split."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.core.identity import identity_evidence
from scholar_agent.evaluation.datasets.scifact_crosswalk import (
    CROSSWALK_CONNECTOR_VERSION,
    CROSSWALK_SOURCE,
    SciFactCrosswalkArtifact,
    SciFactCrosswalkEntry,
    load_crosswalk,
)


DATASET_NAME = "beir_scifact"
OFFICIAL_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
OFFICIAL_BEIR_COMMIT = "6ef8c90"
LICENSE = "Apache-2.0 (BEIR repository); SciFact data attribution follows BEIR release"
SAMPLE_SIZE = 50
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CROSSWALK_PATH = REPO_ROOT / "benchmark" / "beir_scifact_s2_crosswalk.json"


def load_beir_scifact(path: str | Path) -> list[EvalQuery]:
    """Load and deterministically sample the official SciFact test qrels.

    ``path`` is the extracted ``scifact`` directory or the official archive.
    The loader never sends gold data to search; gold records are attached only
    to the offline ``EvalQuery`` objects.
    """

    root, close = _open_dataset(Path(path))
    try:
        corpus = _read_jsonl(root / "corpus.jsonl", "corpus")
        queries = _read_jsonl(root / "queries.jsonl", "queries")
        qrels = _read_qrels(root / "qrels" / "test.tsv")
    finally:
        if close is not None:
            close()

    missing_queries = sorted({qid for qid, _, _ in qrels if qid not in queries})
    missing_docs = sorted({docid for _, docid, _ in qrels if docid not in corpus})
    if missing_queries or missing_docs:
        raise ValueError(
            "invalid beir_scifact qrels mapping: "
            f"missing_queries={len(missing_queries)} missing_docs={len(missing_docs)}"
        )

    by_query: dict[str, list[tuple[str, int]]] = {}
    for query_id, doc_id, grade in qrels:
        by_query.setdefault(query_id, []).append((doc_id, grade))

    selected_ids = sorted(
        by_query,
        key=lambda query_id: (
            hashlib.sha256(query_id.encode("utf-8")).hexdigest(),
            query_id,
        ),
    )[:SAMPLE_SIZE]
    result: list[EvalQuery] = []
    for query_id in selected_ids:
        gold: list[EvalGoldPaper] = []
        seen_docs: set[str] = set()
        for doc_id, grade in by_query[query_id]:
            if grade <= 0 or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            document = corpus[doc_id]
            title = _optional_string(document.get("title"))
            abstract = _optional_string(document.get("text"))
            gold.append(
                EvalGoldPaper(
                    title=title,
                    s2orc_corpus_id=doc_id,
                    relevance_grade=float(grade),
                    metadata={
                        "dataset": DATASET_NAME,
                        "s2orc_corpus_id": doc_id,
                        "identity_status": "corpus_id_resolved",
                        "abstract": abstract,
                        "qrel_grade": grade,
                    },
                )
            )
        result.append(
            EvalQuery(
                query_id=query_id,
                query=_required_string(queries[query_id], "text", query_id),
                gold_papers=gold,
                top_k_values=[5, 10, 20],
                run_profile="balanced",
                metadata={
                    "dataset": DATASET_NAME,
                    "split": "test",
                    "sampling": "sha256_query_id_ascending",
                    "sample_size": SAMPLE_SIZE,
                    "gold_source": "qrels/test.tsv",
                },
            )
        )
    return result


def load_beir_scifact_enriched(
    path: str | Path,
    *,
    crosswalk_path: str | Path = DEFAULT_CROSSWALK_PATH,
) -> list[EvalQuery]:
    """Load SciFact and enrich only evaluator gold with official stable IDs."""

    queries = load_beir_scifact(path)
    artifact = load_crosswalk(crosswalk_path)
    by_corpus_id = _crosswalk_index(artifact)
    enriched: list[EvalQuery] = []
    for query in queries:
        gold_papers: list[EvalGoldPaper] = []
        for gold in query.gold_papers:
            corpus_id = str(gold.s2orc_corpus_id or "").strip()
            entry = by_corpus_id.get(corpus_id)
            if entry is None:
                raise ValueError(
                    f"SciFact crosswalk missing expected Corpus ID: {corpus_id}"
                )
            gold_papers.append(_enrich_gold(gold, entry))
        enriched.append(query.model_copy(update={"gold_papers": gold_papers}))
    return enriched


def _crosswalk_index(
    artifact: SciFactCrosswalkArtifact,
) -> dict[str, SciFactCrosswalkEntry]:
    result: dict[str, SciFactCrosswalkEntry] = {}
    for entry in artifact.entries:
        prior = result.get(entry.s2orc_corpus_id)
        if prior is not None and prior != entry:
            raise ValueError(
                f"conflicting duplicate SciFact crosswalk ID: {entry.s2orc_corpus_id}"
            )
        result[entry.s2orc_corpus_id] = entry
    return result


def _enrich_gold(
    gold: EvalGoldPaper,
    entry: SciFactCrosswalkEntry,
) -> EvalGoldPaper:
    metadata = dict(gold.metadata)
    metadata["evaluator_crosswalk"] = {
        "source": CROSSWALK_SOURCE,
        "connector_version": CROSSWALK_CONNECTOR_VERSION,
        "status": entry.status,
        "external_id_fields": list(entry.external_id_fields),
        "error_type": entry.error_type,
        "http_status": entry.http_status,
    }
    if entry.status != "success":
        metadata["identity_status"] = f"crosswalk_{entry.status}"
        return gold.model_copy(update={"metadata": metadata})

    enriched = gold.model_copy(
        update={
            "doi": entry.doi or gold.doi,
            "arxiv_id": entry.arxiv_id or gold.arxiv_id,
            "semantic_scholar_id": (
                entry.semantic_scholar_id or gold.semantic_scholar_id
            ),
            "pubmed_id": entry.pubmed_id or gold.pubmed_id,
            "metadata": {
                **metadata,
                "identity_status": "official_crosswalk_resolved",
            },
        }
    )
    evidence = identity_evidence(gold, enriched)
    if not evidence.equivalent or evidence.conflicting_identifiers:
        raise ValueError(
            "SciFact crosswalk conflicts with an existing stable identifier"
        )
    return enriched


def _open_dataset(path: Path) -> tuple[Path, Any | None]:
    if path.is_dir():
        root = path / "scifact" if (path / "scifact").is_dir() else path
        return root, None
    if path.is_file() and path.suffix == ".zip":
        temporary = tempfile.TemporaryDirectory(prefix="spar-beir-scifact-")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(temporary.name)
        return Path(temporary.name) / "scifact", temporary.cleanup
    raise ValueError(f"beir_scifact dataset path not found or not a directory: {path}")


def _read_jsonl(path: Path, label: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"beir_scifact missing {label}: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict) or not isinstance(payload.get("_id"), str):
            raise ValueError(f"invalid beir_scifact {label} at line {line_number}")
        key = payload["_id"]
        if key in records:
            raise ValueError(f"duplicate beir_scifact {label} id: {key}")
        records[key] = payload
    return records


def _read_qrels(path: Path) -> list[tuple[str, str, int]]:
    if not path.is_file():
        raise ValueError(f"beir_scifact missing qrels/test.tsv: {path}")
    rows: list[tuple[str, str, int]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("invalid beir_scifact qrels header")
        for row in reader:
            rows.append((row["query-id"], row["corpus-id"], int(row["score"])))
    return rows


def _required_string(payload: dict[str, Any], field: str, query_id: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"beir_scifact query {query_id} missing {field}")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
