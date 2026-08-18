from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.datasets.beir_scifact import (
    load_beir_scifact,
    load_beir_scifact_enriched,
)
from scholar_agent.evaluation.datasets.scifact_crosswalk import (
    SciFactCrosswalkArtifact,
    SciFactCrosswalkEntry,
    write_crosswalk,
)


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "scifact"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "\n".join([
            json.dumps({"_id": "d1", "title": "T—one", "text": "A"}),
            json.dumps({"_id": "d2", "text": "B"}),
        ]) + "\n", encoding="utf-8")
    (root / "queries.jsonl").write_text(
        "\n".join(json.dumps({"_id": str(i), "text": f"query {i}"}) for i in range(55)) + "\n",
        encoding="utf-8")
    qrels = ["query-id\tcorpus-id\tscore"]
    qrels.extend(f"{i}\t{'d1' if i % 2 == 0 else 'd2'}\t1" for i in range(55))
    qrels.append("0\td2\t1")
    qrels.append("0\td1\t1")
    (root / "qrels" / "test.tsv").write_text("\n".join(qrels) + "\n", encoding="utf-8")
    return root


def test_scifact_sampling_is_sha256_deterministic_and_preserves_metadata(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    first = load_beir_scifact(root)
    second = load_beir_scifact(root)
    assert [q.query_id for q in first] == [q.query_id for q in second]
    assert len(first) == 50
    assert any(len(query.gold_papers) == 2 for query in first)
    assert all(
        paper.metadata["identity_status"] == "corpus_id_resolved"
        for query in first
        for paper in query.gold_papers
    )
    assert all(
        paper.s2orc_corpus_id == paper.metadata["s2orc_corpus_id"]
        for query in first
        for paper in query.gold_papers
    )
    assert any(
        paper.metadata["abstract"] == "B"
        and paper.title is None
        for query in first
        for paper in query.gold_papers
    )


def test_scifact_rejects_unmapped_qrels(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nmissing\td1\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="qrels mapping"):
        load_beir_scifact(root)


def test_scifact_requires_complete_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing corpus"):
        load_beir_scifact(tmp_path)


def test_scifact_enrichment_changes_only_offline_gold(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    plain = load_beir_scifact(root)
    corpus_ids = sorted(
        {
            str(gold.s2orc_corpus_id)
            for query in plain
            for gold in query.gold_papers
        }
    )
    crosswalk = tmp_path / "crosswalk.json"
    write_crosswalk(
        crosswalk,
        SciFactCrosswalkArtifact(
            entries=[
                SciFactCrosswalkEntry(
                    s2orc_corpus_id=corpus_id,
                    status="success",
                    doi=f"10.1000/{corpus_id}",
                    external_id_fields=["DOI"],
                )
                for corpus_id in corpus_ids
            ]
        ),
    )
    enriched = load_beir_scifact_enriched(root, crosswalk_path=crosswalk)
    assert [query.query for query in enriched] == [query.query for query in plain]
    assert [query.metadata for query in enriched] == [query.metadata for query in plain]
    assert all(
        gold.doi and gold.metadata["identity_status"] == "official_crosswalk_resolved"
        for query in enriched
        for gold in query.gold_papers
    )
    assert all(
        "evaluator_crosswalk" not in query.metadata for query in enriched
    )
