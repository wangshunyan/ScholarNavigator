from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scholar_agent.core.evaluation_schemas import EvalGoldPaper
from scholar_agent.core.identity import identity_evidence
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.evaluation.datasets.beir_scifact import _enrich_gold
from scholar_agent.evaluation.datasets.scifact_crosswalk import (
    SciFactCrosswalkArtifact,
    SciFactCrosswalkEntry,
    SciFactCrosswalkSnapshot,
    SciFactCrosswalkStore,
    crosswalk_content_hash,
    crosswalk_snapshot_key,
    fetch_exact_corpus_id,
    load_crosswalk,
    record_missing_crosswalk,
    replay_crosswalk,
    write_crosswalk,
)
from scholar_agent.evaluation.metrics import evaluable_gold_count


class _Response:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _no_throttle(**_kwargs: object) -> float:
    return 0.0


def _snapshot(
    corpus_id: str,
    *,
    status: str = "success",
) -> SciFactCrosswalkSnapshot:
    snapshot = SciFactCrosswalkSnapshot(
        key=crosswalk_snapshot_key(corpus_id),
        s2orc_corpus_id=corpus_id,
        status=status,
        semantic_scholar_id="paper-1" if status == "success" else None,
        returned_corpus_id=corpus_id if status == "success" else None,
        external_ids={"DOI": "10.1000/example"} if status == "success" else {},
        error_type=None if status == "success" else "network_timeout",
        request_count=1,
        recorded_at="2026-07-21T00:00:00+00:00",
        content_hash="0" * 64,
    )
    return snapshot.model_copy(update={"content_hash": crosswalk_content_hash(snapshot)})


def test_exact_lookup_normalizes_multiple_external_ids_and_numeric_corpus_id() -> None:
    captured: dict[str, Any] = {}

    def opener(request: Any, *, timeout: float) -> _Response:
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        return _Response(
            {
                "paperId": "ABC123",
                "corpusId": 42,
                "externalIds": {
                    "CorpusId": "42",
                    "DOI": "https://doi.org/10.1000/Example",
                    "ArXiv": "arXiv:2401.00001v2",
                    "PubMed": 123456,
                },
            }
        )

    result = fetch_exact_corpus_id(
        42,
        opener=opener,
        max_retries=0,
        throttle=_no_throttle,
    )
    assert result.status == "success"
    assert result.returned_corpus_id == "42"
    assert result.external_ids == {
        "DOI": "10.1000/example",
        "ArXiv": "2401.00001",
        "PubMed": "123456",
    }
    assert "CorpusId%3A42" not in captured["url"]
    assert "/CorpusId:42?" in captured["url"]
    assert captured["timeout"] > 0


def test_exact_lookup_accepts_missing_external_ids_without_title_inference() -> None:
    result = fetch_exact_corpus_id(
        "CorpusId:7",
        opener=lambda *_args, **_kwargs: _Response(
            {"paperId": "paper-7", "corpusId": "7", "externalIds": None}
        ),
        max_retries=0,
        throttle=_no_throttle,
    )
    assert result.status == "success"
    assert result.external_ids == {}
    gold = EvalGoldPaper(
        title="Same title",
        year=2020,
        s2orc_corpus_id="7",
        metadata={"evaluator_crosswalk": {"status": "success"}},
    )
    candidate = Paper(title="Same title", year=2020, authors=["Author"])
    assert identity_evidence(gold, candidate).equivalent is False


def test_enrichment_reuses_unified_conflict_rule() -> None:
    gold = EvalGoldPaper(s2orc_corpus_id="9", doi="10.1000/original")
    entry = SciFactCrosswalkEntry(
        s2orc_corpus_id="9",
        status="success",
        doi="10.1000/different",
    )
    with pytest.raises(ValueError, match="conflicts"):
        _enrich_gold(gold, entry)


def test_failed_crosswalk_is_explicit_and_excluded_from_retrieval_miss_denominator() -> None:
    gold = _enrich_gold(
        EvalGoldPaper(title="Unresolved", s2orc_corpus_id="11"),
        SciFactCrosswalkEntry(
            s2orc_corpus_id="11",
            status="failed",
            error_type="network_timeout",
        ),
    )
    assert gold.metadata["identity_status"] == "crosswalk_failed"
    assert evaluable_gold_count([gold]) == 0


def test_record_missing_deduplicates_ids_and_replay_is_network_free(tmp_path: Path) -> None:
    store = SciFactCrosswalkStore(tmp_path / "snapshots")
    calls: list[str] = []

    def fetcher(corpus_id: Any) -> SciFactCrosswalkSnapshot:
        calls.append(str(corpus_id))
        return _snapshot(str(corpus_id))

    first = record_missing_crosswalk(["2", 1, "2", "1"], store, fetcher=fetcher)
    second = record_missing_crosswalk(
        ["1", "2"],
        store,
        fetcher=lambda _value: pytest.fail("existing terminal must not be fetched"),
    )
    assert calls == ["1", "2"]
    assert first == {"planned": 2, "existing": 0, "written": 2}
    assert second == {"planned": 2, "existing": 2, "written": 0}
    before = sorted(path.read_bytes() for path in store.entries_dir.glob("*.json"))
    first_replay = replay_crosswalk(["2", "1", "2"], store)
    second_replay = replay_crosswalk(["1", "2"], store)
    after = sorted(path.read_bytes() for path in store.entries_dir.glob("*.json"))
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    write_crosswalk(first_output, first_replay)
    write_crosswalk(second_output, second_replay)
    assert first_replay == second_replay
    assert before == after
    assert first_output.read_bytes() == second_output.read_bytes()


def test_crosswalk_artifact_rejects_conflicting_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "crosswalk.json"
    payload = SciFactCrosswalkArtifact(
        entries=[
            SciFactCrosswalkEntry(s2orc_corpus_id="4", status="success", doi="10.1/a"),
            SciFactCrosswalkEntry(s2orc_corpus_id="4", status="success", doi="10.1/b"),
        ]
    )
    write_crosswalk(path, payload)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        load_crosswalk(path)


def test_old_gold_without_crosswalk_metadata_remains_evaluable() -> None:
    assert evaluable_gold_count([EvalGoldPaper(s2orc_corpus_id="legacy")]) == 1
