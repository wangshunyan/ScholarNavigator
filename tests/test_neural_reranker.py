from __future__ import annotations

from pathlib import Path

from scholar_agent.agents.neural_reranker import (
    NeuralReranker,
    NeuralRerankerConfig,
    model_fingerprint,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers


def _papers() -> list[Paper]:
    return [
        Paper(
            title="First paper",
            abstract="first abstract",
            identifiers=PaperIdentifiers(arxiv_id="2501.00002"),
        ),
        Paper(
            title="Second paper",
            abstract="second abstract",
            identifiers=PaperIdentifiers(arxiv_id="2501.00001"),
        ),
    ]


def test_missing_local_model_falls_back_without_changing_candidates(tmp_path: Path) -> None:
    reranker = NeuralReranker(
        NeuralRerankerConfig(model_path=tmp_path / "missing-model")
    )

    result = reranker.rerank("retrieval", _papers(), limit=2)

    assert result.fallback_used is True
    assert result.error == "OSError"
    assert [paper.title for paper in result.papers] == [
        "First paper",
        "Second paper",
    ]


def test_model_fingerprint_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    file = model / "config.json"
    file.write_text('{"labels": 1}\n', encoding="utf-8")
    first = model_fingerprint(model)
    second = model_fingerprint(model)
    file.write_text('{"labels": 2}\n', encoding="utf-8")

    assert first == second
    assert first != model_fingerprint(model)
