from __future__ import annotations

import subprocess
from pathlib import Path

from scholar_agent.agents.neural_reranker import (
    NeuralReranker,
    NeuralRerankerConfig,
    RERANKER_PROMPT_VERSION,
    _format_reranker_prompt,
    _causal_relevance_scores,
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


def test_subprocess_inference_failure_falls_back_without_raising(
    tmp_path: Path,
) -> None:
    reranker = NeuralReranker(
        NeuralRerankerConfig(model_path=tmp_path / "model")
    )
    reranker._load = lambda: (_ for _ in ()).throw(  # type: ignore[method-assign]
        subprocess.CalledProcessError(1, "compiler")
    )

    result = reranker.rerank("retrieval", _papers(), limit=2)

    assert result.fallback_used is True
    assert result.error == "CalledProcessError"
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


def test_causal_scores_use_last_token_for_left_padding() -> None:
    torch = __import__("torch")

    class Tokenizer:
        padding_side = "left"

        def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
            return [1] if value.lower() == "yes" else [0] if value.lower() == "no" else [2, 3]

    input_ids = torch.zeros((2, 4), dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    logits = torch.zeros((2, 4, 4), dtype=torch.float32)
    logits[0, 3, 1] = 4
    logits[0, 3, 0] = -4
    logits[1, 3, 1] = 3
    logits[1, 3, 0] = -3

    scores = _causal_relevance_scores(logits, input_ids, attention_mask, Tokenizer())

    assert scores.tolist() == [8.0, 6.0]


def test_causal_scores_use_last_available_logit_when_qwen_omits_one_position() -> None:
    torch = __import__("torch")

    class Tokenizer:
        padding_side = "left"

        def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
            return [1] if value.lower() == "yes" else [0] if value.lower() == "no" else [2]

    input_ids = torch.zeros((1, 5), dtype=torch.long)
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    logits[0, 3, 1] = 4
    logits[0, 3, 0] = -2

    assert _causal_relevance_scores(logits, input_ids, attention_mask, Tokenizer()).tolist() == [6.0]


def test_qwen_prompt_is_deterministic_and_preserves_query_and_document() -> None:
    prompt = _format_reranker_prompt(
        "find retrieval papers",
        "A paper about dense retrieval.",
        "academic retrieval task",
    )

    assert RERANKER_PROMPT_VERSION == "qwen3-reranker-v1"
    assert prompt == (
        "<Instruct>: academic retrieval task\n\n"
        "<Query>: find retrieval papers\n\n"
        "<Document>: A paper about dense retrieval."
    )


def test_causal_scores_move_logits_to_cpu_before_indexing() -> None:
    torch = __import__("torch")

    class Tokenizer:
        padding_side = "right"

        def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
            return [1] if value == "yes" else [0]

    input_ids = torch.zeros((1, 3), dtype=torch.long)
    mask = torch.tensor([[1, 1, 0]])
    logits = torch.zeros((1, 3, 2), dtype=torch.float32)
    logits[0, 1, 1] = 2
    logits[0, 1, 0] = -1

    assert _causal_relevance_scores(logits, input_ids, mask, Tokenizer()).tolist() == [3.0]


def test_causal_scores_gather_each_row_from_a_contiguous_cpu_tensor() -> None:
    torch = __import__("torch")

    class Tokenizer:
        padding_side = "left"

        def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
            return [1] if value == "yes" else [0]

    input_ids = torch.zeros((2, 4), dtype=torch.long)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    # A transposed view covers the non-contiguous storage case before gather.
    logits = torch.zeros((4, 2, 2), dtype=torch.float32).transpose(0, 1)
    logits[0, 3, 1] = 5
    logits[0, 3, 0] = -1
    logits[1, 3, 1] = 2
    logits[1, 3, 0] = -2

    assert _causal_relevance_scores(logits, input_ids, mask, Tokenizer()).tolist() == [6.0, 4.0]


def test_rerank_result_exposes_fixed_runtime_limits_and_vram_defaults(tmp_path: Path) -> None:
    result = NeuralReranker(
        NeuralRerankerConfig(model_path=tmp_path / "missing-model")
    ).rerank("query", _papers(), limit=2)

    assert result.batch_size == 8
    assert result.candidate_limit == 120
    assert result.peak_vram_bytes == 0
