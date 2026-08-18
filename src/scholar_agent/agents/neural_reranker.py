"""Optional local cross-encoder reranking with auditable resource metadata."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scholar_agent.core.paper_schemas import Paper


@dataclass(frozen=True)
class NeuralRerankerConfig:
    model_path: Path
    candidate_limit: int = 120
    batch_size: int = 8
    device: str = "auto"


@dataclass(frozen=True)
class NeuralRerankResult:
    papers: list[Paper]
    latency_seconds: float
    model_fingerprint: str
    batch_count: int = 0
    fallback_used: bool = False
    error: str | None = None


class NeuralReranker:
    """Load a local sequence-classification cross-encoder on first use.

    The adapter intentionally accepts only query/paper text and never receives
    evaluation labels or benchmark case metadata.
    """

    def __init__(self, config: NeuralRerankerConfig) -> None:
        if config.candidate_limit <= 0 or config.batch_size <= 0:
            raise ValueError("neural_reranker_limits_invalid")
        self.config = config
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._model_kind: str | None = None
        self._fingerprint = model_fingerprint(config.model_path)

    def rerank(self, query: str, papers: list[Paper], *, limit: int) -> NeuralRerankResult:
        started = time.perf_counter()
        selected = list(papers[: self.config.candidate_limit])
        if not selected or limit <= 0:
            return NeuralRerankResult(
                papers=[],
                latency_seconds=time.perf_counter() - started,
                model_fingerprint=self._fingerprint,
                batch_count=0,
            )
        try:
            self._load()
            assert self._tokenizer is not None
            assert self._model is not None
            assert self._torch is not None
            scores: list[float] = []
            batch_count = 0
            for start in range(0, len(selected), self.config.batch_size):
                batch_count += 1
                batch = selected[start : start + self.config.batch_size]
                texts = [
                    f"{paper.title}\n{paper.abstract}".strip()
                    for paper in batch
                ]
                encoded = self._tokenizer(
                    [query] * len(texts),
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                with self._torch.inference_mode():
                    output = self._model(**encoded)
                if self._model_kind == "causal_lm":
                    scores.extend(
                        _causal_relevance_scores(
                            output.logits,
                            encoded["input_ids"],
                            encoded.get("attention_mask"),
                            self._tokenizer,
                        ).tolist()
                    )
                else:
                    scores.extend(_positive_scores(output.logits).tolist())
            ranked = sorted(
                zip(selected, scores, strict=True),
                key=lambda item: (-float(item[1]), _stable_paper_key(item[0])),
            )
            return NeuralRerankResult(
                papers=[paper for paper, _score in ranked[:limit]],
                latency_seconds=time.perf_counter() - started,
                model_fingerprint=self._fingerprint,
                batch_count=batch_count,
            )
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            ImportError,
            subprocess.SubprocessError,
        ) as exc:
            return NeuralRerankResult(
                papers=selected[:limit],
                latency_seconds=time.perf_counter() - started,
                model_fingerprint=self._fingerprint,
                batch_count=(len(selected) + self.config.batch_size - 1) // self.config.batch_size,
                fallback_used=True,
                error=type(exc).__name__,
            )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._device = torch.device(device_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.model_path), local_files_only=True
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        config_path = Path(self.config.model_path) / "config.json"
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        architectures = model_config.get("architectures") or []
        prefer_causal_lm = any("CausalLM" in str(value) for value in architectures)
        if prefer_causal_lm:
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.config.model_path), local_files_only=True
            )
            self._model_kind = "causal_lm"
            self._model.to(self._device)
            self._model.eval()
            return
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.config.model_path), local_files_only=True
            )
            self._model_kind = "sequence_classification"
        except (OSError, RuntimeError, TypeError, ValueError):
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.config.model_path), local_files_only=True
            )
            self._model_kind = "causal_lm"
        self._model.to(self._device)
        self._model.eval()


def rerank_local_papers(
    query: str,
    papers: list[Paper],
    config: NeuralRerankerConfig,
    *,
    limit: int,
) -> NeuralRerankResult:
    return NeuralReranker(config).rerank(query, papers, limit=limit)


def model_fingerprint(path: Path) -> str:
    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    inventory: list[dict[str, Any]] = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = file_path.relative_to(root).as_posix()
        stat = file_path.stat()
        inventory.append({"path": relative, "size": stat.st_size})
        digest.update(relative.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(file_path.read_bytes())
    if not inventory:
        return "missing-or-empty:" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return digest.hexdigest()


def _positive_scores(logits: Any) -> Any:
    if getattr(logits, "ndim", 0) == 1:
        return logits
    if int(logits.shape[-1]) == 1:
        return logits[:, 0]
    return logits[:, -1]


def _causal_relevance_scores(
    logits: Any,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
) -> Any:
    """Score the final-token yes/no decision used by Qwen rerankers."""

    if attention_mask is None:
        positions = [int(input_ids.shape[1]) - 1] * int(input_ids.shape[0])
    else:
        positions = attention_mask.sum(dim=1).to("cpu").tolist()
        positions = [int(position) - 1 for position in positions]
    positive = _single_token_id(tokenizer, ("yes", "Yes", "是"))
    negative = _single_token_id(tokenizer, ("no", "No", "否"))
    if positive is None or negative is None:
        raise ValueError("neural_reranker_yes_no_tokens_missing")
    row_indices = logits.new_tensor(range(len(positions)), dtype=None).long()
    column_indices = logits.new_tensor(positions, dtype=None).long()
    positive_logits = logits[row_indices, column_indices, positive]
    negative_logits = logits[row_indices, column_indices, negative]
    return positive_logits - negative_logits


def _single_token_id(tokenizer: Any, values: tuple[str, ...]) -> int | None:
    for value in values:
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        if len(token_ids) == 1:
            return int(token_ids[0])
    return None


def _stable_paper_key(paper: Paper) -> str:
    identifier = str(paper.identifiers.arxiv_id or "").strip().casefold()
    if identifier:
        return "arxiv:" + identifier
    return "title:" + " ".join(paper.title.casefold().split())
