"""Optional local cross-encoder reranking with auditable resource metadata."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scholar_agent.core.paper_schemas import Paper


RERANKER_PROMPT_VERSION = "qwen3-reranker-v1"
RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
# Qwen3 accepts a longer context, but 2,048 tokens keeps a full batch of eight
# academic title/abstract pairs within the qualification GPU resource envelope.
RERANKER_MAX_LENGTH = 2048
_RERANKER_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and '
    'the Instruct provided. Note that the answer can only be "yes" or "no."'
    '<|im_end|>\n<|im_start|>user\n'
)
_RERANKER_SUFFIX = (
    '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
)

@dataclass(frozen=True)
class NeuralRerankerConfig:
    model_path: Path
    candidate_limit: int = 120
    batch_size: int = 8
    device: str = "auto"
    max_length: int = RERANKER_MAX_LENGTH
    prompt_version: str = RERANKER_PROMPT_VERSION


@dataclass(frozen=True)
class NeuralRerankResult:
    papers: list[Paper]
    latency_seconds: float
    model_fingerprint: str
    batch_count: int = 0
    fallback_used: bool = False
    error: str | None = None
    prompt_version: str = RERANKER_PROMPT_VERSION
    model_kind: str | None = None
    device: str | None = None
    max_length: int = RERANKER_MAX_LENGTH
    candidate_count: int = 0
    inference_success: bool = False
    batch_size: int = 0
    candidate_limit: int = 0
    peak_vram_bytes: int = 0


class NeuralReranker:
    """Load a local sequence-classification cross-encoder on first use.

    The adapter intentionally accepts only query/paper text and never receives
    evaluation labels or benchmark case metadata.
    """

    def __init__(self, config: NeuralRerankerConfig) -> None:
        if (
            config.candidate_limit <= 0
            or config.batch_size <= 0
            or config.max_length <= 0
            or not config.prompt_version.strip()
        ):
            raise ValueError("neural_reranker_limits_invalid")
        self.config = config
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._model_kind: str | None = None
        self._device_name: str | None = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
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
                prompt_version=self.config.prompt_version,
                model_kind=self._model_kind,
                device=self._device_name,
                max_length=self.config.max_length,
                candidate_count=0,
            )
        try:
            self._load()
            assert self._tokenizer is not None
            assert self._model is not None
            assert self._torch is not None
            if self._device is not None and getattr(self._device, "type", "") == "cuda":
                self._torch.cuda.reset_peak_memory_stats(self._device)
            scores: list[float] = []
            batch_count = 0
            for start in range(0, len(selected), self.config.batch_size):
                batch_count += 1
                batch = selected[start : start + self.config.batch_size]
                texts = [
                    f"{paper.title}\n{paper.abstract}".strip()
                    for paper in batch
                ]
                encoded = self._encode_pairs(query, texts)
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
                        padding_side="left",
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
                prompt_version=self.config.prompt_version,
                model_kind=self._model_kind,
                device=self._device_name,
                max_length=self.config.max_length,
                candidate_count=len(selected),
                inference_success=True,
                batch_size=self.config.batch_size,
                candidate_limit=self.config.candidate_limit,
                peak_vram_bytes=self._peak_vram_bytes(),
            )
        except Exception as exc:
            return NeuralRerankResult(
                papers=selected[:limit],
                latency_seconds=time.perf_counter() - started,
                model_fingerprint=self._fingerprint,
                batch_count=(len(selected) + self.config.batch_size - 1) // self.config.batch_size,
                fallback_used=True,
                error=type(exc).__name__,
                prompt_version=self.config.prompt_version,
                model_kind=self._model_kind,
                device=self._device_name,
                max_length=self.config.max_length,
                candidate_count=len(selected),
                batch_size=self.config.batch_size,
                candidate_limit=self.config.candidate_limit,
                peak_vram_bytes=self._peak_vram_bytes(),
            )

    def _peak_vram_bytes(self) -> int:
        if self._torch is None or self._device is None:
            return 0
        if getattr(self._device, "type", "") != "cuda":
            return 0
        try:
            return int(self._torch.cuda.max_memory_allocated(self._device))
        except Exception:
            return 0

    def _encode_pairs(self, query: str, documents: list[str]) -> Any:
        assert self._tokenizer is not None
        body_limit = self.config.max_length - len(self._prefix_tokens) - len(
            self._suffix_tokens
        )
        if body_limit <= 0:
            raise ValueError("neural_reranker_prompt_exceeds_max_length")
        pairs = [
            _format_reranker_prompt(query, document, RERANKER_INSTRUCTION)
            for document in documents
        ]
        tokenized = self._tokenizer(
            pairs,
            padding=False,
            truncation=True,
            max_length=body_limit,
            return_attention_mask=False,
        )
        input_ids = [
            self._prefix_tokens + list(tokens) + self._suffix_tokens
            for tokens in tokenized["input_ids"]
        ]
        encoded = self._tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.config.max_length:
            raise ValueError("neural_reranker_encoded_length_exceeded")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None and not bool(attention_mask[:, -1].all()):
            raise ValueError("neural_reranker_final_token_is_padding")
        return encoded

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.config.model_path.is_dir():
            raise OSError(f"neural_reranker_model_missing:{self.config.model_path}")
        import torch
        from transformers import AutoTokenizer

        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._device = torch.device(device_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.model_path), local_files_only=True, padding_side="left"
        )
        self._tokenizer.padding_side = "left"
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
            self._finish_load()
            return
        from transformers import AutoModelForSequenceClassification
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
        self._finish_load()

    def _finish_load(self) -> None:
        self._device_name = str(self._device)
        assert self._tokenizer is not None
        self._prefix_tokens = self._tokenizer.encode(
            _RERANKER_PREFIX, add_special_tokens=False
        )
        self._suffix_tokens = self._tokenizer.encode(
            _RERANKER_SUFFIX, add_special_tokens=False
        )


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
    *,
    padding_side: str | None = None,
) -> Any:
    """Score the final-token yes/no decision used by Qwen rerankers."""

    if getattr(logits, "ndim", 0) != 3:
        raise ValueError("neural_reranker_logits_shape_invalid")
    input_sequence_length = int(input_ids.shape[1])
    logits_sequence_length = int(logits.shape[1])
    batch_size = int(input_ids.shape[0])
    if logits_sequence_length <= 0 or input_sequence_length - logits_sequence_length not in (0, 1):
        raise ValueError("neural_reranker_logits_sequence_misaligned")
    if attention_mask is None:
        positions = [logits_sequence_length - 1] * batch_size
    elif (padding_side or getattr(tokenizer, "padding_side", "right")) == "left":
        # Causal-LM APIs score the token following the final prompt token. Some
        # Qwen attention paths return one fewer logit than input tokens, so the
        # last actual logit, rather than the input width, is authoritative.
        positions = [logits_sequence_length - 1] * batch_size
    else:
        positions = [
            min(int(position) - 1, logits_sequence_length - 1)
            for position in attention_mask.sum(dim=1).to("cpu").tolist()
        ]
    if any(position < 0 or position >= logits_sequence_length for position in positions):
        raise ValueError("neural_reranker_token_position_out_of_bounds")
    positive = _single_token_id(tokenizer, ("yes", "Yes", "是"))
    negative = _single_token_id(tokenizer, ("no", "No", "否"))
    vocab_size = int(logits.shape[-1])
    if positive is None or negative is None or not (
        0 <= positive < vocab_size and 0 <= negative < vocab_size
    ):
        raise ValueError("neural_reranker_yes_no_tokens_missing")
    # Move to a contiguous CPU tensor before selecting the decision token.
    # CPU gather avoids CUDA's asynchronous advanced-index kernel and remains
    # well-defined when Qwen returns a view with nonstandard strides.
    logits_cpu = logits.detach().to("cpu").contiguous()
    position_indices = (
        logits_cpu.new_tensor(positions)
        .long()
        .view(batch_size, 1, 1)
        .expand(-1, 1, vocab_size)
    )
    final_logits = logits_cpu.gather(1, position_indices).squeeze(1)
    positive_logits = final_logits[:, positive]
    negative_logits = final_logits[:, negative]
    return positive_logits - negative_logits


def _format_reranker_prompt(instruction: str, document: str, task: str) -> str:
    """Build the stable body used by the Qwen3 reranker prompt."""

    return (
        f"<Instruct>: {task}\n\n<Query>: {instruction}\n\n<Document>: {document}"
    )


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
