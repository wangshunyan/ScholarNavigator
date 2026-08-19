"""Loopback-only OpenAI-compatible provider backed by a local Transformers model."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


LOGGER = logging.getLogger("scholar_agent.local_provider")


class LocalProviderError(RuntimeError):
    """Raised when a local model cannot produce a strict JSON response."""


class ChatMessage(BaseModel):
    """Minimal OpenAI-compatible chat message accepted by the local service."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of a non-streaming OpenAI chat-completions request."""

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 0
    max_tokens: int = Field(default=1024, ge=1, le=1024)
    stream: bool = False


@dataclass(frozen=True)
class LocalCompletion:
    """Validated completion payload and provider-neutral token accounting."""

    content: str
    prompt_tokens: int
    completion_tokens: int


class LocalChatService(Protocol):
    """Small boundary that keeps HTTP tests independent from Torch and a GPU."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LocalCompletion: ...


def canonical_json_object(content: str) -> str:
    """Reject non-object model output instead of repairing it before auditing."""

    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LocalProviderError("model_output_not_valid_json") from exc
    if not isinstance(value, dict):
        raise LocalProviderError("model_output_not_json_object")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def create_app(service: LocalChatService, *, model_id: str) -> FastAPI:
    """Create a loopback service compatible with the project's raw HTTP client."""

    app = FastAPI(title="ScholarNavigator local LLM provider", version="1")

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming_not_supported")
        if request.temperature != 0:
            raise HTTPException(status_code=400, detail="temperature_must_be_zero")
        try:
            completion = service.complete(
                [message.model_dump() for message in request.messages],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            content = canonical_json_object(completion.content)
        except LocalProviderError as exc:
            # Error codes are fixed identifiers, so logs remain useful without
            # retaining untrusted prompts, completions, or request metadata.
            LOGGER.warning("local_provider_error:%s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive service boundary
            LOGGER.exception("local_provider_error:unexpected_%s", type(exc).__name__)
            raise HTTPException(
                status_code=502,
                detail="local_provider_internal_error",
            ) from exc

        prompt_tokens = max(0, int(completion.prompt_tokens))
        completion_tokens = max(0, int(completion.completion_tokens))
        return {
            "id": f"local-{time.time_ns()}",
            "object": "chat.completion",
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


class TransformersLocalChatService:
    """Lazy GPU inference adapter for an instruction-tuned causal language model."""

    def __init__(self, *, tokenizer: Any, model: Any, device: str, max_input_tokens: int):
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._max_input_tokens = max_input_tokens

    @classmethod
    def load(
        cls,
        *,
        model_path: str,
        device: str,
        max_input_tokens: int,
    ) -> "TransformersLocalChatService":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment failure
            raise LocalProviderError("local_transformers_dependencies_missing") from exc

        if not torch.cuda.is_available():
            raise LocalProviderError("cuda_not_available")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_input_tokens=max_input_tokens,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> LocalCompletion:
        if temperature != 0:
            raise LocalProviderError("temperature_must_be_zero")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - validated at load time
            raise LocalProviderError("torch_not_available") from exc

        try:
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as exc:
            raise LocalProviderError("chat_template_failed") from exc
        try:
            encoded = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_input_tokens,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            prompt_tokens = int(encoded["input_ids"].shape[-1])
        except Exception as exc:
            raise LocalProviderError("tokenization_failed") from exc
        try:
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_tokens,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            completion_ids = generated[0, prompt_tokens:]
            content = self._tokenizer.decode(
                completion_ids, skip_special_tokens=True
            ).strip()
        except Exception as exc:
            raise LocalProviderError("model_generation_failed") from exc
        canonical = canonical_json_object(content)
        return LocalCompletion(
            content=canonical,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(completion_ids.shape[-1]),
        )
