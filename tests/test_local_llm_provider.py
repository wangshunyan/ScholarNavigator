from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from scholar_agent.llm.local_provider import (
    LocalCompletion,
    LocalProviderError,
    canonical_json_object,
    create_app,
)


class FakeService:
    def __init__(self, content: str = '{"queries":["graph neural networks"]}') -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        messages,
        *,
        max_tokens: int,
        temperature: float,
        force_json_object: bool = False,
    ):  # noqa: ANN001
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "force_json_object": force_json_object,
            }
        )
        return LocalCompletion(content=self.content, prompt_tokens=11, completion_tokens=7)


class FailingService:
    def complete(
        self,
        messages,
        *,
        max_tokens: int,
        temperature: float,
        force_json_object: bool = False,
    ):  # noqa: ANN001
        del messages, max_tokens, temperature, force_json_object
        raise LocalProviderError("model_generation_failed")


def test_canonical_json_object_rejects_non_object_output() -> None:
    assert canonical_json_object('{"b":2,"a":1}') == '{"a":1,"b":2}'
    for content in ("not json", "[]", "null"):
        try:
            canonical_json_object(content)
        except LocalProviderError:
            continue
        raise AssertionError("invalid model output was accepted")


def test_openai_compatible_endpoint_records_usage_and_ignores_optional_fields() -> None:
    service = FakeService()
    client = TestClient(create_app(service, model_id="Qwen/Qwen3-4B"))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-4B",
            "messages": [{"role": "user", "content": "Return a JSON plan."}],
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"thinking": False},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert json.loads(body["choices"][0]["message"]["content"]) == {
        "queries": ["graph neural networks"]
    }
    assert body["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert service.calls == [
        {
            "messages": [{"role": "user", "content": "Return a JSON plan."}],
            "max_tokens": 64,
            "temperature": 0,
            "force_json_object": True,
        }
    ]


def test_endpoint_rejects_nonzero_temperature_and_invalid_model_json() -> None:
    client = TestClient(create_app(FakeService("not-json"), model_id="Qwen/Qwen3-4B"))
    base = {
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role": "user", "content": "Return JSON."}],
    }

    assert client.post("/v1/chat/completions", json={**base, "temperature": 1}).status_code == 400
    assert client.post("/v1/chat/completions", json={**base, "temperature": 0}).status_code == 502


def test_endpoint_only_enables_json_prefix_for_json_object_response_format() -> None:
    service = FakeService()
    client = TestClient(create_app(service, model_id="Qwen/Qwen3-4B"))
    base = {
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role": "user", "content": "Return JSON."}],
        "temperature": 0,
    }

    assert client.post("/v1/chat/completions", json=base).status_code == 200
    assert (
        client.post(
            "/v1/chat/completions",
            json={**base, "response_format": {"type": "json_object"}},
        ).status_code
        == 200
    )
    assert [call["force_json_object"] for call in service.calls] == [False, True]


def test_endpoint_logs_only_stable_error_code(caplog) -> None:  # noqa: ANN001
    client = TestClient(create_app(FailingService(), model_id="Qwen/Qwen3-4B"))
    request_content = "private benchmark query must never reach provider logs"

    with caplog.at_level(logging.WARNING, logger="scholar_agent.local_provider"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-4B",
                "messages": [{"role": "user", "content": request_content}],
                "temperature": 0,
            },
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "model_generation_failed"
    assert "local_provider_error:model_generation_failed" in caplog.text
    assert request_content not in caplog.text
