from __future__ import annotations

import json
import socket
from typing import Literal
from urllib.error import HTTPError, URLError

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from scholar_agent.llm import provider


class FakeResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_llm_disabled_when_provider_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)

    runtime = provider.get_llm_runtime_config()

    assert provider.is_llm_enabled() is False
    assert runtime.provider == "disabled"
    assert runtime.available is False
    assert runtime.reason == "provider_disabled"


def test_openai_compatible_missing_api_key_is_not_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv(provider.PROVIDER_ENV, provider.SUPPORTED_PROVIDER)
    monkeypatch.setenv(provider.BASE_URL_ENV, "https://api.example.test/v1")
    monkeypatch.setenv(provider.MODEL_ENV, "gpt-test")

    runtime = provider.get_llm_runtime_config().model_dump()

    assert runtime["available"] is False
    assert runtime["provider"] == "openai_compatible"
    assert runtime["model"] == "gpt-test"
    assert runtime["base_url_host"] == "api.example.test"
    assert provider.API_KEY_ENV in runtime["reason"]
    assert "secret" not in json.dumps(runtime)


def test_runtime_config_does_not_expose_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_enabled_env(monkeypatch)

    runtime = provider.get_llm_runtime_config().model_dump()

    assert runtime["available"] is True
    assert runtime["provider"] == "openai_compatible"
    assert runtime["base_url_host"] == "api.example.test"
    assert "sk-test-secret" not in json.dumps(runtime)


def test_chat_json_parses_openai_compatible_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "language": "en",
                                    "intent": "recent_progress",
                                    "domain": "machine_learning",
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    result = provider.chat_json(
        [{"role": "user", "content": "Analyze LLM reranking."}],
        timeout=3,
    )

    assert result["intent"] == "recent_progress"
    assert captured["url"] == "https://api.example.test/v1/chat/completions"
    assert captured["timeout"] == 3
    assert captured["authorization"] == "Bearer sk-test-secret"
    assert captured["payload"]["max_tokens"] == provider.DEFAULT_MAX_TOKENS
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["chat_template_kwargs"] == {"thinking": False}
    assert captured["payload"]["stream"] is False


def test_chat_json_records_token_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    client = provider.OpenAICompatibleLLMClient(
        base_url="https://api.example.test/v1",
        api_key="sk-test-secret",
        model="gpt-test",
    )

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        return FakeResponse(
            {
                "choices": [{"message": {"content": json.dumps({"ok": True})}}],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 5,
                    "total_tokens": 22,
                },
            }
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    assert client.chat_json([{"role": "user", "content": "query"}]) == {"ok": True}
    assert client.token_usage.prompt_tokens == 17
    assert client.token_usage.completion_tokens == 5
    assert client.token_usage.total_tokens == 22
    assert client.last_call_usage is not None
    assert client.last_call_usage.prompt_tokens == 17
    assert client.last_call_usage.completion_tokens == 5
    assert client.last_call_usage.total_tokens == 22
    assert client.last_call_usage_fields == {
        "prompt_tokens": 17,
        "completion_tokens": 5,
        "total_tokens": 22,
    }


def test_chat_json_missing_usage_keeps_zero_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = provider.OpenAICompatibleLLMClient(
        base_url="https://api.example.test/v1",
        api_key="sk-test-secret",
        model="gpt-test",
    )

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        return FakeResponse(
            {"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    assert client.chat_json([{"role": "user", "content": "query"}]) == {"ok": True}
    assert client.token_usage.prompt_tokens == 0
    assert client.token_usage.completion_tokens == 0
    assert client.token_usage.total_tokens == 0
    assert client.last_call_usage is None
    assert client.last_call_usage_fields is None


def test_chat_json_clears_previous_call_usage_before_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = provider.OpenAICompatibleLLMClient(
        base_url="https://api.example.test/v1",
        api_key="sk-test-secret",
        model="gpt-test",
    )

    monkeypatch.setattr(
        client,
        "_send_with_retries",
        lambda payload, *, timeout: (
            {
                "choices": [{"message": {"content": json.dumps({"ok": True})}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
            1,
        ),
    )
    client.chat_json([{"role": "user", "content": "first"}])
    assert client.last_call_usage is not None
    assert client.last_call_diagnostics is not None

    def fail(payload, *, timeout):  # noqa: ANN001, ARG001
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(client, "_send_with_retries", fail)
    with pytest.raises(RuntimeError, match="controlled failure"):
        client.chat_json([{"role": "user", "content": "second"}])

    assert client.last_call_usage is None
    assert client.last_call_usage_fields is None
    assert client.last_call_diagnostics is None


def test_chat_json_uses_configured_max_tokens_and_thinking_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    monkeypatch.setenv(provider.MAX_TOKENS_ENV, "256")
    monkeypatch.setenv(provider.NVIDIA_THINKING_ENV, "false")
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": json.dumps({"ok": True})}},
                ]
            }
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    result = provider.chat_json([{"role": "user", "content": "query"}])

    assert result == {"ok": True}
    assert captured["payload"]["max_tokens"] == 256
    assert captured["payload"]["chat_template_kwargs"] == {"thinking": False}


@pytest.mark.parametrize("raw_value", ["0", "-1", "not-an-int"])
def test_chat_json_invalid_max_tokens_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    _set_enabled_env(monkeypatch)
    monkeypatch.setenv(provider.MAX_TOKENS_ENV, raw_value)
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": json.dumps({"ok": True})}},
                ]
            }
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    assert provider.chat_json([{"role": "user", "content": "query"}]) == {"ok": True}
    assert captured["payload"]["max_tokens"] == provider.DEFAULT_MAX_TOKENS


def test_chat_json_timeout_raises_sanitized_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        raise socket.timeout("timed out with sk-test-secret")

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    with pytest.raises(provider.LLMTimeoutError, match="llm_request_timeout"):
        provider.chat_json([{"role": "user", "content": "query"}])


def test_chat_json_http_error_redacts_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        raise HTTPError(
            url="https://api.example.test/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=FakeErrorBody("upstream sk-test-secret failed"),
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    with pytest.raises(provider.LLMProviderError) as exc_info:
        provider.chat_json([{"role": "user", "content": "query"}])

    message = str(exc_info.value)
    assert "llm_http_error:503" in message
    assert "sk-test-secret" not in message
    assert exc_info.value.details.http_status == 503
    assert exc_info.value.details.summary == "upstream [redacted] failed"


def test_chat_json_http_error_exposes_only_sanitized_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        raise HTTPError(
            url="https://api.example.test/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=FakeErrorBody(
                "Unsupported parameter(s): `extra_body`; "
                "Authorization: Bearer sk-test-secret",
                error_type="invalid_request_error",
                code="unsupported_parameter",
                param="extra_body",
            ),
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    with pytest.raises(provider.LLMProviderError) as exc_info:
        provider.chat_json([{"role": "user", "content": "query"}])

    error = exc_info.value
    assert error.details.model_dump() == {
        "http_status": 400,
        "error_type": "invalid_request_error",
        "service_error_code": "unsupported_parameter",
        "summary": (
            "Unsupported parameter(s): `extra_body`; "
            "Authorization: [redacted]"
        ),
        "unsupported_parameters": [],
    }
    assert "sk-test-secret" not in str(error)
    assert "sk-test-secret" not in json.dumps(error.details.model_dump())


def test_chat_json_retries_once_without_unsupported_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    client = provider.OpenAICompatibleLLMClient.from_env()
    messages = [{"role": "user", "content": "Return a status object."}]
    captured_payloads: list[dict[str, object]] = []

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        captured_payloads.append(payload)
        if "response_format" in payload:
            raise HTTPError(
                url=request.full_url,
                code=400,
                msg="Bad Request",
                hdrs=None,
                fp=FakeErrorBody(
                    "Unsupported parameter: `response_format`",
                    error_type="invalid_request_error",
                    code="unsupported_parameter",
                    param="response_format",
                ),
            )
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": json.dumps({"status": "ok"})}}
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            }
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    assert client.chat_json(messages) == {"status": "ok"}

    assert messages == [{"role": "user", "content": "Return a status object."}]
    assert len(captured_payloads) == 2
    assert captured_payloads[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in captured_payloads[1]
    assert captured_payloads[1]["messages"][0] == {
        "role": "system",
        "content": provider.JSON_ONLY_COMPATIBILITY_INSTRUCTION,
    }
    assert client.token_usage.total_tokens == 16
    assert client.last_call_diagnostics is not None
    assert client.last_call_diagnostics.mode == "json_only_prompt"
    assert client.last_call_diagnostics.http_attempts == 2
    assert (
        client.last_call_diagnostics.fallback_reason
        == "unsupported_parameters:response_format"
    )

    captured_payloads.clear()
    assert client.chat_json(messages) == {"status": "ok"}
    assert len(captured_payloads) == 1
    assert "response_format" not in captured_payloads[0]


def test_chat_json_does_not_retry_unrelated_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    attempts = 0

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        nonlocal attempts
        del timeout
        attempts += 1
        raise HTTPError(
            url=request.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=FakeErrorBody("rate limited", code="rate_limit"),
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    with pytest.raises(provider.LLMProviderError) as exc_info:
        provider.chat_json([{"role": "user", "content": "query"}])

    assert attempts == 1
    assert exc_info.value.details.http_status == 429


def test_chat_json_retries_transient_503_and_reports_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    monkeypatch.setattr(provider.time, "sleep", lambda _: None)
    attempts = 0

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        nonlocal attempts
        del timeout
        attempts += 1
        if attempts < 3:
            raise HTTPError(
                url=request.full_url,
                code=503,
                msg="Service Unavailable",
                hdrs=None,
                fp=FakeErrorBody(
                    "ResourceExhausted: Worker local total request limit reached"
                ),
            )
        return FakeResponse(
            {"choices": [{"message": {"content": '{"ok":true}'}}]}
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    client = provider.OpenAICompatibleLLMClient.from_env()
    assert client.chat_json([{"role": "user", "content": "query"}]) == {
        "ok": True
    }
    assert attempts == 3
    assert client.last_call_diagnostics is not None
    assert client.last_call_diagnostics.http_attempts == 3


def test_chat_json_retries_transient_timeout_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    monkeypatch.setattr(provider.time, "sleep", lambda _: None)
    attempts = 0

    def fake_urlopen(request, timeout: float):  # noqa: ANN001
        nonlocal attempts
        del request, timeout
        attempts += 1
        if attempts == 1:
            raise socket.timeout("upstream timeout")
        return FakeResponse(
            {"choices": [{"message": {"content": '{"ok":true}'}}]}
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    client = provider.OpenAICompatibleLLMClient.from_env()
    assert client.chat_json([{"role": "user", "content": "query"}]) == {
        "ok": True
    }
    assert attempts == 2
    assert client.last_call_diagnostics is not None
    assert client.last_call_diagnostics.http_attempts == 2


def test_chat_json_result_can_be_strictly_schema_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = provider.OpenAICompatibleLLMClient(
        base_url="https://api.example.test/v1",
        api_key="sk-test-secret",
        model="gpt-test",
    )

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        return FakeResponse(
            {"choices": [{"message": {"content": '{"status":"ok"}'}}]},
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    parsed = client.chat_json([{"role": "user", "content": "query"}])

    assert StrictPreflightResult.model_validate(parsed).status == "ok"
    with pytest.raises(ValidationError):
        StrictPreflightResult.model_validate({**parsed, "unexpected": True})


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (
            "https://api.example.test/v1",
            "https://api.example.test/v1/chat/completions",
        ),
        (
            "https://api.example.test/v1/",
            "https://api.example.test/v1/chat/completions",
        ),
        (
            "https://api.example.test/v1/chat/completions",
            "https://api.example.test/v1/chat/completions",
        ),
    ],
)
def test_chat_completions_url_is_joined_once(base_url: str, expected: str) -> None:
    assert provider._chat_completions_url(base_url) == expected


def test_chat_json_url_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_enabled_env(monkeypatch)

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        raise URLError("network down")

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    with pytest.raises(provider.LLMProviderError, match="llm_url_error:network down"):
        provider.chat_json([{"role": "user", "content": "query"}])


def test_chat_json_rejects_non_json_content(monkeypatch: pytest.MonkeyPatch) -> None:
    client = provider.OpenAICompatibleLLMClient(
        base_url="https://api.example.test/v1",
        api_key="sk-test-secret",
        model="gpt-test",
    )

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        return FakeResponse(
            {"choices": [{"message": {"content": "not json"}}]},
        )

    monkeypatch.setattr(provider, "urlopen", fake_urlopen)

    with pytest.raises(provider.LLMResponseError, match="llm_invalid_json_content"):
        client.chat_json([{"role": "user", "content": "query"}])


class FakeErrorBody:
    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        self._body = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": code,
                    "param": param,
                }
            }
        ).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        return None


class StrictPreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in (
        provider.PROVIDER_ENV,
        provider.BASE_URL_ENV,
        provider.API_KEY_ENV,
        provider.MODEL_ENV,
        provider.TIMEOUT_ENV,
        provider.MAX_TOKENS_ENV,
        provider.NVIDIA_THINKING_ENV,
    ):
        monkeypatch.delenv(env_name, raising=False)


def _set_enabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv(provider.PROVIDER_ENV, provider.SUPPORTED_PROVIDER)
    monkeypatch.setenv(provider.BASE_URL_ENV, "https://api.example.test/v1")
    monkeypatch.setenv(provider.API_KEY_ENV, "sk-test-secret")
    monkeypatch.setenv(provider.MODEL_ENV, "gpt-test")
