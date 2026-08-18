"""OpenAI-compatible LLM provider utilities."""

from __future__ import annotations

import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROVIDER_ENV = "SCHOLAR_AGENT_LLM_PROVIDER"
BASE_URL_ENV = "SCHOLAR_AGENT_LLM_BASE_URL"
API_KEY_ENV = "SCHOLAR_AGENT_LLM_API_KEY"
MODEL_ENV = "SCHOLAR_AGENT_LLM_MODEL"
TIMEOUT_ENV = "SCHOLAR_AGENT_LLM_TIMEOUT_SECONDS"
MAX_TOKENS_ENV = "SCHOLAR_AGENT_LLM_MAX_TOKENS"
NVIDIA_THINKING_ENV = "SCHOLAR_AGENT_LLM_NVIDIA_THINKING"

SUPPORTED_PROVIDER = "openai_compatible"
DISABLED_PROVIDER = "disabled"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 1024
TRANSIENT_RETRY_STATUS_CODES = frozenset({408, 425, 500, 502, 503, 504})
TRANSIENT_RETRY_DELAYS_SECONDS = (1.0, 3.0)
JSON_ONLY_COMPATIBILITY_INSTRUCTION = (
    "Compatibility requirement: return exactly one valid JSON object and no "
    "markdown, prose, or code fences."
)
_OPTIONAL_JSON_PARAMETERS = ("response_format", "chat_template_kwargs")
_UNSUPPORTED_PARAMETER_MARKERS = (
    "unsupported parameter",
    "unsupported field",
    "unknown parameter",
    "unknown field",
    "unrecognized parameter",
    "unrecognized field",
    "not supported",
    "not permitted",
)


@dataclass(frozen=True)
class LLMErrorDetails:
    """脱敏后的 provider 诊断信息，不包含响应正文或请求凭据。"""

    http_status: int | None = None
    error_type: str | None = None
    service_error_code: str | None = None
    summary: str | None = None
    unsupported_parameters: tuple[str, ...] = ()

    def model_dump(self) -> dict[str, Any]:
        return {
            "http_status": self.http_status,
            "error_type": self.error_type,
            "service_error_code": self.service_error_code,
            "summary": self.summary,
            "unsupported_parameters": list(self.unsupported_parameters),
        }


@dataclass(frozen=True)
class LLMCallDiagnostics:
    """最近一次逻辑调用的公开诊断。"""

    mode: str
    http_attempts: int
    latency_ms: int
    fallback_reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "http_attempts": self.http_attempts,
            "latency_ms": self.latency_ms,
            "fallback_reason": self.fallback_reason,
        }


class LLMProviderError(RuntimeError):
    """Base error for LLM provider failures."""

    def __init__(
        self,
        message: str,
        *,
        details: LLMErrorDetails | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details or LLMErrorDetails(summary=message)


class LLMConfigurationError(LLMProviderError):
    """Raised when the configured provider is invalid or incomplete."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM request times out."""


class LLMResponseError(LLMProviderError):
    """Raised when an LLM response cannot be parsed as a JSON object."""


@dataclass
class LLMTokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: "LLMTokenUsage") -> None:
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens

    def model_dump(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMRuntimeConfig:
    provider: str
    model: str | None
    available: bool
    base_url_host: str | None = None
    reason: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.available,
            "base_url_host": self.base_url_host,
            "reason": self.reason,
        }


def is_llm_enabled() -> bool:
    return get_llm_runtime_config().available


def get_llm_runtime_config() -> LLMRuntimeConfig:
    provider = _provider_name()
    if provider == DISABLED_PROVIDER:
        return LLMRuntimeConfig(
            provider=DISABLED_PROVIDER,
            model=os.getenv(MODEL_ENV),
            available=False,
            base_url_host=_base_url_host(os.getenv(BASE_URL_ENV)),
            reason="provider_disabled",
        )

    if provider != SUPPORTED_PROVIDER:
        return LLMRuntimeConfig(
            provider=provider,
            model=os.getenv(MODEL_ENV),
            available=False,
            base_url_host=_base_url_host(os.getenv(BASE_URL_ENV)),
            reason="unsupported_provider",
        )

    base_url = os.getenv(BASE_URL_ENV, "").strip()
    api_key = os.getenv(API_KEY_ENV, "").strip()
    model = os.getenv(MODEL_ENV, "").strip()
    missing = []
    if not base_url:
        missing.append(BASE_URL_ENV)
    if not api_key:
        missing.append(API_KEY_ENV)
    if not model:
        missing.append(MODEL_ENV)
    if missing:
        return LLMRuntimeConfig(
            provider=SUPPORTED_PROVIDER,
            model=model or None,
            available=False,
            base_url_host=_base_url_host(base_url),
            reason="missing_env:" + ",".join(missing),
        )

    return LLMRuntimeConfig(
        provider=SUPPORTED_PROVIDER,
        model=model,
        available=True,
        base_url_host=_base_url_host(base_url),
    )


def get_llm_request_options() -> dict[str, int | float]:
    """返回可进入快照键的公开请求参数，不包含密钥或完整 URL。"""

    return {
        "timeout_seconds": _timeout_from_env(),
        "max_tokens": _max_tokens_from_env(),
    }


def chat_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0,
    timeout: float | None = None,
) -> dict[str, Any]:
    return OpenAICompatibleLLMClient.from_env().chat_json(
        messages,
        temperature=temperature,
        timeout=timeout,
    )


@dataclass
class OpenAICompatibleLLMClient:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    token_usage: LLMTokenUsage = field(default_factory=LLMTokenUsage)
    last_call_usage: LLMTokenUsage | None = field(default=None, init=False)
    last_call_usage_fields: dict[str, int] | None = field(
        default=None,
        init=False,
    )
    last_call_diagnostics: LLMCallDiagnostics | None = field(
        default=None,
        init=False,
    )
    _json_only_compatibility: bool = field(default=False, init=False, repr=False)
    _omit_thinking_parameter: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLMClient":
        config = get_llm_runtime_config()
        if not config.available:
            raise LLMConfigurationError(config.reason or "llm_disabled")
        return cls(
            base_url=os.environ[BASE_URL_ENV].strip(),
            api_key=os.environ[API_KEY_ENV].strip(),
            model=os.environ[MODEL_ENV].strip(),
            timeout_seconds=_timeout_from_env(),
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        # Per-call diagnostics must never leak forward from an earlier request.
        # This matters to resumable audit callers that persist supplier usage
        # and transport attempts even when the current request fails.
        self.last_call_usage = None
        self.last_call_usage_fields = None
        self.last_call_diagnostics = None
        started = time.monotonic()
        timeout_seconds = timeout if timeout is not None else self.timeout_seconds
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _json_only_messages(messages)
            if self._json_only_compatibility
            else messages,
            "temperature": temperature,
            "max_tokens": _max_tokens_from_env(),
            "stream": False,
        }
        mode = "json_only_prompt" if self._json_only_compatibility else "structured_json"
        if not self._json_only_compatibility:
            payload["response_format"] = {"type": "json_object"}
        if not _nvidia_thinking_from_env() and not self._omit_thinking_parameter:
            # ``extra_body`` is an OpenAI SDK option, not a wire-format field.
            # OpenAI-compatible raw HTTP endpoints expect the extension itself.
            payload["chat_template_kwargs"] = {"thinking": False}

        http_attempts = 1
        fallback_reason: str | None = None
        try:
            parsed_response, http_attempts = self._send_with_retries(
                payload,
                timeout=timeout_seconds,
            )
        except LLMProviderError as exc:
            fallback = self._compatibility_fallback(payload, exc)
            if fallback is None:
                raise
            payload, mode, fallback_reason = fallback
            parsed_response, fallback_attempts = self._send_with_retries(
                payload,
                timeout=timeout_seconds,
            )
            http_attempts += fallback_attempts

        self.last_call_diagnostics = LLMCallDiagnostics(
            mode=mode,
            http_attempts=http_attempts,
            latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            fallback_reason=fallback_reason,
        )
        self.last_call_usage_fields = _parse_token_usage_fields(
            parsed_response.get("usage")
        )
        self.last_call_usage = _token_usage_from_fields(
            self.last_call_usage_fields
        )
        if self.last_call_usage is not None:
            self.token_usage.add(self.last_call_usage)

        try:
            content = parsed_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("llm_malformed_chat_response") from exc

        try:
            parsed_content = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMResponseError("llm_invalid_json_content") from exc
        if not isinstance(parsed_content, dict):
            raise LLMResponseError("llm_json_content_not_object")
        return parsed_content

    def _send_with_retries(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> tuple[dict[str, Any], int]:
        """Retry only provider/network failures that are safe to repeat.

        The request is a deterministic, non-streaming JSON planning call. A
        bounded retry is appropriate for transient 5xx responses and socket
        timeouts, while schema/4xx errors must still surface immediately.
        """

        attempts = 0
        for retry_index, delay in enumerate((0.0, *TRANSIENT_RETRY_DELAYS_SECONDS)):
            if delay:
                time.sleep(delay)
            attempts += 1
            try:
                return self._send_request(payload, timeout=timeout), attempts
            except LLMProviderError as exc:
                if not _is_transient_error(exc) or retry_index >= len(
                    TRANSIENT_RETRY_DELAYS_SECONDS
                ):
                    raise
        raise AssertionError("transient retry loop did not return or raise")

    def _send_request(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        request = Request(
            _chat_completions_url(self.base_url),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(  # noqa: S310 - URL is configured by trusted backend env.
                request,
                timeout=timeout,
            ) as response:
                response_body = response.read().decode("utf-8")
        except socket.timeout as exc:
            raise LLMTimeoutError(
                "llm_request_timeout",
                details=LLMErrorDetails(
                    error_type=type(exc).__name__,
                    summary="request timed out",
                ),
            ) from exc
        except HTTPError as exc:
            raise _http_provider_error(exc) from exc
        except URLError as exc:
            summary = _sanitize_error_message(str(exc.reason))
            raise LLMProviderError(
                f"llm_url_error:{summary}",
                details=LLMErrorDetails(
                    error_type=type(exc.reason).__name__,
                    summary=summary,
                ),
            ) from exc

        try:
            parsed_response = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("llm_malformed_chat_response") from exc
        if not isinstance(parsed_response, dict):
            raise LLMResponseError("llm_malformed_chat_response")
        return parsed_response

    def _compatibility_fallback(
        self,
        payload: dict[str, Any],
        error: LLMProviderError,
    ) -> tuple[dict[str, Any], str, str] | None:
        details = error.details
        unsupported = set(details.unsupported_parameters)
        if details.http_status not in {400, 422} or not unsupported:
            return None
        if not unsupported.intersection(_OPTIONAL_JSON_PARAMETERS):
            return None

        fallback_payload = dict(payload)
        fallback_payload["messages"] = [dict(item) for item in payload["messages"]]
        fallback_reason = "unsupported_parameters:" + ",".join(
            parameter
            for parameter in _OPTIONAL_JSON_PARAMETERS
            if parameter in unsupported
        )

        if "chat_template_kwargs" in unsupported:
            fallback_payload.pop("chat_template_kwargs", None)
            self._omit_thinking_parameter = True
        if "response_format" in unsupported:
            fallback_payload.pop("response_format", None)
            fallback_payload["messages"] = _json_only_messages(
                fallback_payload["messages"]
            )
            self._json_only_compatibility = True
            mode = "json_only_prompt"
        else:
            mode = "structured_without_optional_parameters"
        return fallback_payload, mode, fallback_reason


def _provider_name() -> str:
    raw_provider = os.getenv(PROVIDER_ENV)
    if raw_provider is None or not raw_provider.strip():
        return DISABLED_PROVIDER
    return raw_provider.strip().lower()


def _timeout_from_env() -> float:
    raw_value = os.getenv(TIMEOUT_ENV)
    if raw_value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw_value)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return timeout if timeout > 0 else DEFAULT_TIMEOUT_SECONDS


def _max_tokens_from_env() -> int:
    raw_value = os.getenv(MAX_TOKENS_ENV)
    if raw_value is None:
        return DEFAULT_MAX_TOKENS
    try:
        max_tokens = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_TOKENS
    return max_tokens if max_tokens >= 1 else DEFAULT_MAX_TOKENS


def _nvidia_thinking_from_env() -> bool:
    raw_value = os.getenv(NVIDIA_THINKING_ENV)
    if raw_value is None:
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_token_usage(raw_usage: object) -> LLMTokenUsage:
    return _parse_token_usage_optional(raw_usage) or LLMTokenUsage()


def _parse_token_usage_optional(raw_usage: object) -> LLMTokenUsage | None:
    return _token_usage_from_fields(_parse_token_usage_fields(raw_usage))


def _parse_token_usage_fields(raw_usage: object) -> dict[str, int] | None:
    if not isinstance(raw_usage, dict):
        return None
    fields = {
        name: _token_count(raw_usage[name])
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if name in raw_usage
    }
    return fields or None


def _token_usage_from_fields(
    fields: dict[str, int] | None,
) -> LLMTokenUsage | None:
    if fields is None:
        return None
    return LLMTokenUsage(**fields)


def _token_count(value: object) -> int:
    try:
        count = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return count if count > 0 else 0


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _base_url_host(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    return parsed.netloc or None


def _http_provider_error(error: HTTPError) -> LLMProviderError:
    error_type: str | None = None
    service_error_code: str | None = None
    summary = _sanitize_error_message(str(error.reason))
    unsupported_parameters: tuple[str, ...] = ()
    try:
        error_body = error.read().decode("utf-8", errors="replace")
        parsed = json.loads(error_body)
        raw_error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(raw_error, dict):
            error_type = _safe_error_identifier(raw_error.get("type"))
            service_error_code = _safe_error_identifier(raw_error.get("code"))
            detail = raw_error.get("message")
            if detail:
                summary = _sanitize_error_message(str(detail))
            parameter = _safe_error_identifier(raw_error.get("param"))
            unsupported_parameters = _extract_unsupported_parameters(
                summary,
                parameter=parameter,
            )
        elif raw_error:
            summary = _sanitize_error_message(str(raw_error))
            unsupported_parameters = _extract_unsupported_parameters(summary)
    except Exception:
        # Provider bodies are untrusted and optional. The HTTP status remains useful.
        pass

    details = LLMErrorDetails(
        http_status=error.code,
        error_type=error_type,
        service_error_code=service_error_code,
        summary=summary,
        unsupported_parameters=unsupported_parameters,
    )
    message_parts = [f"llm_http_error:{error.code}"]
    if error_type:
        message_parts.append(f"type={error_type}")
    if service_error_code:
        message_parts.append(f"code={service_error_code}")
    if summary:
        message_parts.append(f"summary={summary}")
    return LLMProviderError(":".join(message_parts), details=details)


def _is_transient_error(error: LLMProviderError) -> bool:
    if isinstance(error, LLMTimeoutError):
        return True
    return error.details.http_status in TRANSIENT_RETRY_STATUS_CODES


def _extract_unsupported_parameters(
    message: str,
    *,
    parameter: str | None = None,
) -> tuple[str, ...]:
    normalized = message.casefold()
    if not any(marker in normalized for marker in _UNSUPPORTED_PARAMETER_MARKERS):
        return ()
    matched = []
    for candidate in _OPTIONAL_JSON_PARAMETERS:
        if candidate.casefold() in normalized or candidate == parameter:
            matched.append(candidate)
    return tuple(matched)


def _json_only_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JSON_ONLY_COMPATIBILITY_INSTRUCTION},
        *(dict(message) for message in messages),
    ]


def _safe_error_identifier(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,80}", normalized):
        return None
    return normalized


def _sanitize_error_message(message: str) -> str:
    api_key = os.getenv(API_KEY_ENV, "")
    sanitized = re.sub(r"[\r\n\t]+", " ", message).strip()
    sanitized = re.sub(
        r"(?i)\b(authorization|api[_-]?key|access[_-]?token)"
        r"(\s*[:=]\s*)(?:Bearer\s+)?[^\s,;]+",
        r"\1\2[redacted]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*",
        "Bearer [redacted]",
        sanitized,
    )
    if api_key:
        sanitized = sanitized.replace(api_key, "[redacted]")
    return sanitized[:240]
