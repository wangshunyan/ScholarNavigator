"""OpenAI-compatible LLM provider utilities."""

from __future__ import annotations

import json
import os
import random
import re
import socket
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import BoundedSemaphore, RLock
from typing import Any
from email.utils import parsedate_to_datetime
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
# A logical planning call has a strict two-attempt transport budget. 429 is
# transient, but repeated rate-limit responses must fail fast and let the
# planner's controlled fallback handle the query.
TRANSIENT_RETRY_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_HTTP_ATTEMPTS_PER_CALL = 2
TRANSIENT_RETRY_DELAYS_SECONDS = (1.0,)
LLM_GLOBAL_CONCURRENCY_LIMIT = 1
_PROVIDER_REQUEST_SLOT = BoundedSemaphore(LLM_GLOBAL_CONCURRENCY_LIMIT)
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
    failure_class: str | None = None
    retry_after_seconds: float | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "http_status": self.http_status,
            "error_type": self.error_type,
            "service_error_code": self.service_error_code,
            "summary": self.summary,
            "unsupported_parameters": list(self.unsupported_parameters),
            "failure_class": self.failure_class,
            "retry_after_seconds": self.retry_after_seconds,
        }


@dataclass(frozen=True)
class LLMCallDiagnostics:
    """最近一次逻辑调用的公开诊断。"""

    mode: str
    http_attempts: int
    latency_ms: int
    fallback_reason: str | None = None
    http_429_count: int = 0
    retry_after_seconds: tuple[float, ...] = ()
    retry_wait_seconds: float = 0.0
    failure_class: str | None = None
    cache_hit: bool = False

    def model_dump(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "http_attempts": self.http_attempts,
            "latency_ms": self.latency_ms,
            "fallback_reason": self.fallback_reason,
            "http_429_count": self.http_429_count,
            "retry_after_seconds": list(self.retry_after_seconds),
            "retry_wait_seconds": self.retry_wait_seconds,
            "failure_class": self.failure_class,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True)
class _TransportResult:
    response: dict[str, Any]
    http_attempts: int
    http_429_count: int
    retry_after_seconds: tuple[float, ...]
    retry_wait_seconds: float
    failure_class: str | None = None


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
    _successful_response_cache: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _response_cache_lock: RLock = field(default_factory=RLock, init=False, repr=False)

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

        cache_key = _response_cache_key(payload)
        cached_response = self._load_cached_response(cache_key)
        if cached_response is not None:
            self.last_call_diagnostics = LLMCallDiagnostics(
                mode=mode,
                http_attempts=0,
                latency_ms=max(0, round((time.monotonic() - started) * 1000)),
                cache_hit=True,
            )
            return _response_content_object(cached_response)

        transport: _TransportResult | None = None
        fallback_reason: str | None = None
        try:
            transport = _coerce_transport_result(
                self._send_with_retries(
                    payload,
                    timeout=timeout_seconds,
                )
            )
        except LLMProviderError as exc:
            self.last_call_diagnostics = _failed_call_diagnostics(
                exc,
                started=started,
            )
            fallback = self._compatibility_fallback(payload, exc)
            if fallback is None:
                raise
            payload, mode, fallback_reason = fallback
            remaining_attempts = (
                MAX_HTTP_ATTEMPTS_PER_CALL
                - self.last_call_diagnostics.http_attempts
            )
            if remaining_attempts <= 0:
                raise exc
            try:
                fallback_transport = _coerce_transport_result(
                    self._send_with_retries(
                        payload,
                        timeout=timeout_seconds,
                        max_attempts=remaining_attempts,
                    )
                )
            except LLMProviderError as fallback_error:
                self.last_call_diagnostics = _failed_call_diagnostics(
                    fallback_error,
                    started=started,
                    previous=self.last_call_diagnostics,
                )
                raise
            transport = _combine_transport_results(
                _transport_from_diagnostics(self.last_call_diagnostics),
                fallback_transport,
            )

        if transport is None:
            raise AssertionError("provider transport result unavailable")
        parsed_response = transport.response

        self.last_call_diagnostics = LLMCallDiagnostics(
            mode=mode,
            http_attempts=transport.http_attempts,
            latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            fallback_reason=fallback_reason,
            http_429_count=transport.http_429_count,
            retry_after_seconds=transport.retry_after_seconds,
            retry_wait_seconds=transport.retry_wait_seconds,
            failure_class=transport.failure_class,
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
            content = _response_content_object(parsed_response)
        except LLMResponseError:
            raise
        self._store_cached_response(cache_key, parsed_response)
        return content

    def _load_cached_response(self, cache_key: str) -> dict[str, Any] | None:
        with self._response_cache_lock:
            value = self._successful_response_cache.get(cache_key)
            return dict(value) if value is not None else None

    def _store_cached_response(self, cache_key: str, response: dict[str, Any]) -> None:
        cached = dict(response)
        cached.pop("usage", None)
        with self._response_cache_lock:
            self._successful_response_cache[cache_key] = cached

    def _send_with_retries(
        self,
        payload: dict[str, Any],
        *,
        timeout: float,
        max_attempts: int = MAX_HTTP_ATTEMPTS_PER_CALL,
    ) -> _TransportResult:
        """Retry only provider/network failures that are safe to repeat.

        The request is a deterministic, non-streaming JSON planning call. A
        bounded retry is appropriate for transient 5xx responses and socket
        timeouts, while schema/4xx errors must still surface immediately.
        """

        if max_attempts < 1 or max_attempts > MAX_HTTP_ATTEMPTS_PER_CALL:
            raise ValueError("max_attempts_out_of_range")
        attempts = 0
        http_429_count = 0
        retry_after_values: list[float] = []
        retry_wait_seconds = 0.0
        previous_error: LLMProviderError | None = None
        for attempt_index in range(max_attempts):
            if previous_error is not None:
                delay = _retry_delay_seconds(previous_error, attempt_index, self.base_url)
                retry_wait_seconds += delay
                retry_after = previous_error.details.retry_after_seconds
                if retry_after is not None:
                    retry_after_values.append(retry_after)
                time.sleep(delay)
            attempts += 1
            try:
                response = self._send_request(payload, timeout=timeout)
                return _TransportResult(
                    response=response,
                    http_attempts=attempts,
                    http_429_count=http_429_count,
                    retry_after_seconds=tuple(retry_after_values),
                    retry_wait_seconds=retry_wait_seconds,
                    failure_class=(
                        previous_error.details.failure_class
                        if previous_error is not None
                        else None
                    ),
                )
            except LLMProviderError as exc:
                if exc.details.http_status == 429:
                    http_429_count += 1
                if not _is_transient_error(exc) or attempt_index + 1 >= max_attempts:
                    setattr(exc, "transport_diagnostics", LLMCallDiagnostics(
                        mode="structured_json",
                        http_attempts=attempts,
                        latency_ms=0,
                        http_429_count=http_429_count,
                        retry_after_seconds=tuple(retry_after_values),
                        retry_wait_seconds=retry_wait_seconds,
                        failure_class=exc.details.failure_class,
                    ))
                    raise
                previous_error = exc
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
            with _PROVIDER_REQUEST_SLOT:
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
                    failure_class="timeout",
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
                    failure_class="network_error",
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


def _response_cache_key(payload: dict[str, Any]) -> str:
    """Hash a logical JSON request without persisting its query text."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _response_content_object(parsed_response: dict[str, Any]) -> dict[str, Any]:
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
    retry_after_seconds = _retry_after_seconds(error.headers)
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
        failure_class=_provider_failure_class(
            http_status=error.code,
            error_type=error_type,
            service_error_code=service_error_code,
            summary=summary,
        ),
        retry_after_seconds=retry_after_seconds,
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
    if error.details.failure_class == "insufficient_quota":
        return False
    if isinstance(error, LLMTimeoutError):
        return True
    return error.details.http_status in TRANSIENT_RETRY_STATUS_CODES


def _provider_failure_class(
    *,
    http_status: int | None,
    error_type: str | None,
    service_error_code: str | None,
    summary: str | None,
) -> str:
    text = " ".join(
        value.casefold()
        for value in (error_type, service_error_code, summary)
        if value
    )
    if any(marker in text for marker in ("insufficient_quota", "insufficient quota", "quota exceeded", "billing")):
        return "insufficient_quota"
    if http_status == 429:
        return "rate_limit"
    if http_status in {500, 502, 503, 504} or "overload" in text or "resourceexhausted" in text:
        return "temporary_overload"
    if http_status is not None and 400 <= http_status < 500:
        return "client_error"
    return "provider_error"


def _retry_after_seconds(headers: Any) -> float | None:
    getter = getattr(headers, "get", None)
    raw = getter("Retry-After") if callable(getter) else None
    if raw is None:
        return None
    text = str(raw).strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            target = parsedate_to_datetime(text)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            seconds = (target - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
    return min(60.0, max(0.0, seconds))


def _retry_delay_seconds(
    error: LLMProviderError,
    attempt_index: int,
    base_url: str,
) -> float:
    if _is_loopback_provider(base_url):
        return TRANSIENT_RETRY_DELAYS_SECONDS[0]
    retry_after = error.details.retry_after_seconds
    if retry_after is not None:
        return retry_after
    base = min(8.0, float(2 ** max(0, attempt_index - 1)))
    return base + random.uniform(0.0, min(1.0, base * 0.25))


def _is_loopback_provider(base_url: str) -> bool:
    host = (_base_url_host(base_url) or "").split(":", 1)[0].casefold()
    return host in {"127.0.0.1", "localhost", "::1"}


def _failed_call_diagnostics(
    error: LLMProviderError,
    *,
    started: float,
    previous: LLMCallDiagnostics | None = None,
) -> LLMCallDiagnostics:
    transport = getattr(error, "transport_diagnostics", None)
    attempts = int(getattr(transport, "http_attempts", 1))
    http_429_count = int(getattr(transport, "http_429_count", 0))
    retry_after = tuple(getattr(transport, "retry_after_seconds", ()))
    retry_wait = float(getattr(transport, "retry_wait_seconds", 0.0))
    if previous is not None:
        attempts += previous.http_attempts
        http_429_count += previous.http_429_count
        retry_after = (*previous.retry_after_seconds, *retry_after)
        retry_wait += previous.retry_wait_seconds
    return LLMCallDiagnostics(
        mode="structured_json",
        http_attempts=attempts,
        latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        http_429_count=http_429_count,
        retry_after_seconds=retry_after,
        retry_wait_seconds=retry_wait,
        failure_class=error.details.failure_class,
    )


def _transport_from_diagnostics(diagnostics: LLMCallDiagnostics | None) -> _TransportResult:
    if diagnostics is None:
        raise AssertionError("transport diagnostics unavailable")
    return _TransportResult(
        response={},
        http_attempts=diagnostics.http_attempts,
        http_429_count=diagnostics.http_429_count,
        retry_after_seconds=diagnostics.retry_after_seconds,
        retry_wait_seconds=diagnostics.retry_wait_seconds,
        failure_class=diagnostics.failure_class,
    )


def _combine_transport_results(
    first: _TransportResult,
    second: _TransportResult,
) -> _TransportResult:
    return _TransportResult(
        response=second.response,
        http_attempts=first.http_attempts + second.http_attempts,
        http_429_count=first.http_429_count + second.http_429_count,
        retry_after_seconds=(*first.retry_after_seconds, *second.retry_after_seconds),
        retry_wait_seconds=first.retry_wait_seconds + second.retry_wait_seconds,
        failure_class=second.failure_class or first.failure_class,
    )


def _coerce_transport_result(value: Any) -> _TransportResult:
    """Keep test/integration seams that supplied the previous tuple shape."""

    if isinstance(value, _TransportResult):
        return value
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], dict)
    ):
        return _TransportResult(
            response=value[0],
            http_attempts=max(0, int(value[1])),
            http_429_count=0,
            retry_after_seconds=(),
            retry_wait_seconds=0.0,
        )
    raise TypeError("invalid_transport_result")


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
