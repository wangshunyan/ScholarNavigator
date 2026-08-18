"""Shared runtime accounting for SearchService execution budgets."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scholar_agent.core.search_schemas import (
    BudgetStatus,
    CandidateTruncation,
    SearchBudget,
)


class BudgetStopError(RuntimeError):
    """Raised internally when an LLM request must not be started."""


@dataclass(frozen=True)
class TokenUsageSnapshot:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    available: bool = False


class SearchBudgetRuntime:
    """Mutable, per-search counters over an immutable validated budget."""

    def __init__(
        self,
        budget: SearchBudget | None = None,
        *,
        elapsed_seconds_provider: Callable[[], float] | None = None,
        resource_accounting_observer: Any | None = None,
    ) -> None:
        self.budget = budget or SearchBudget()
        self.started_at = time.perf_counter()
        self.stop_reasons: list[str] = []
        self.diagnostics: list[str] = []
        self.completed_search_rounds = 0
        self.used_llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.token_usage_precise = True
        self.candidate_truncations: list[CandidateTruncation] = []
        self._elapsed_seconds_provider = elapsed_seconds_provider
        self._resource_accounting_observer = resource_accounting_observer

    @property
    def elapsed_seconds(self) -> float:
        actual = max(0.0, time.perf_counter() - self.started_at)
        if self._elapsed_seconds_provider is None:
            return actual
        return max(actual, max(0.0, self._elapsed_seconds_provider()))

    def stop(self, reason: str) -> str:
        if reason not in self.stop_reasons:
            self.stop_reasons.append(reason)
        return reason

    def diagnose(self, diagnostic: str) -> None:
        if diagnostic not in self.diagnostics:
            self.diagnostics.append(diagnostic)

    def latency_stop_reason(self) -> str | None:
        if self.elapsed_seconds < self.budget.max_latency_seconds:
            return None
        return self.stop("budget_stop:max_latency_seconds")

    def candidate_stop_reason(self, candidate_count: int) -> str | None:
        if candidate_count < self.budget.max_candidate_papers:
            return None
        return self.stop("budget_stop:max_candidate_papers")

    def record_search_round(self) -> None:
        self.completed_search_rounds += 1
        self._observe_budget(
            "search_round_consumed",
            {"completed_search_rounds": self.completed_search_rounds},
        )

    def record_candidate_count(self, candidate_count: int) -> None:
        """Expose the already enforced candidate count to an optional observer."""

        self._observe_budget(
            "candidate_budget_observed",
            {"candidate_count": max(0, int(candidate_count))},
        )

    def can_start_llm(self) -> str | None:
        latency_reason = self.latency_stop_reason()
        if latency_reason is not None:
            return latency_reason
        if self.used_llm_calls >= self.budget.max_llm_calls:
            return self.stop("budget_stop:max_llm_calls")
        if self.total_tokens >= self.budget.max_total_tokens:
            return self.stop("budget_stop:max_total_tokens")
        return None

    def begin_llm_call(self) -> int:
        self.used_llm_calls += 1
        return self.used_llm_calls

    def finish_llm_call(
        self,
        before: TokenUsageSnapshot,
        after: TokenUsageSnapshot,
        *,
        sequence: int | None = None,
        terminal_status: str = "success",
    ) -> None:
        prompt_delta = max(0, after.prompt_tokens - before.prompt_tokens)
        completion_delta = max(0, after.completion_tokens - before.completion_tokens)
        total_delta = max(0, after.total_tokens - before.total_tokens)
        self.prompt_tokens += prompt_delta
        self.completion_tokens += completion_delta
        self.total_tokens += total_delta
        if not before.available or not after.available or total_delta == 0:
            self.token_usage_precise = False
            self.diagnose("budget_diagnostic:llm_usage_unavailable")
        if self.total_tokens >= self.budget.max_total_tokens:
            self.stop("budget_stop:max_total_tokens")
        observer = self._resource_accounting_observer
        callback = getattr(observer, "observe_llm_call", None)
        if callable(callback):
            try:
                callback(
                    {
                        "sequence": sequence or self.used_llm_calls,
                        "prompt_tokens": prompt_delta,
                        "completion_tokens": completion_delta,
                        "total_tokens": total_delta,
                        "usage_available": bool(
                            before.available and after.available
                        ),
                        "terminal_status": terminal_status,
                    }
                )
            except Exception:  # noqa: BLE001 - observation must not alter execution
                pass

    def record_candidate_truncation(
        self,
        *,
        stage: str,
        before_count: int,
        after_count: int,
    ) -> None:
        if after_count >= before_count:
            return
        self.candidate_truncations.append(
            CandidateTruncation(
                stage=stage,
                before_count=before_count,
                after_count=after_count,
                truncated_count=before_count - after_count,
            )
        )
        self.stop("budget_stop:max_candidate_papers")

    def status(self) -> BudgetStatus:
        budget = self.budget
        return BudgetStatus(
            exhausted=bool(self.stop_reasons),
            stop_reasons=list(self.stop_reasons),
            diagnostics=list(self.diagnostics),
            max_search_rounds=budget.max_search_rounds,
            completed_search_rounds=self.completed_search_rounds,
            max_candidate_papers=budget.max_candidate_papers,
            candidate_limit_applied=bool(self.candidate_truncations),
            candidate_truncations=list(self.candidate_truncations),
            max_llm_calls=budget.max_llm_calls,
            used_llm_calls=self.used_llm_calls,
            max_total_tokens=budget.max_total_tokens,
            used_total_tokens=self.total_tokens,
            token_usage_precise=self.token_usage_precise,
            max_latency_seconds=budget.max_latency_seconds,
            elapsed_seconds=self.elapsed_seconds,
        )

    def finalize_resource_accounting(self, candidate_count: int) -> None:
        """Publish final existing counters without changing budget decisions."""

        self._observe_budget(
            "budget_finalized",
            {
                "completed_search_rounds": self.completed_search_rounds,
                "candidate_count": max(0, int(candidate_count)),
                "elapsed_seconds": self.elapsed_seconds,
                "stop_reasons": list(self.stop_reasons),
            },
        )

    def _observe_budget(self, name: str, payload: dict[str, Any]) -> None:
        observer = self._resource_accounting_observer
        callback = getattr(observer, "observe_budget_event", None)
        if not callable(callback):
            return
        try:
            callback(name, payload)
        except Exception:  # noqa: BLE001 - observation must not alter execution
            pass


class BudgetedLLMClient:
    """LLM client proxy that blocks calls before global limits are exceeded."""

    def __init__(self, client: Any, runtime: SearchBudgetRuntime) -> None:
        self._client = client
        self._runtime = runtime

    @property
    def token_usage(self) -> Any:
        return getattr(self._client, "token_usage", None)

    @property
    def model(self) -> str | None:
        value = getattr(self._client, "model", None)
        return str(value) if value else None

    @property
    def provider(self) -> str | None:
        value = getattr(self._client, "provider", None)
        return str(value) if value else None

    @property
    def base_url_host(self) -> str | None:
        value = getattr(self._client, "base_url_host", None)
        return str(value) if value else None

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        reason = self._runtime.can_start_llm()
        if reason is not None:
            raise BudgetStopError(reason)
        before = token_usage_snapshot(self._client)
        sequence = self._runtime.begin_llm_call()
        terminal_status = "success"
        try:
            return self._client.chat_json(
                messages,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception:  # noqa: BLE001 - preserve provider exception and account it
            terminal_status = "failed"
            raise
        finally:
            self._runtime.finish_llm_call(
                before,
                token_usage_snapshot(self._client),
                sequence=sequence,
                terminal_status=terminal_status,
            )


def token_usage_snapshot(client: Any | None) -> TokenUsageSnapshot:
    usage = getattr(client, "token_usage", None)
    if usage is None:
        return TokenUsageSnapshot()
    return TokenUsageSnapshot(
        prompt_tokens=_usage_count(usage, "prompt_tokens"),
        completion_tokens=_usage_count(usage, "completion_tokens"),
        total_tokens=_usage_count(usage, "total_tokens"),
        available=True,
    )


def _usage_count(usage: Any, key: str) -> int:
    raw_value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, 0)
    try:
        count = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return count if count > 0 else 0
