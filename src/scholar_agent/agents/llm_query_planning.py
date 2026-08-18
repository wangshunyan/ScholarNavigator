"""受限、可校验的 LLM 语义初始查询规划。"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_agent.agents.query_planning import identify_query_facets
from scholar_agent.core.search_schemas import (
    LLM_QUERY_PLANNING_SCHEMA_VERSION,
    LLMQueryPlanningOutput,
    LLMSemanticQuery,
    QueryAnalysis,
    QueryConstraint,
    QueryPlanningResult,
    SearchSubquery,
)
from scholar_agent.llm.provider import (
    DEFAULT_MAX_TOKENS,
    get_llm_request_options,
    get_llm_runtime_config,
)
from scholar_agent.prompts.loader import load_prompt, render_messages


LLM_QUERY_PLANNING_PROMPT = "llm_query_planning"
MAX_SUPPLEMENTAL_QUERIES = 2
MAX_QUERY_CHARACTERS = 200
MAX_QUERY_TERMS = 24
MIN_INFORMATION_RETENTION = 0.34


class LLMPlanningRequest(BaseModel):
    """快照键与外部调用共享的无敏感信息请求描述。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = LLM_QUERY_PLANNING_SCHEMA_VERSION
    query_planning_policy: Literal[
        "llm_semantic", "llm_constrained_rewrite"
    ] = "llm_semantic"
    provider: str
    model: str | None = None
    base_url_host: str | None = None
    prompt_name: str
    prompt_version: str
    prompt_hash: str
    input_payload: dict[str, Any]
    run_profile: str
    max_supplemental_queries: int = Field(ge=0, le=MAX_SUPPLEMENTAL_QUERIES)
    temperature: float = 0.0
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1)


class LLMPlanningExecution(BaseModel):
    """一次 live/record/replay 调用返回的公开诊断。"""

    raw_response: dict[str, Any]
    snapshot_key: str | None = None
    snapshot_status: str | None = None
    llm_call_attempted: bool = False
    replayed: bool = False
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    recorded_latency_seconds: float = Field(default=0.0, ge=0.0)


class LLMPlanningRuntime(Protocol):
    def identity(self) -> tuple[str, str | None, str | None] | None:
        ...

    def execute(
        self,
        request: LLMPlanningRequest,
        messages: list[dict[str, str]],
        client: Any | None,
        *,
        timeout: float,
    ) -> LLMPlanningExecution:
        ...


class LLMPlanningOutcome(BaseModel):
    subqueries: list[SearchSubquery]
    result: QueryPlanningResult
    warnings: list[str] = Field(default_factory=list)


def plan_llm_semantic(
    query_analysis: QueryAnalysis,
    *,
    current_subqueries: list[SearchSubquery],
    current_result: QueryPlanningResult,
    selected_sources: list[str],
    max_subqueries: int,
    run_profile: str,
    explicit_constraints: QueryConstraint | None,
    llm_client: Any | None,
    runtime: LLMPlanningRuntime | None = None,
) -> LLMPlanningOutcome:
    """保留原查询，最多接受两条经确定性校验的 LLM 补充查询。"""

    maximum = min(MAX_SUPPLEMENTAL_QUERIES, max(0, max_subqueries - 1))
    try:
        prompt = load_prompt(LLM_QUERY_PLANNING_PROMPT)
    except Exception:  # Prompt 错误必须安全回退，具体错误不进入公开输出。
        return _fallback(
            current_subqueries,
            current_result,
            reason="prompt_load_failed",
        )

    provider, model, base_url_host = _client_identity(llm_client)
    identity_provider = getattr(runtime, "identity", None)
    runtime_identity = identity_provider() if callable(identity_provider) else None
    if llm_client is None and runtime_identity is not None:
        provider, model, base_url_host = runtime_identity
    base_updates: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "prompt_name": prompt.name,
        "prompt_version": prompt.version,
        "prompt_hash": prompt.content_hash,
        "original_query_retained": True,
    }
    if maximum == 0:
        return _fallback(
            current_subqueries,
            current_result,
            reason="supplemental_query_budget_zero",
            updates=base_updates,
        )
    if llm_client is None and runtime is None:
        return _fallback(
            current_subqueries,
            current_result,
            reason="llm_unconfigured",
            updates=base_updates,
        )

    input_payload = _planning_input(
        query_analysis,
        explicit_constraints=explicit_constraints,
        run_profile=run_profile,
        maximum=maximum,
    )
    request_options = get_llm_request_options()
    request = LLMPlanningRequest(
        provider=provider,
        model=model,
        base_url_host=base_url_host,
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        prompt_hash=prompt.content_hash,
        input_payload=input_payload,
        run_profile=run_profile,
        max_supplemental_queries=maximum,
        temperature=0,
        max_tokens=request_options["max_tokens"],
    )
    messages = render_messages(prompt.name, input_payload)
    execution: LLMPlanningExecution | None = None
    try:
        execution = (
            runtime.execute(
                request,
                messages,
                llm_client,
                timeout=float(request_options["timeout_seconds"]),
            )
            if runtime is not None
            else _execute_live(
                request,
                messages,
                llm_client,
                timeout=float(request_options["timeout_seconds"]),
            )
        )
        output = LLMQueryPlanningOutput.model_validate(execution.raw_response)
    except ValidationError:
        return _fallback(
            current_subqueries,
            current_result,
            reason="invalid_schema",
            updates=_execution_updates(base_updates, execution),
        )
    except Exception as exc:  # noqa: BLE001 - optional planner never breaks search
        reason = _fallback_reason(exc)
        failure_updates = _execution_updates(base_updates, execution)
        diagnostics_provider = getattr(runtime, "failure_diagnostics", None)
        if callable(diagnostics_provider):
            failure_updates.update(diagnostics_provider())
        if (
            llm_client is not None
            and reason not in {"budget_exhausted", "snapshot_missing"}
        ):
            failure_updates["llm_call_attempted"] = True
        return _fallback(
            current_subqueries,
            current_result,
            reason=reason,
            updates=failure_updates,
        )

    accepted: list[LLMSemanticQuery] = []
    rejection_reasons: Counter[str] = Counter()
    seen = {_query_key(query_analysis.original_query)}
    for candidate in output.supplemental_queries:
        normalized = " ".join(candidate.query.split())
        reason = _validate_candidate(
            candidate.model_copy(update={"query": normalized}),
            output=output,
            analysis=query_analysis,
            explicit_constraints=explicit_constraints,
            seen=seen,
        )
        if reason is not None:
            rejection_reasons[reason] += 1
            continue
        accepted.append(candidate.model_copy(update={"query": normalized}))
        seen.add(_query_key(normalized))

    updates = {
        **_execution_updates(base_updates, execution),
        "output_valid": True,
        "generated_query_count": len(output.supplemental_queries),
        "accepted_query_count": len(accepted),
        "rejected_query_count": sum(rejection_reasons.values()),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "terminology_expansions": _dedupe(
            expansion
            for item in accepted
            for expansion in item.terminology_expansions
        ),
    }
    if not accepted:
        return _fallback(
            current_subqueries,
            current_result,
            reason="all_queries_rejected",
            updates=updates,
            warnings=[
                f"llm_semantic_rejected:{reason}:{count}"
                for reason, count in sorted(rejection_reasons.items())
            ],
        )

    original = current_subqueries[0].model_copy(
        update={
            "query": query_analysis.original_query,
            "purpose": "original_query",
            "priority": 1,
        }
    )
    subqueries = [original]
    for index, candidate in enumerate(accepted[:maximum], start=2):
        facet_types = _search_facet_types(candidate.covered_facets)
        subqueries.append(
            SearchSubquery(
                query=candidate.query,
                source_hints=selected_sources,
                priority=index,
                purpose=f"llm_semantic:{_safe_purpose(candidate.purpose)}",
                facet_types=facet_types,
                provenance=[
                    "llm_semantic",
                    f"prompt:{prompt.name}@{prompt.version}",
                    *[f"facet:{value}" for value in candidate.covered_facets],
                ],
            )
        )
    result = current_result.model_copy(
        update={
            **updates,
            "policy": "llm_semantic",
            "selected_subqueries": subqueries,
            "selected_subquery_count": len(subqueries),
            "accepted_queries": [item.query for item in accepted],
            "fallback_used": False,
            "fallback_reason": None,
        }
    )
    warnings = [
        *output.warnings,
        *[
            f"llm_semantic_rejected:{reason}:{count}"
            for reason, count in sorted(rejection_reasons.items())
        ],
    ]
    return LLMPlanningOutcome(
        subqueries=subqueries,
        result=result,
        warnings=_dedupe(warnings),
    )


def _planning_input(
    analysis: QueryAnalysis,
    *,
    explicit_constraints: QueryConstraint | None,
    run_profile: str,
    maximum: int,
) -> dict[str, Any]:
    facets = identify_query_facets(analysis)
    return {
        "original_query": analysis.original_query,
        "explicit_constraints": (
            explicit_constraints.model_dump(mode="json", exclude={"explicit_fields"})
            if explicit_constraints is not None
            else None
        ),
        "rule_analysis": {
            "intent": analysis.intent,
            "domain": analysis.domain,
            "constraints": analysis.constraints.model_dump(
                mode="json",
                exclude={"explicit_fields"},
            ),
            "facets": [
                {
                    "facet_type": facet.facet_type,
                    "terms": list(facet.terms),
                    "required": facet.required,
                }
                for facet in facets
            ],
        },
        "run_profile": run_profile,
        "max_supplemental_queries": maximum,
    }


def _execute_live(
    request: LLMPlanningRequest,
    messages: list[dict[str, str]],
    client: Any | None,
    *,
    timeout: float,
) -> LLMPlanningExecution:
    if client is None:
        raise RuntimeError("llm_unconfigured")
    before = _token_usage(client)
    started = time.perf_counter()
    raw = client.chat_json(messages, temperature=0, timeout=timeout)
    elapsed = time.perf_counter() - started
    after = _token_usage(client)
    return LLMPlanningExecution(
        raw_response=raw,
        llm_call_attempted=True,
        snapshot_status="live",
        prompt_tokens=max(0, after[0] - before[0]),
        completion_tokens=max(0, after[1] - before[1]),
        total_tokens=max(0, after[2] - before[2]),
        recorded_latency_seconds=elapsed,
    )


def _validate_candidate(
    candidate: LLMSemanticQuery,
    *,
    output: LLMQueryPlanningOutput,
    analysis: QueryAnalysis,
    explicit_constraints: QueryConstraint | None,
    seen: set[str],
) -> str | None:
    query = candidate.query.strip()
    if not query:
        return "empty_query"
    key = _query_key(query)
    if key in seen:
        return "duplicate_query"
    if len(query) > MAX_QUERY_CHARACTERS:
        return "query_too_long"
    terms = _tokens(query)
    if len(terms) > MAX_QUERY_TERMS:
        return "too_many_terms"
    original = analysis.original_query
    if _has_identifier(query) and not _has_identifier(original):
        return "suspicious_identifier"
    if (
        (_looks_like_citation(query) or _looks_like_title_request(query))
        and not (
            _looks_like_citation(original) or _looks_like_title_request(original)
        )
    ):
        return "suspicious_citation"
    if _looks_like_prompt_leak(query):
        return "invalid_schema"

    required_terms = (
        list(explicit_constraints.must_include_terms)
        if explicit_constraints is not None
        and "must_include_terms" in explicit_constraints.explicit_fields
        else []
    )
    if any(not _contains_phrase(query, term) for term in required_terms):
        return "missing_must_have"
    excluded = analysis.constraints.exclude_terms
    if any(_contains_phrase(query, term) for term in excluded):
        return "contains_excluded_term"

    facets = identify_query_facets(analysis)
    core_terms = next(
        (facet.terms for facet in facets if facet.facet_type == "topic"),
        [],
    )
    core_tokens = _meaningful_tokens(" ".join(core_terms))
    query_tokens = set(_meaningful_tokens(query))
    mapped = _mapped_core_terms(output, core_tokens, query_tokens)
    overlap = len(set(core_tokens) & query_tokens)
    if core_tokens and overlap == 0 and not mapped:
        return "missing_core_topic"
    denominator = max(1, min(3, len(set(core_tokens))))
    retention = 1.0 if mapped else overlap / denominator
    if core_tokens and retention < MIN_INFORMATION_RETENTION:
        return "low_information_retention"
    original_tokens = set(_meaningful_tokens(original))
    if not (original_tokens & query_tokens) and not mapped:
        return "unrelated_expansion"
    return None


def _mapped_core_terms(
    output: LLMQueryPlanningOutput,
    core_tokens: list[str],
    query_tokens: set[str],
) -> bool:
    core = set(core_tokens)
    for facet in output.facets:
        original = set(_meaningful_tokens(" ".join(facet.original_terms)))
        normalized = set(_meaningful_tokens(" ".join(facet.normalized_terms)))
        if core.intersection(original) and query_tokens.intersection(normalized):
            return True
    return False


def _fallback(
    subqueries: list[SearchSubquery],
    result: QueryPlanningResult,
    *,
    reason: str,
    updates: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> LLMPlanningOutcome:
    warning = f"llm_semantic_fallback:{reason}"
    merged_updates = {
        **(updates or {}),
        "policy": "llm_semantic",
        "selected_subqueries": subqueries,
        "selected_subquery_count": len(subqueries),
        "fallback_used": True,
        "fallback_reason": reason,
    }
    return LLMPlanningOutcome(
        subqueries=subqueries,
        result=result.model_copy(update=merged_updates),
        warnings=_dedupe([warning, *(warnings or [])]),
    )


def _execution_updates(
    base: dict[str, Any],
    execution: LLMPlanningExecution | None,
) -> dict[str, Any]:
    if execution is None:
        return dict(base)
    return {
        **base,
        "snapshot_key": execution.snapshot_key,
        "snapshot_status": execution.snapshot_status,
        "llm_call_attempted": execution.llm_call_attempted,
        "replayed": execution.replayed,
        "llm_prompt_tokens": execution.prompt_tokens,
        "llm_completion_tokens": execution.completion_tokens,
        "llm_total_tokens": execution.total_tokens,
        "recorded_llm_latency_seconds": execution.recorded_latency_seconds,
    }


def _fallback_reason(exc: Exception) -> str:
    message = str(exc).casefold()
    if type(exc).__name__ == "BudgetStopError" or "budget_stop" in message:
        return "budget_exhausted"
    if "timeout" in message:
        return "llm_timeout"
    if "snapshot_missing" in message:
        return "snapshot_missing"
    if "invalid_json" in message or "malformed" in message or "invalid_schema" in message:
        return "invalid_schema"
    if "llm_unconfigured" in message or "disabled" in message:
        return "llm_unconfigured"
    return "llm_request_failed"


def _client_identity(client: Any | None) -> tuple[str, str | None, str | None]:
    runtime = get_llm_runtime_config()
    current = client
    for _ in range(3):
        if current is None:
            break
        provider = getattr(current, "provider", None)
        model = getattr(current, "model", None)
        base_url_host = getattr(current, "base_url_host", None)
        if provider or model or base_url_host:
            return (
                str(provider or runtime.provider or "injected"),
                str(model) if model else runtime.model,
                str(base_url_host) if base_url_host else runtime.base_url_host,
            )
        current = getattr(current, "_client", None)
    return runtime.provider, runtime.model, runtime.base_url_host


def _token_usage(client: Any | None) -> tuple[int, int, int]:
    usage = getattr(client, "token_usage", None)
    if usage is None:
        return 0, 0, 0

    def value(name: str) -> int:
        raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, 0)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    return value("prompt_tokens"), value("completion_tokens"), value("total_tokens")


def _search_facet_types(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        normalized = "topic" if value == "synonym" else value
        if normalized not in result:
            result.append(normalized)
    return result


def _safe_purpose(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    return normalized[:48] or "semantic_expansion"


def _has_identifier(value: str) -> bool:
    return bool(
        re.search(r"(?i)\b10\.\d{4,9}/\S+|\barxiv\s*:\s*\d{4}\.\d{4,5}\b|https?://", value)
    )


def _looks_like_citation(value: str) -> bool:
    return bool(
        re.search(r"(?i)\bet\s+al\.?\b|\[[0-9]{1,3}\]|\([12][0-9]{3}\)", value)
    )


def _looks_like_title_request(value: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:paper\s+)?titled\s+['\"“]|\btitle\s*:\s*['\"“]",
            value,
        )
    )


def _looks_like_prompt_leak(value: str) -> bool:
    lowered = value.casefold()
    return any(
        marker in lowered
        for marker in ("supplemental_queries", "json schema", "system prompt")
    )


def _contains_phrase(value: str, phrase: str) -> bool:
    normalized_phrase = " ".join(str(phrase).casefold().split())
    if not normalized_phrase:
        return True
    return normalized_phrase in " ".join(value.casefold().split())


def _tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9+.#_-]*|[\u4e00-\u9fff]+", value)


def _meaningful_tokens(value: str) -> list[str]:
    stop = {
        "a", "an", "and", "for", "in", "of", "on", "paper", "papers",
        "recent", "review", "study", "survey", "the", "to", "with",
    }
    return [token.casefold() for token in _tokens(value) if token.casefold() not in stop]


def _query_key(value: str) -> str:
    return " ".join(_tokens(value.casefold()))


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result
