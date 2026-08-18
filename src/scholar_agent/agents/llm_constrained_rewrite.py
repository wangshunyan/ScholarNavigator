"""固定预算内、默认关闭的受约束 LLM 学术检索改写。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from scholar_agent.agents.llm_query_planning import (
    LLMPlanningExecution,
    LLMPlanningOutcome,
    LLMPlanningRequest,
    LLMPlanningRuntime,
    _client_identity,
    _execute_live,
    _execution_updates,
    _fallback_reason,
    _has_identifier,
    _looks_like_citation,
    _looks_like_prompt_leak,
    _looks_like_title_request,
)
from scholar_agent.core.search_schemas import (
    LLMConstrainedRewriteOutput,
    QueryAnalysis,
    QueryConstraint,
    QueryPlanningResult,
    SearchSubquery,
)
from scholar_agent.llm.provider import get_llm_request_options
from scholar_agent.prompts.loader import load_prompt, render_messages


LLM_CONSTRAINED_REWRITE_PROMPT = "llm_constrained_rewrite"
MAX_REWRITE_CHARACTERS = 200
MAX_REWRITE_TERMS = 24
_ALLOWED_GENERIC_SYNONYMS = (
    "academic",
    "analysis",
    "assessment",
    "benchmark",
    "comparison",
    "evaluation",
    "evidence",
    "literature",
    "method",
    "methods",
    "model",
    "models",
    "performance",
    "review",
    "study",
    "studies",
    "survey",
    "学术",
    "分析",
    "评估",
    "基准",
    "比较",
    "证据",
    "文献",
    "方法",
    "模型",
    "性能",
    "综述",
    "研究",
)
_NEGATION_PATTERN = re.compile(
    r"(?i)(?<!\w)(?:without|excluding|except|not|no)(?:\s+[\w+.#-]+){0,3}"
    r"|(?:不含|不包括|排除|不要|无)[\u4e00-\u9fffA-Za-z0-9+.#-]*"
)
_TIME_PATTERN = re.compile(
    r"(?i)(?<!\w)(?:since|after|before|through|until)\s+(?:19|20)\d{2}"
    r"|(?<!\w)(?:19|20)\d{2}(?:\s*[-–—~至到]\s*(?:19|20)\d{2})?(?!\w)"
    r"|(?<!\w)(?:last\s+\d{1,2}\s+years?|recent|latest)(?!\w)"
    r"|(?:近\d+年|近三年|近年|最新)"
)


def plan_llm_constrained_rewrite(
    query_analysis: QueryAnalysis,
    *,
    current_subqueries: list[SearchSubquery],
    current_result: QueryPlanningResult,
    run_profile: str,
    explicit_constraints: QueryConstraint | None,
    llm_client: Any | None,
    runtime: LLMPlanningRuntime | None = None,
) -> LLMPlanningOutcome:
    """用一条经本地质量门校验的 LLM 查询替换最低优先级派生查询。"""

    selected = [item.model_copy(deep=True) for item in current_subqueries]
    replace_index = _replacement_index(selected)
    protected_terms = _protected_terms(query_analysis, explicit_constraints)
    summary = _input_summary(
        query_analysis,
        selected,
        replace_index=replace_index,
        protected_terms=protected_terms,
    )
    base_updates: dict[str, Any] = {
        "policy": "llm_constrained_rewrite",
        "original_query_retained": True,
        "constrained_rewrite_input_summary": summary,
        "constrained_rewrite_replaced_index": replace_index,
    }
    if replace_index is None:
        return _fallback(
            selected,
            current_result,
            reason="no_derived_query_to_replace",
            updates=base_updates,
        )

    replaced = selected[replace_index]
    base_updates.update(
        {
            "constrained_rewrite_replaced_query": replaced.query,
            "constrained_rewrite_replaced_purpose": replaced.purpose,
        }
    )
    try:
        prompt = load_prompt(LLM_CONSTRAINED_REWRITE_PROMPT)
    except Exception:  # Prompt 故障只能稳定回退，不能影响正式检索。
        return _fallback(
            selected,
            current_result,
            reason="prompt_load_failed",
            updates=base_updates,
        )

    provider, model, base_url_host = _client_identity(llm_client)
    identity_provider = getattr(runtime, "identity", None)
    runtime_identity = identity_provider() if callable(identity_provider) else None
    if llm_client is None and runtime_identity is not None:
        provider, model, base_url_host = runtime_identity
    base_updates.update(
        {
            "provider": provider,
            "model": model,
            "prompt_name": prompt.name,
            "prompt_version": prompt.version,
            "prompt_hash": prompt.content_hash,
        }
    )
    if llm_client is None and runtime is None:
        return _fallback(
            selected,
            current_result,
            reason="llm_unconfigured",
            updates=base_updates,
        )

    input_payload = {
        **summary,
        "run_profile": run_profile,
        "allowed_generic_synonyms": list(_ALLOWED_GENERIC_SYNONYMS),
    }
    request_options = get_llm_request_options()
    request = LLMPlanningRequest(
        query_planning_policy="llm_constrained_rewrite",
        provider=provider,
        model=model,
        base_url_host=base_url_host,
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        prompt_hash=prompt.content_hash,
        input_payload=input_payload,
        run_profile=run_profile,
        max_supplemental_queries=1,
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
        output = LLMConstrainedRewriteOutput.model_validate(execution.raw_response)
    except ValidationError:
        return _fallback(
            selected,
            current_result,
            reason="invalid_schema",
            updates=_execution_updates(base_updates, execution),
        )
    except Exception as exc:  # noqa: BLE001 - 可选策略不得中断检索
        reason = _fallback_reason(exc)
        failure_updates = _execution_updates(base_updates, execution)
        diagnostics_provider = getattr(runtime, "failure_diagnostics", None)
        if callable(diagnostics_provider):
            failure_updates.update(diagnostics_provider())
        if llm_client is not None and reason not in {
            "budget_exhausted",
            "snapshot_missing",
        }:
            failure_updates["llm_call_attempted"] = True
        return _fallback(
            selected,
            current_result,
            reason=reason,
            updates=failure_updates,
        )

    rewritten = " ".join(output.rewritten_query.split())
    rejection = _validate_rewrite(
        rewritten,
        query_analysis=query_analysis,
        current_subqueries=selected,
        protected_terms=protected_terms,
    )
    updates = {
        **_execution_updates(base_updates, execution),
        "output_valid": True,
        "generated_query_count": 1,
        "constrained_rewrite_query": rewritten,
        "terminology_expansions": _dedupe(output.generic_synonyms_used),
    }
    if rejection is not None:
        return _fallback(
            selected,
            current_result,
            reason="rewrite_rejected",
            updates={
                **updates,
                "rejected_query_count": 1,
                "rejection_reasons": {rejection: 1},
                "constrained_rewrite_validation_rejections": [rejection],
            },
            warnings=[f"llm_constrained_rewrite_rejected:{rejection}"],
        )

    selected[replace_index] = replaced.model_copy(
        update={
            "query": rewritten,
            "purpose": "llm_constrained_rewrite",
            "provenance": _dedupe(
                [
                    *replaced.provenance,
                    "llm_constrained_rewrite",
                    f"prompt:{prompt.name}@{prompt.version}",
                    f"replaced:{replaced.purpose}",
                ]
            ),
        }
    )
    result = current_result.model_copy(
        update={
            **updates,
            "policy": "llm_constrained_rewrite",
            "selected_subqueries": selected,
            "selected_subquery_count": len(selected),
            "accepted_query_count": 1,
            "accepted_queries": [rewritten],
            "rejected_query_count": 0,
            "rejection_reasons": {},
            "constrained_rewrite_validation_rejections": [],
            "fallback_used": False,
            "fallback_reason": None,
            "constrained_rewrite_skip_reason": None,
        }
    )
    return LLMPlanningOutcome(
        subqueries=selected,
        result=result,
        warnings=_dedupe(output.warnings),
    )


def _replacement_index(subqueries: list[SearchSubquery]) -> int | None:
    replaceable = [
        (item.priority, index)
        for index, item in enumerate(subqueries)
        if item.purpose != "original_query"
    ]
    return max(replaceable)[1] if replaceable else None


def _input_summary(
    analysis: QueryAnalysis,
    subqueries: list[SearchSubquery],
    *,
    replace_index: int | None,
    protected_terms: list[str],
) -> dict[str, object]:
    target = subqueries[replace_index] if replace_index is not None else None
    return {
        "original_query": analysis.original_query,
        "language": analysis.language,
        "intent": analysis.intent,
        "domain": analysis.domain,
        "protected_terms": protected_terms,
        "existing_queries": [item.query for item in subqueries],
        "replace_target": (
            {
                "index": replace_index,
                "query": target.query,
                "purpose": target.purpose,
            }
            if target is not None
            else None
        ),
    }


def _protected_terms(
    analysis: QueryAnalysis,
    explicit_constraints: QueryConstraint | None,
) -> list[str]:
    original = analysis.original_query
    constraints = analysis.constraints
    explicit_fields = set(constraints.explicit_fields)
    values: list[str] = []
    for field_name in (
        "must_include_terms",
        "exclude_terms",
        "venues",
        "methods",
        "datasets",
    ):
        if field_name in explicit_fields:
            values.extend(getattr(constraints, field_name))
    if explicit_constraints is not None:
        values.extend(explicit_constraints.must_include_terms)
        values.extend(explicit_constraints.exclude_terms)
    values.extend(
        match.group(0)
        for match in re.finditer(
            r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9+.#-]{1,}|[A-Z][a-z]+[A-Z][A-Za-z0-9+.#-]*)(?![A-Za-z0-9])",
            original,
        )
    )
    values.extend(match.group(0) for match in _NEGATION_PATTERN.finditer(original))
    values.extend(match.group(0) for match in _TIME_PATTERN.finditer(original))
    values.extend(
        match.group(1).strip()
        for match in re.finditer(r"['\"“”‘’]([^'\"“”‘’]{2,80})['\"“”‘’]", original)
    )
    return _dedupe(values)


def _validate_rewrite(
    rewritten: str,
    *,
    query_analysis: QueryAnalysis,
    current_subqueries: list[SearchSubquery],
    protected_terms: list[str],
) -> str | None:
    if not rewritten:
        return "empty_query"
    if len(rewritten) > MAX_REWRITE_CHARACTERS:
        return "query_too_long"
    if len(_tokens(rewritten)) > MAX_REWRITE_TERMS:
        return "too_many_terms"
    original = query_analysis.original_query
    if any(_query_key(rewritten) == _query_key(item.query) for item in current_subqueries):
        return "duplicate_query"
    if _has_identifier(rewritten) and not _has_identifier(original):
        return "suspicious_identifier"
    if (
        _looks_like_citation(rewritten)
        or _looks_like_title_request(rewritten)
    ) and not (
        _looks_like_citation(original) or _looks_like_title_request(original)
    ):
        return "suspicious_citation"
    if _looks_like_prompt_leak(rewritten):
        return "invalid_schema"
    if any(not _contains_phrase(rewritten, term) for term in protected_terms):
        return "missing_protected_term"
    if _introduced_tokens(original, rewritten, protected_terms=protected_terms):
        return "introduced_entity_or_term"
    return None


def _introduced_tokens(
    original: str,
    rewritten: str,
    *,
    protected_terms: list[str],
) -> list[str]:
    original_tokens = {token.casefold() for token in _tokens(original)}
    original_tokens.update(
        token.casefold()
        for value in protected_terms
        for token in _tokens(value)
    )
    allowed = {
        token.casefold()
        for value in _ALLOWED_GENERIC_SYNONYMS
        for token in _tokens(value)
    }
    return [
        token
        for token in _tokens(rewritten)
        if token.casefold() not in original_tokens
        and token.casefold() not in allowed
        and not (
            re.fullmatch(r"[\u4e00-\u9fff]+", token)
            and token.casefold() in original.casefold()
        )
        and not token.isdigit()
    ]


def _fallback(
    subqueries: list[SearchSubquery],
    result: QueryPlanningResult,
    *,
    reason: str,
    updates: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> LLMPlanningOutcome:
    merged_updates = {
        **(updates or {}),
        "policy": "llm_constrained_rewrite",
        "selected_subqueries": subqueries,
        "selected_subquery_count": len(subqueries),
        "fallback_used": True,
        "fallback_reason": reason,
        "constrained_rewrite_skip_reason": reason,
    }
    return LLMPlanningOutcome(
        subqueries=subqueries,
        result=result.model_copy(update=merged_updates),
        warnings=_dedupe(
            [f"llm_constrained_rewrite_fallback:{reason}", *(warnings or [])]
        ),
    )


def _contains_phrase(value: str, phrase: str) -> bool:
    normalized_value = " ".join(value.casefold().split())
    normalized_phrase = " ".join(str(phrase).casefold().split())
    return bool(normalized_phrase) and normalized_phrase in normalized_value


def _tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9+.#_-]*|[\u4e00-\u9fff]+", value)


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
