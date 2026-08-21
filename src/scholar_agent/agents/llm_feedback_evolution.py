"""Constrained, post-retrieval LLM feedback for one second-round query."""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from scholar_agent.agents.query_evolution import analyze_query_coverage
from scholar_agent.core.search_schemas import (
    EvolvedSubquery,
    JudgementResult,
    LLMFeedbackDiagnostics,
    LLMQueryPlanningOutput,
    QueryAnalysis,
    QueryEvolutionRecord,
    RankedPaper,
    SearchPlan,
)
from scholar_agent.evaluation.llm_feedback_snapshots import (
    LLMFeedbackExecution,
    LLMFeedbackRequest,
)
from scholar_agent.llm.provider import get_llm_request_options, get_llm_runtime_config
from scholar_agent.prompts.loader import (
    load_prompt,
    render_untrusted_metadata_messages,
)


LLM_FEEDBACK_PROMPT = "llm_feedback_evolution"
LLM_FEEDBACK_SCHEMA_VERSION = "1"
MAX_FEEDBACK_PAPERS = 3
MAX_FEEDBACK_QUERIES = 1
MAX_QUERY_CHARACTERS = 200
MAX_QUERY_TERMS = 24
_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|[\u4e00-\u9fff]{2,}")
_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:arxiv\s*:\s*|10\.\d{4,9}/|pmid\s*[:#]?)", re.IGNORECASE
)
_STOPWORDS = {
    "a", "an", "and", "for", "from", "in", "of", "on", "or", "the",
    "to", "with", "paper", "papers", "research", "study", "studies",
}


def evolve_with_llm_feedback(
    query_analysis: QueryAnalysis,
    search_plan: SearchPlan,
    judgements: list[JudgementResult],
    ranked_papers: list[RankedPaper],
    used_queries: set[str],
    *,
    llm_client: Any | None,
    runtime: Any | None = None,
) -> QueryEvolutionRecord:
    """Generate at most one safe feedback query from first-round result evidence.

    Candidate metadata is intentionally treated as untrusted data. This function
    never receives qrels/gold, never mutates the first-round plan, and returns a
    no-query record on any provider or validation failure.
    """

    gap = analyze_query_coverage(query_analysis, judgements)
    packet = _feedback_packet(query_analysis, gap, ranked_papers)
    diagnostics = _base_diagnostics(llm_client, candidate_count=len(packet["candidates"]))
    if not gap.needs_evolution:
        diagnostics.skipped_reason = "coverage_sufficient"
        return _record(gap, diagnostics, "coverage_sufficient")
    if not packet["candidates"]:
        diagnostics.skipped_reason = "no_ranked_feedback"
        return _record(gap, diagnostics, "no_ranked_feedback")
    diagnostics.eligible_for_feedback = True
    if llm_client is None and runtime is None:
        diagnostics.fallback_used = True
        diagnostics.fallback_reason = "llm_unconfigured"
        return _record(gap, diagnostics, "llm_unconfigured")

    try:
        prompt = load_prompt(LLM_FEEDBACK_PROMPT)
    except Exception:  # Prompt loading must fail closed without an LLM call.
        diagnostics.fallback_used = True
        diagnostics.fallback_reason = "prompt_load_failed"
        return _record(gap, diagnostics, "prompt_load_failed")

    diagnostics.prompt_name = prompt.name
    diagnostics.prompt_version = prompt.version
    diagnostics.prompt_hash = prompt.content_hash
    diagnostics.schema_version = LLM_FEEDBACK_SCHEMA_VERSION
    diagnostics.temperature = 0.0
    options = get_llm_request_options()
    provider, model, base_url_host = _client_identity(llm_client)
    identity_provider = getattr(runtime, "identity", None)
    runtime_identity = identity_provider() if callable(identity_provider) else None
    if llm_client is None and runtime_identity is not None:
        provider, model, base_url_host = runtime_identity
    diagnostics.provider = provider
    diagnostics.model = model
    request = LLMFeedbackRequest(
        provider=provider,
        model=model,
        base_url_host=base_url_host,
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        prompt_hash=prompt.content_hash,
        request_identity=_request_identity(packet),
        temperature=0,
        max_tokens=int(options["max_tokens"]),
        max_supplemental_queries=MAX_FEEDBACK_QUERIES,
    )
    messages = render_untrusted_metadata_messages(prompt.name, packet)
    execution: LLMFeedbackExecution | None = None
    try:
        execution = (
            runtime.execute(
                request,
                messages,
                llm_client,
                timeout=float(options["timeout_seconds"]),
            )
            if runtime is not None
            else _execute_live(
                llm_client,
                messages,
                timeout=float(options["timeout_seconds"]),
            )
        )
        _apply_execution(diagnostics, execution)
        output = LLMQueryPlanningOutput.model_validate(execution.raw_response)
    except ValidationError:
        diagnostics.fallback_used = True
        diagnostics.fallback_reason = "invalid_schema"
        return _record(gap, diagnostics, "invalid_schema")
    except Exception as exc:  # Optional feedback must never break search.
        _apply_runtime_failure(diagnostics, runtime)
        reason = _failure_reason(exc)
        diagnostics.fallback_used = True
        diagnostics.fallback_reason = reason
        return _record(gap, diagnostics, reason)

    diagnostics.output_valid = True
    diagnostics.generated_query_count = min(
        MAX_FEEDBACK_QUERIES, len(output.supplemental_queries)
    )
    rejection_reasons: Counter[str] = Counter()
    query = None
    purpose = "feedback"
    for candidate in output.supplemental_queries[:MAX_FEEDBACK_QUERIES]:
        normalized = " ".join(candidate.query.split())
        reason = _validate_query(
            normalized,
            query_analysis=query_analysis,
            used_queries=used_queries,
        )
        if reason is not None:
            rejection_reasons[reason] += 1
            continue
        query = normalized
        purpose = _safe_purpose(candidate.purpose)
        break
    diagnostics.rejection_reasons = dict(sorted(rejection_reasons.items()))
    if query is None:
        # The first-round result remains authoritative. A locally rejected
        # follow-up is a safe no-op, not a provider fallback or degraded path.
        diagnostics.skipped_reason = "all_queries_rejected"
        return _record(gap, diagnostics, "all_queries_rejected")

    diagnostics.accepted_query_count = 1
    return QueryEvolutionRecord(
        round_index=2,
        policy="llm_feedback",
        coverage_gap=gap,
        seed_count=len(packet["candidates"]),
        seed_paper_titles=[item["title"] for item in packet["candidates"]],
        generated_queries=[
            EvolvedSubquery(
                query=query,
                source_hints=search_plan.selected_sources,
                purpose=f"llm_feedback:{purpose}",
                seed_paper_titles=[item["title"] for item in packet["candidates"]],
                generated_by="llm",
                generation_policy="llm_feedback",
                gap_dimensions=list(gap.reasons),
            )
        ],
        llm_feedback=diagnostics,
    )


def _record(
    gap: Any,
    diagnostics: LLMFeedbackDiagnostics,
    reason: str,
) -> QueryEvolutionRecord:
    return QueryEvolutionRecord(
        round_index=2,
        policy="llm_feedback",
        coverage_gap=gap,
        llm_feedback=diagnostics,
        skipped_reasons=[reason],
        warnings=[f"llm_feedback_fallback:{reason}"],
    )


def _feedback_packet(
    query_analysis: QueryAnalysis,
    gap: Any,
    ranked_papers: list[RankedPaper],
) -> dict[str, Any]:
    candidates = []
    for ranked in ranked_papers[:MAX_FEEDBACK_PAPERS]:
        title = " ".join(ranked.paper.title.split())[:240]
        if not title:
            continue
        candidates.append(
            {
                "rank": ranked.rank,
                "title": title,
                "abstract_excerpt": " ".join(ranked.paper.abstract.split())[:600],
                "matched_terms": list(ranked.matched_terms[:8]),
            }
        )
    return {
        "original_query": query_analysis.original_query,
        "constraints": query_analysis.constraints.model_dump(
            mode="json", exclude={"explicit_fields"}
        ),
        "coverage_gap": gap.model_dump(mode="json"),
        "max_supplemental_queries": MAX_FEEDBACK_QUERIES,
        "candidates": candidates,
    }


def _base_diagnostics(client: Any | None, *, candidate_count: int) -> LLMFeedbackDiagnostics:
    runtime = get_llm_runtime_config()
    return LLMFeedbackDiagnostics(
        provider=str(getattr(client, "provider", None) or runtime.provider or "disabled"),
        model=(str(getattr(client, "model", None)) if getattr(client, "model", None) else runtime.model),
        candidate_count=candidate_count,
    )


def _client_identity(client: Any | None) -> tuple[str, str | None, str | None]:
    runtime = get_llm_runtime_config()
    provider = str(getattr(client, "provider", None) or runtime.provider or "disabled")
    model = str(getattr(client, "model", None)) if getattr(client, "model", None) else runtime.model
    raw_url = getattr(client, "base_url", None)
    parsed = urlparse(str(raw_url or ""))
    return provider, model, parsed.netloc or runtime.base_url_host


def _request_identity(packet: dict[str, Any]) -> dict[str, Any]:
    import hashlib
    import json

    def digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()

    return {
        "original_query_hash": digest(packet["original_query"]),
        "constraints_hash": digest(packet["constraints"]),
        "coverage_gap_hash": digest(packet["coverage_gap"]),
        "candidate_hashes": [digest(item) for item in packet["candidates"]],
        "candidate_count": len(packet["candidates"]),
        "max_supplemental_queries": int(packet["max_supplemental_queries"]),
    }


def _execute_live(
    client: Any | None,
    messages: list[dict[str, str]],
    *,
    timeout: float,
) -> LLMFeedbackExecution:
    if client is None:
        raise RuntimeError("llm_unconfigured")
    before = _token_usage(client)
    started = time.perf_counter()
    raw = client.chat_json(messages, temperature=0, timeout=timeout)
    after = _token_usage(client)
    transport = getattr(client, "last_call_diagnostics", None)
    if not isinstance(raw, dict):
        raise ValueError("llm_feedback_invalid_schema")
    return LLMFeedbackExecution(
        raw_response=raw,
        snapshot_status="live",
        llm_call_attempted=True,
        prompt_tokens=max(0, after[0] - before[0]),
        completion_tokens=max(0, after[1] - before[1]),
        total_tokens=max(0, after[2] - before[2]),
        recorded_latency_seconds=time.perf_counter() - started,
        http_attempts=max(0, int(getattr(transport, "http_attempts", 0))),
        http_429_count=max(0, int(getattr(transport, "http_429_count", 0))),
        retry_after_seconds=[max(0.0, float(value)) for value in getattr(transport, "retry_after_seconds", ())],
        retry_wait_seconds=max(0.0, float(getattr(transport, "retry_wait_seconds", 0.0))),
        provider_failure_class=getattr(transport, "failure_class", None),
        provider_cache_hit=bool(getattr(transport, "cache_hit", False)),
    )


def _apply_execution(
    diagnostics: LLMFeedbackDiagnostics,
    execution: LLMFeedbackExecution,
) -> None:
    diagnostics.snapshot_key = execution.snapshot_key
    diagnostics.snapshot_status = execution.snapshot_status
    diagnostics.llm_call_attempted = execution.llm_call_attempted
    diagnostics.replayed = execution.replayed
    diagnostics.prompt_tokens = execution.prompt_tokens
    diagnostics.completion_tokens = execution.completion_tokens
    diagnostics.total_tokens = execution.total_tokens
    diagnostics.latency_seconds = execution.recorded_latency_seconds
    diagnostics.http_attempts = execution.http_attempts
    diagnostics.http_429_count = execution.http_429_count
    diagnostics.retry_after_seconds = list(execution.retry_after_seconds)
    diagnostics.retry_wait_seconds = execution.retry_wait_seconds
    diagnostics.provider_failure_class = execution.provider_failure_class
    diagnostics.provider_cache_hit = execution.provider_cache_hit


def _apply_runtime_failure(diagnostics: LLMFeedbackDiagnostics, runtime: Any | None) -> None:
    provider = getattr(runtime, "failure_diagnostics", None)
    values = provider() if callable(provider) else {}
    for name in ("snapshot_key", "snapshot_status", "llm_call_attempted"):
        if name in values:
            setattr(diagnostics, name, values[name])


def _validate_query(
    query: str,
    *,
    query_analysis: QueryAnalysis,
    used_queries: set[str],
) -> str | None:
    if not query:
        return "empty_query"
    if query.casefold() in {value.casefold() for value in used_queries}:
        return "duplicate_query"
    if len(query) > MAX_QUERY_CHARACTERS:
        return "query_too_long"
    query_tokens = _meaningful_tokens(query)
    if len(query_tokens) > MAX_QUERY_TERMS:
        return "too_many_terms"
    original = query_analysis.original_query
    if _IDENTIFIER_PATTERN.search(query) and not _IDENTIFIER_PATTERN.search(original):
        return "suspicious_identifier"
    if any(_contains_phrase(query, term) for term in query_analysis.constraints.exclude_terms):
        return "contains_excluded_term"
    if "must_include_terms" in query_analysis.constraints.explicit_fields:
        for term in query_analysis.constraints.must_include_terms:
            if not _contains_phrase(query, term):
                return "missing_must_have"
    original_tokens = set(_meaningful_tokens(original))
    if original_tokens and not original_tokens.intersection(query_tokens):
        return "missing_core_topic"
    return None


def _meaningful_tokens(value: str) -> list[str]:
    return [
        token.casefold()
        for token in _TOKEN_PATTERN.findall(value)
        if token.casefold() not in _STOPWORDS
    ]


def _contains_phrase(text: str, term: str) -> bool:
    return " ".join(term.casefold().split()) in " ".join(text.casefold().split())


def _safe_purpose(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    return normalized[:80] or "feedback"


def _token_usage(client: Any) -> tuple[int, int, int]:
    usage = getattr(client, "token_usage", None)
    if usage is None:
        return (0, 0, 0)
    getter = usage.get if isinstance(usage, dict) else lambda key, default=0: getattr(usage, key, default)
    return tuple(max(0, int(getter(key, 0) or 0)) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))


def _update_usage(
    diagnostics: LLMFeedbackDiagnostics,
    before: tuple[int, int, int],
    after: tuple[int, int, int],
) -> None:
    diagnostics.prompt_tokens = max(0, after[0] - before[0])
    diagnostics.completion_tokens = max(0, after[1] - before[1])
    diagnostics.total_tokens = max(0, after[2] - before[2])


def _update_transport(diagnostics: LLMFeedbackDiagnostics, client: Any) -> None:
    value = getattr(client, "last_call_diagnostics", None)
    if value is None:
        return
    diagnostics.http_attempts = max(0, int(getattr(value, "http_attempts", 0)))
    diagnostics.http_429_count = max(0, int(getattr(value, "http_429_count", 0)))
    diagnostics.retry_after_seconds = [max(0.0, float(item)) for item in getattr(value, "retry_after_seconds", ())]
    diagnostics.retry_wait_seconds = max(0.0, float(getattr(value, "retry_wait_seconds", 0.0)))
    diagnostics.provider_failure_class = getattr(value, "failure_class", None)
    diagnostics.provider_cache_hit = bool(getattr(value, "cache_hit", False))


def _failure_reason(exc: Exception) -> str:
    text = str(exc).casefold()
    if type(exc).__name__ == "BudgetStopError" or "budget_stop" in text:
        return "budget_exhausted"
    if "timeout" in text:
        return "llm_timeout"
    if "snapshot_missing" in text:
        return "snapshot_missing"
    if "disabled" in text or "unconfigured" in text:
        return "llm_unconfigured"
    return "llm_request_failed"
