"""FastAPI routes for the ScholarNavigator backend API."""

from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...core.api_schemas import (
    ConnectorRuntimeConfig,
    CostReport,
    HealthResponse,
    LLMRuntimeConfig,
    RunProgress,
    RuntimeConfigResponse,
    RuntimeFeatures,
    RuntimeLimits,
    SearchConstraints,
    SearchPlan,
    SearchRunCreateRequest,
    SearchRunCreateResponse,
    SearchRunResultResponse,
    SearchRunStatusResponse,
)
from ...core.search_schemas import (
    QueryConstraint,
    QueryEvolutionPolicy,
    JudgementPolicy,
    QueryPlanningPolicy,
    RankingPolicy,
    RunProfile,
    SearchBudget,
    TimeRange,
)
from ...core.local_bm25_env import (
    configure_local_bm25_from_env,
    local_bm25_config_from_env,
)
from ...core.local_hybrid_env import (
    configure_local_hybrid_from_env,
    local_hybrid_config_from_env,
)
from ...core.full_text_evidence import (
    DEFAULT_FULL_TEXT_TIMEOUT_SECONDS,
    DEFAULT_MAX_FULL_TEXT_BYTES,
    DEFAULT_MAX_PDF_PAGES,
    FullTextFetchResult,
    fetch_open_full_text,
)
from ...llm.provider import get_llm_runtime_config
from ...services.api_mapper import map_search_service_output_to_api_result
from ...services.search_service import (
    ENABLE_LLM_JUDGEMENT_ENV,
    ENABLE_LLM_QUERY_UNDERSTANDING_ENV,
    SearchCancelled,
    SearchService,
    _env_flag,
)


API_VERSION = "0.1.0"
DEFAULT_REAL_PREVIEW_MAX_WORKERS = 2
REAL_PREVIEW_MAX_WORKERS_ENV = "REAL_PREVIEW_MAX_WORKERS"
DEFAULT_REAL_SEARCH_MAX_WORKERS = 2
REAL_SEARCH_MAX_WORKERS_ENV = "REAL_SEARCH_MAX_WORKERS"
DEFAULT_REAL_SEARCH_BACKGROUND_WORKERS = 2
REAL_SEARCH_BACKGROUND_WORKERS_ENV = "REAL_SEARCH_BACKGROUND_WORKERS"
DEFAULT_REAL_SEARCH_RUN_TTL_SECONDS = 3600
REAL_SEARCH_RUN_TTL_SECONDS_ENV = "REAL_SEARCH_RUN_TTL_SECONDS"
DEFAULT_REAL_SEARCH_MAX_STORED_RUNS = 200
REAL_SEARCH_MAX_STORED_RUNS_ENV = "REAL_SEARCH_MAX_STORED_RUNS"
SEMANTIC_SCHOLAR_API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"
NCBI_API_KEY_ENV = "NCBI_API_KEY"
PUBMED_API_KEY_ENV = "PUBMED_API_KEY"
REAL_SEARCH_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

router = APIRouter(prefix="/api/v1", tags=["api"])


class FullTextFetchRequest(BaseModel):
    """Explicit, license-gated full-text request; never used by ranking."""

    source_url: str = Field(..., min_length=1, max_length=2048)
    license_id: str = Field(..., min_length=1, max_length=120)
    license_verified: bool = False
    allowed_hosts: list[str] = Field(..., min_length=1, max_length=1)
    timeout_seconds: float = Field(
        default=DEFAULT_FULL_TEXT_TIMEOUT_SECONDS, gt=0.0, le=30.0
    )
    max_bytes: int = Field(default=DEFAULT_MAX_FULL_TEXT_BYTES, gt=0, le=10_000_000)
    max_pdf_pages: int = Field(default=DEFAULT_MAX_PDF_PAGES, gt=0, le=200)


@dataclass
class RealRun:
    run_id: str
    request: SearchRunCreateRequest
    status: str
    current_stage: str
    progress: RunProgress
    cost_report: CostReport
    result: SearchRunResultResponse | None
    events: list[tuple[str, dict[str, Any]]]
    error_message: str | None
    cancel_requested: bool
    created_at: datetime
    updated_at: datetime
    terminal_event_emitted: bool = False


class InternalSearchPreviewRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=20, ge=1, le=100)
    run_profile: RunProfile = "balanced"
    enable_refchain: bool = False
    enable_semantic_seed_expansion: bool = False
    enable_query_evolution: bool = False
    query_evolution_policy: QueryEvolutionPolicy = "coverage_gap"
    query_planning_policy: QueryPlanningPolicy = "current_rules"
    ranking_policy: RankingPolicy = "current_rules"
    judgement_policy: JudgementPolicy = "current_rules"
    enable_llm_query_understanding: bool | None = None
    enable_llm_judgement: bool | None = None
    current_year: int | None = Field(default=None, ge=1900, le=2200)


class InternalSearchPreviewResponse(BaseModel):
    query_analysis: dict[str, Any]
    search_plan: dict[str, Any]
    query_evolution_records: list[dict[str, Any]]
    refchain_output: dict[str, Any] | None
    semantic_seed_expansion_output: dict[str, Any] | None
    synthesis_output: dict[str, Any] | None
    ranked_papers: list[dict[str, Any]]
    raw_count: int
    deduplicated_count: int
    warnings: list[str]
    source_stats: list[dict[str, Any]]
    latency_seconds: float


_REAL_RUNS: dict[str, RealRun] = {}
_REAL_RUNS_LOCK = RLock()
_REAL_SEARCH_EXECUTOR: ThreadPoolExecutor | None = None
_REAL_SEARCH_EXECUTOR_LOCK = RLock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _real_links(run_id: str) -> dict[str, str]:
    return {
        "self": f"/api/v1/real/search/runs/{run_id}",
        "events": f"/api/v1/real/search/runs/{run_id}/events",
        "result": f"/api/v1/real/search/runs/{run_id}/result",
    }


def _get_real_run(run_id: str) -> RealRun:
    with _REAL_RUNS_LOCK:
        try:
            return _REAL_RUNS[run_id]
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown real run_id: {run_id}",
            ) from exc


def _model_dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


@router.post(
    "/full-text/fetch",
    response_model=FullTextFetchResult,
    tags=["full-text-evidence"],
)
def fetch_full_text_evidence(request: FullTextFetchRequest) -> FullTextFetchResult:
    """Fetch one caller-authorized open document without touching search ranking.

    The caller must provide both a verified license assertion and an exact
    single-host allow-list.  This endpoint is intentionally separate from
    search runs so full-text retrieval is observable, bounded, and opt-in.
    """

    parsed = urlparse(request.source_url.strip())
    allowed_host = request.allowed_hosts[0].strip().casefold()
    if not parsed.hostname or parsed.hostname.casefold() != allowed_host:
        raise HTTPException(
            status_code=400,
            detail="full_text_allowed_host_must_match_source_url",
        )
    try:
        return fetch_open_full_text(
            source_url=request.source_url,
            license_id=request.license_id,
            license_verified=request.license_verified,
            allowed_hosts={allowed_host},
            timeout_seconds=request.timeout_seconds,
            max_bytes=request.max_bytes,
            max_pdf_pages=request.max_pdf_pages,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _max_workers_from_env(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default


def _real_preview_max_workers() -> int:
    return _max_workers_from_env(
        REAL_PREVIEW_MAX_WORKERS_ENV,
        DEFAULT_REAL_PREVIEW_MAX_WORKERS,
    )


def _real_search_max_workers() -> int:
    return _max_workers_from_env(
        REAL_SEARCH_MAX_WORKERS_ENV,
        DEFAULT_REAL_SEARCH_MAX_WORKERS,
    )


def _real_search_background_workers() -> int:
    return _max_workers_from_env(
        REAL_SEARCH_BACKGROUND_WORKERS_ENV,
        DEFAULT_REAL_SEARCH_BACKGROUND_WORKERS,
    )


def _int_from_env(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _real_search_run_ttl_seconds() -> int:
    return _int_from_env(
        REAL_SEARCH_RUN_TTL_SECONDS_ENV,
        DEFAULT_REAL_SEARCH_RUN_TTL_SECONDS,
    )


def _real_search_max_stored_runs() -> int:
    return _int_from_env(
        REAL_SEARCH_MAX_STORED_RUNS_ENV,
        DEFAULT_REAL_SEARCH_MAX_STORED_RUNS,
    )


def _cleanup_real_runs(
    *,
    protected_run_id: str | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Remove old terminal real-search runs from the in-memory store."""

    timestamp = now or _now()
    deleted: list[str] = []
    ttl_seconds = _real_search_run_ttl_seconds()
    max_stored_runs = _real_search_max_stored_runs()

    with _REAL_RUNS_LOCK:
        if ttl_seconds > 0:
            threshold = timestamp.timestamp() - ttl_seconds
            expired_run_ids = [
                run_id
                for run_id, run in _REAL_RUNS.items()
                if run_id != protected_run_id
                and run.status in REAL_SEARCH_TERMINAL_STATUSES
                and run.updated_at.timestamp() < threshold
            ]
            for run_id in expired_run_ids:
                if _REAL_RUNS.pop(run_id, None) is not None:
                    deleted.append(run_id)

        if max_stored_runs > 0 and len(_REAL_RUNS) > max_stored_runs:
            delete_count = len(_REAL_RUNS) - max_stored_runs
            candidates = sorted(
                (
                    run
                    for run_id, run in _REAL_RUNS.items()
                    if run_id != protected_run_id
                    and run.status in REAL_SEARCH_TERMINAL_STATUSES
                ),
                key=lambda run: (run.updated_at, run.created_at, run.run_id),
            )
            for run in candidates[:delete_count]:
                if _REAL_RUNS.pop(run.run_id, None) is not None:
                    deleted.append(run.run_id)

    return deleted


def _preview_search_service() -> SearchService:
    _configure_local_sources_from_env(build_index=True)
    return SearchService(max_workers=_real_preview_max_workers())


def _real_search_service() -> SearchService:
    _configure_local_sources_from_env(build_index=True)
    return SearchService(max_workers=_real_search_max_workers())


def _configure_local_sources_from_env(*, build_index: bool) -> None:
    configure_local_hybrid_from_env(build_index=build_index)
    configure_local_bm25_from_env(build_index=build_index)


def _real_search_executor() -> ThreadPoolExecutor:
    global _REAL_SEARCH_EXECUTOR
    with _REAL_SEARCH_EXECUTOR_LOCK:
        if _REAL_SEARCH_EXECUTOR is None:
            _REAL_SEARCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=_real_search_background_workers(),
                thread_name_prefix="real-search",
            )
        return _REAL_SEARCH_EXECUTOR


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=API_VERSION, time=_now())


@router.get("/runtime/config", response_model=RuntimeConfigResponse)
def runtime_config() -> RuntimeConfigResponse:
    llm_runtime = get_llm_runtime_config()
    llm_feature_enabled = (
        llm_runtime.available
        and _env_flag(ENABLE_LLM_QUERY_UNDERSTANDING_ENV, default=False)
    )
    llm_judgement_enabled = (
        llm_runtime.available
        and _env_flag(ENABLE_LLM_JUDGEMENT_ENV, default=False)
    )
    return RuntimeConfigResponse(
        mode="real_search",
        llm=LLMRuntimeConfig(
            provider=llm_runtime.provider,
            model=llm_runtime.model,
            available=llm_runtime.available,
            base_url_host=llm_runtime.base_url_host,
            reason=llm_runtime.reason,
        ),
        connectors=[
            ConnectorRuntimeConfig(
                name="openalex",
                available=True,
                requires_key=False,
                reason="implemented_for_real_search",
            ),
            ConnectorRuntimeConfig(
                name="arxiv",
                available=True,
                requires_key=False,
                reason="implemented_for_real_search",
            ),
            ConnectorRuntimeConfig(
                name="semantic_scholar",
                available=True,
                requires_key=False,
                reason=_semantic_scholar_runtime_reason(),
            ),
            ConnectorRuntimeConfig(
                name="pubmed",
                available=True,
                requires_key=False,
                reason=_pubmed_runtime_reason(),
            ),
            _local_bm25_runtime_config(),
            _local_hybrid_runtime_config(),
        ],
        limits=RuntimeLimits(
            max_top_k=100,
            max_search_rounds=3,
            max_candidate_papers=300,
            max_latency_seconds=120,
            real_search_max_workers=_real_search_max_workers(),
            real_search_background_workers=_real_search_background_workers(),
            real_search_run_ttl_seconds=_real_search_run_ttl_seconds(),
            real_search_max_stored_runs=_real_search_max_stored_runs(),
        ),
        features=RuntimeFeatures(
            query_evolution=True,
            refchain=True,
            evaluation=False,
            sse=True,
            real_search=True,
            real_search_cancel=True,
            real_search_sse=True,
            retrieval_cache=True,
            batch_cli=True,
            llm_query_understanding=llm_feature_enabled,
            llm_judgement=llm_judgement_enabled,
        ),
    )


def _semantic_scholar_runtime_reason() -> str:
    if os.getenv(SEMANTIC_SCHOLAR_API_KEY_ENV, "").strip():
        return "implemented_for_real_search_with_optional_api_key"
    return "implemented_for_real_search_without_api_key_rate_limited"


def _pubmed_runtime_reason() -> str:
    if os.getenv(NCBI_API_KEY_ENV, "").strip() or os.getenv(PUBMED_API_KEY_ENV, "").strip():
        return "implemented_for_real_search_with_optional_api_key"
    return "implemented_for_real_search_without_api_key_rate_limited"


def _local_bm25_runtime_config() -> ConnectorRuntimeConfig:
    try:
        metadata = configure_local_bm25_from_env(build_index=False)
    except (OSError, ValueError) as exc:
        return ConnectorRuntimeConfig(
            name="local_bm25",
            available=False,
            requires_key=False,
            reason=f"local_bm25_unavailable:{type(exc).__name__}",
        )
    if metadata is None:
        return ConnectorRuntimeConfig(
            name="local_bm25",
            available=False,
            requires_key=False,
            reason="local_bm25_env_not_configured",
        )
    return ConnectorRuntimeConfig(
        name="local_bm25",
        available=True,
        requires_key=False,
        reason=f"configured_from_env:{metadata.document_count}_documents",
        details={
            "field_completeness": metadata.field_completeness,
            "corpus_sha256": metadata.corpus_sha256,
            "index_fingerprint": metadata.fingerprint,
            "metadata_quality_scope": "diagnostic_only",
        },
    )


def _local_hybrid_runtime_config() -> ConnectorRuntimeConfig:
    try:
        metadata = configure_local_hybrid_from_env(build_index=False)
    except (OSError, ValueError, RuntimeError) as exc:
        return ConnectorRuntimeConfig(
            name="local_hybrid",
            available=False,
            requires_key=False,
            reason=f"local_hybrid_unavailable:{type(exc).__name__}",
        )
    if metadata is None:
        return ConnectorRuntimeConfig(
            name="local_hybrid",
            available=False,
            requires_key=False,
            reason="local_hybrid_env_not_configured",
        )
    return ConnectorRuntimeConfig(
        name="local_hybrid",
        available=True,
        requires_key=False,
        reason=(
            "configured_from_env:"
            f"{metadata.document_count}_documents:"
            f"{metadata.abstract_document_count}_abstracts"
        ),
        details={
            "field_completeness": metadata.field_completeness,
            "corpus_sha256": metadata.semantic_corpus_sha256,
            "index_fingerprint": metadata.index_fingerprint,
            "model_fingerprint": metadata.model_fingerprint,
            "metadata_quality_scope": "diagnostic_only",
        },
    )


def _to_internal_constraints(constraints: SearchConstraints) -> QueryConstraint:
    raw_time_range = constraints.time_range
    time_range: TimeRange | None = None
    if raw_time_range is not None and (
        raw_time_range.start_year is not None
        or raw_time_range.end_year is not None
    ):
        time_range = raw_time_range.model_copy(
            update={"label": raw_time_range.label or "explicit"}
        )
    return QueryConstraint(
        time_range=time_range,
        venues=constraints.venues,
        datasets=constraints.datasets,
        must_include_terms=constraints.must_have_terms,
        exclude_terms=constraints.excluded_terms,
        paper_types=constraints.paper_types,
    )


def _to_internal_budget(budgets: BaseModel) -> SearchBudget:
    return SearchBudget.model_validate(budgets.model_dump())


@router.post(
    "/real/search/runs",
    response_model=SearchRunCreateResponse,
    status_code=201,
    tags=["real-search"],
)
def create_real_search_run(request: SearchRunCreateRequest) -> SearchRunCreateResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    _validate_explicit_local_source_configuration(request.source_preferences)

    _cleanup_real_runs()
    run_id = f"run_real_{uuid4().hex[:12]}"
    timestamp = _now()
    with _REAL_RUNS_LOCK:
        _REAL_RUNS[run_id] = RealRun(
            run_id=run_id,
            request=request,
            status="queued",
            current_stage="queued",
            progress=RunProgress(),
            cost_report=CostReport(),
            result=None,
            events=[],
            error_message=None,
            cancel_requested=False,
            created_at=timestamp,
            updated_at=timestamp,
        )
    _cleanup_real_runs(protected_run_id=run_id)
    _append_real_event(
        run_id,
        "run_started",
        {
            "query": request.query,
            "mode": "real_search",
            "status": "queued",
        },
    )
    _real_search_executor().submit(_execute_real_search_run, run_id)
    return SearchRunCreateResponse(
        run_id=run_id,
        status="queued",
        created_at=timestamp,
        links=_real_links(run_id),
    )


def _validate_explicit_local_source_configuration(
    sources: list[str] | None,
) -> None:
    """Fail fast for explicitly requested local sources with no usable config.

    This is a read-only environment check. Index construction remains in the
    background worker when configuration exists, so normal startup latency and
    the existing asynchronous API contract are unchanged.
    """

    if not sources:
        return
    checks = {
        "local_bm25": lambda: local_bm25_config_from_env(),
        "local_hybrid": lambda: local_hybrid_config_from_env(),
    }
    for source in (item for item in sources if item in checks):
        try:
            config = checks[source]()
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=409,
                detail=f"{source}_configuration_unavailable:{type(exc).__name__}",
            ) from exc
        if config is None:
            raise HTTPException(
                status_code=409,
                detail=f"{source}_not_configured",
            )
        if source == "local_bm25" and not config.corpus_path.is_file():
            raise HTTPException(
                status_code=409,
                detail="local_bm25_corpus_not_found",
            )
        if source == "local_hybrid":
            missing = (
                "local_hybrid_bm25_corpus_not_found"
                if not config.bm25_config.corpus_path.is_file()
                else "local_hybrid_semantic_corpus_not_found"
                if not config.semantic_corpus_path.is_file()
                else "local_hybrid_model_not_found"
                if not config.model_path.is_dir()
                else None
            )
            if missing is not None:
                raise HTTPException(status_code=409, detail=missing)


@router.get(
    "/real/search/runs/{run_id}",
    response_model=SearchRunStatusResponse,
    tags=["real-search"],
)
def get_real_search_run(run_id: str) -> SearchRunStatusResponse:
    _cleanup_real_runs(protected_run_id=run_id)
    with _REAL_RUNS_LOCK:
        run = _get_real_run(run_id)
        return _real_status_response(run)


@router.get(
    "/real/search/runs/{run_id}/result",
    response_model=SearchRunResultResponse,
    tags=["real-search"],
)
def get_real_search_result(run_id: str) -> SearchRunResultResponse:
    _cleanup_real_runs(protected_run_id=run_id)
    with _REAL_RUNS_LOCK:
        run = _get_real_run(run_id)
        if run.status in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="result not ready")
        if run.status == "cancelled":
            raise HTTPException(status_code=409, detail="run cancelled")
        if run.status == "failed":
            raise HTTPException(
                status_code=500,
                detail=run.error_message or "real search failed",
            )
        if run.result is None:
            raise HTTPException(status_code=409, detail="result not ready")
        return run.result


@router.post(
    "/real/search/runs/{run_id}/cancel",
    response_model=SearchRunStatusResponse,
    tags=["real-search"],
)
def cancel_real_search_run(run_id: str) -> SearchRunStatusResponse:
    _cleanup_real_runs(protected_run_id=run_id)
    with _REAL_RUNS_LOCK:
        run = _get_real_run(run_id)
        if run.status in {"queued", "running"}:
            run.status = "cancelled"
            run.current_stage = "cancelled"
            run.cancel_requested = True
            run.result = None
            run.error_message = "run cancelled"
            run.updated_at = _now()
            _append_real_event_locked(
                run,
                "run_cancelled",
                {"status": "cancelled", "reason": "user_requested"},
            )
            _append_terminal_event_locked(run, "cancelled")
        return _real_status_response(run)


@router.get("/real/search/runs/{run_id}/events", tags=["real-search"])
def stream_real_search_events(run_id: str) -> StreamingResponse:
    _cleanup_real_runs(protected_run_id=run_id)
    _get_real_run(run_id)

    async def event_generator():
        event_index = 0
        while True:
            with _REAL_RUNS_LOCK:
                run = _REAL_RUNS.get(run_id)
                if run is None:
                    break
                pending_events = run.events[event_index:]
                event_count = len(run.events)
                terminal = run.terminal_event_emitted

            for event_name, payload in pending_events:
                yield f"event: {event_name}\n"
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            event_index += len(pending_events)
            if terminal and event_index >= event_count:
                break
            await asyncio.sleep(0.05)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/internal/search/preview",
    response_model=InternalSearchPreviewResponse,
    tags=["internal-preview"],
)
def internal_search_preview(
    request: InternalSearchPreviewRequest,
) -> InternalSearchPreviewResponse:
    try:
        output = _preview_search_service().run_search(
            request.query,
            top_k=request.top_k,
            run_profile=request.run_profile,
            enable_refchain=request.enable_refchain,
            **(
                {"enable_semantic_seed_expansion": True}
                if request.enable_semantic_seed_expansion
                else {}
            ),
            enable_query_evolution=request.enable_query_evolution,
            query_evolution_policy=request.query_evolution_policy,
            query_planning_policy=request.query_planning_policy,
            **(
                {"ranking_policy": request.ranking_policy}
                if request.ranking_policy != "current_rules"
                else {}
            ),
            **(
                {"judgement_policy": request.judgement_policy}
                if request.judgement_policy != "current_rules"
                else {}
            ),
            enable_llm_query_understanding=request.enable_llm_query_understanding,
            enable_llm_judgement=request.enable_llm_judgement,
            current_year=request.current_year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return InternalSearchPreviewResponse(
        query_analysis=_model_dump(output.search_plan.query_analysis),
        search_plan=_model_dump(output.search_plan),
        query_evolution_records=[
            _model_dump(record) for record in output.query_evolution_records
        ],
        refchain_output=(
            _model_dump(output.refchain_output)
            if output.refchain_output is not None
            else None
        ),
        semantic_seed_expansion_output=(
            _model_dump(output.semantic_seed_expansion_output)
            if output.semantic_seed_expansion_output is not None
            else None
        ),
        synthesis_output=(
            _model_dump(output.synthesis_output)
            if output.synthesis_output is not None
            else None
        ),
        ranked_papers=[_model_dump(paper) for paper in output.ranked_papers],
        raw_count=output.raw_count,
        deduplicated_count=output.deduplicated_count,
        warnings=output.warnings,
        source_stats=[_model_dump(stats) for stats in output.source_stats],
        latency_seconds=output.latency_seconds,
    )


@router.post(
    "/internal/search/preview/api-result",
    response_model=SearchRunResultResponse,
    tags=["internal-preview"],
)
def internal_search_preview_api_result(
    request: InternalSearchPreviewRequest,
) -> SearchRunResultResponse:
    try:
        output = _preview_search_service().run_search(
            request.query,
            top_k=request.top_k,
            run_profile=request.run_profile,
            enable_refchain=request.enable_refchain,
            **(
                {"enable_semantic_seed_expansion": True}
                if request.enable_semantic_seed_expansion
                else {}
            ),
            enable_query_evolution=request.enable_query_evolution,
            query_evolution_policy=request.query_evolution_policy,
            query_planning_policy=request.query_planning_policy,
            **(
                {"ranking_policy": request.ranking_policy}
                if request.ranking_policy != "current_rules"
                else {}
            ),
            **(
                {"judgement_policy": request.judgement_policy}
                if request.judgement_policy != "current_rules"
                else {}
            ),
            enable_llm_query_understanding=request.enable_llm_query_understanding,
            enable_llm_judgement=request.enable_llm_judgement,
            current_year=request.current_year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return map_search_service_output_to_api_result(
        run_id=f"run_internal_{uuid4().hex[:12]}",
        output=output,
        status="succeeded",
        partial=False,
    )

def _execute_real_search_run(run_id: str) -> None:
    with _REAL_RUNS_LOCK:
        run = _REAL_RUNS.get(run_id)
        if run is None:
            return
        if run.status == "cancelled" or run.cancel_requested:
            return
        run.status = "running"
        run.current_stage = "starting"
        run.updated_at = _now()
        request = run.request

    try:
        output = _real_search_service().run_search(
            request.query,
            top_k=request.top_k,
            run_profile=request.run_profile,
            enable_refchain=request.options.enable_refchain,
            **(
                {"enable_semantic_seed_expansion": True}
                if request.options.enable_semantic_seed_expansion
                else {}
            ),
            enable_query_evolution=request.options.enable_query_evolution,
            query_evolution_policy=request.options.query_evolution_policy,
            query_planning_policy=request.options.query_planning_policy,
            **(
                {"ranking_policy": request.options.ranking_policy}
                if request.options.ranking_policy != "current_rules"
                else {}
            ),
            **(
                {"judgement_policy": request.options.judgement_policy}
                if request.options.judgement_policy != "current_rules"
                else {}
            ),
            enable_synthesis=True,
            current_year=None,
            enable_llm_query_understanding=(
                request.options.enable_llm_query_understanding
            ),
            enable_llm_judgement=request.options.enable_llm_judgement,
            sources_override=request.source_preferences,
            explicit_constraints=_to_internal_constraints(request.constraints),
            budget=_to_internal_budget(request.budgets),
            event_callback=lambda event_name, payload: _handle_real_search_event(
                run_id,
                event_name,
                payload,
            ),
            should_cancel=lambda: _real_run_is_cancelled(run_id),
        )
        result = map_search_service_output_to_api_result(
            run_id=run_id,
            output=output,
            status="succeeded",
            partial=False,
        )
        with _REAL_RUNS_LOCK:
            run = _REAL_RUNS.get(run_id)
            if run is None:
                return
            if run.status == "cancelled" or run.cancel_requested:
                return
            run.status = "succeeded"
            run.cost_report = result.cost_report
            run.result = result
            run.error_message = None
            run.updated_at = _now()
            _append_real_event_locked(
                run,
                "cost_updated",
                {"cost_report": _model_dump(result.cost_report)},
            )
            _append_terminal_event_locked(
                run,
                "succeeded",
                {"cost_report": _model_dump(result.cost_report)},
            )
    except SearchCancelled as exc:
        _cancel_real_run_from_worker(run_id, exc.stage)
    except ValueError as exc:
        _fail_real_run(run_id, str(exc))
    except Exception as exc:  # noqa: BLE001 - isolate background failure
        _fail_real_run(run_id, str(exc))


def _handle_real_search_event(
    run_id: str,
    event_name: str,
    payload: dict[str, Any],
) -> None:
    with _REAL_RUNS_LOCK:
        run = _REAL_RUNS.get(run_id)
        if run is None or run.cancel_requested or run.status == "cancelled":
            return
        stage = str(payload.get("stage") or _event_stage(event_name) or "running")
        if event_name == "connector_started":
            run.status = "running"
            run.current_stage = "retrieval"
        elif event_name == "connector_completed":
            run.status = "running"
            run.current_stage = "retrieval"
        elif event_name.endswith("_started"):
            run.status = "running"
            run.current_stage = stage
        elif event_name.endswith("_completed"):
            run.status = "running"
            run.current_stage = stage
            if stage not in run.progress.completed_stages:
                run.progress.completed_stages.append(stage)
        elif event_name.endswith("_skipped"):
            if stage not in run.progress.skipped_stages:
                run.progress.skipped_stages.append(stage)

        candidate_count = payload.get("deduplicated_candidate_count")
        if candidate_count is None:
            candidate_count = payload.get("candidate_paper_count")
        if isinstance(candidate_count, int):
            run.progress.candidate_paper_count = max(0, candidate_count)
        judged_count = payload.get("judged_paper_count")
        if isinstance(judged_count, int):
            run.progress.judged_paper_count = max(0, judged_count)

        if event_name == "connector_completed":
            is_reference = stage == "refchain" or payload.get("source") == "refchain"
            request_count = _nonnegative_int(payload.get("request_count"))
            retry_count = _nonnegative_int(payload.get("retry_count"))
            error_count = _nonnegative_int(payload.get("error_count"))
            cache_hit_count = _nonnegative_int(payload.get("cache_hit_count"))
            wait_seconds = _nonnegative_float(
                payload.get("rate_limit_wait_seconds")
            )
            if is_reference:
                run.cost_report.reference_api_call_count += request_count
            else:
                run.cost_report.logical_search_call_count += 1
                run.cost_report.search_api_call_count += request_count
            run.cost_report.retry_count += retry_count
            run.cost_report.error_count += error_count
            run.cost_report.cache_hit_count += cache_hit_count
            run.cost_report.rate_limit_wait_seconds += wait_seconds
            run.cost_report.api_call_count = (
                run.cost_report.search_api_call_count
                + run.cost_report.reference_api_call_count
                + run.cost_report.llm_call_count
            )

        _append_real_event_locked(run, event_name, payload)


def _real_run_is_cancelled(run_id: str) -> bool:
    with _REAL_RUNS_LOCK:
        run = _REAL_RUNS.get(run_id)
        if run is None:
            return True
        return run.status == "cancelled" or run.cancel_requested


def _append_real_event(
    run_id: str,
    event_name: str,
    payload: dict[str, Any] | None = None,
) -> None:
    with _REAL_RUNS_LOCK:
        run = _REAL_RUNS.get(run_id)
        if run is None:
            return
        _append_real_event_locked(run, event_name, payload)


def _append_real_event_locked(
    run: RealRun,
    event_name: str,
    payload: dict[str, Any] | None = None,
) -> None:
    event_payload = dict(payload or {})
    event_payload.setdefault("run_id", run.run_id)
    event_payload.setdefault("timestamp", _now().isoformat())
    run.events.append((event_name, event_payload))
    run.updated_at = _now()


def _append_terminal_event_locked(
    run: RealRun,
    status: str,
    payload: dict[str, Any] | None = None,
) -> bool:
    if run.terminal_event_emitted:
        return False
    run.terminal_event_emitted = True
    terminal_payload = {"status": status, **dict(payload or {})}
    _append_real_event_locked(run, "run_completed", terminal_payload)
    return True


def _real_status_response(run: RealRun) -> SearchRunStatusResponse:
    return SearchRunStatusResponse(
        run_id=run.run_id,
        status=run.status,  # type: ignore[arg-type]
        current_stage=run.current_stage,
        progress=run.progress.model_copy(deep=True),
        cost_report=run.cost_report.model_copy(deep=True),
        error_message=run.error_message,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _cancel_real_run_from_worker(run_id: str, stage: str) -> None:
    with _REAL_RUNS_LOCK:
        run = _REAL_RUNS.get(run_id)
        if run is None:
            return
        if run.status != "cancelled":
            run.status = "cancelled"
            run.current_stage = "cancelled"
            run.cancel_requested = True
            run.result = None
            run.error_message = "run cancelled"
            _append_real_event_locked(
                run,
                "run_cancelled",
                {"status": "cancelled", "reason": "cooperative", "stage": stage},
            )
        _append_terminal_event_locked(run, "cancelled")


def _event_stage(event_name: str) -> str | None:
    for suffix in ("_started", "_completed", "_skipped"):
        if event_name.endswith(suffix):
            return event_name[: -len(suffix)]
    return None


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _fail_real_run(run_id: str, message: str) -> None:
    with _REAL_RUNS_LOCK:
        run = _REAL_RUNS.get(run_id)
        if run is None:
            return
        if run.status == "cancelled" or run.cancel_requested:
            return
        run.status = "failed"
        run.current_stage = "failed"
        run.error_message = message
        run.cost_report = CostReport()
        run.updated_at = _now()
        _append_real_event_locked(run, "error", {"message": message})
        _append_terminal_event_locked(
            run,
            "failed",
            {"error": message},
        )
