"""SearchService 可选阶段快照与候选来源追踪。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from scholar_agent.agents.retriever import (
    QueryAdaptationProvenance,
    RetrievalOutput,
)
from scholar_agent.core.dedup import deduplicate_papers
from scholar_agent.core.identity import (
    IdentityProfile,
    build_identity_profile,
    identity_evidence_from_profiles,
)
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.search_schemas import (
    JudgementFeatureVector,
    JudgementResult,
    RankedPaper,
)


SnapshotStatus = Literal["completed", "skipped"]
OriginKind = Literal[
    "initial_query",
    "initial_generated_subquery",
    "query_evolution",
    "refchain",
    "semantic_seed_expansion",
]


class CandidateProvenance(BaseModel):
    origin_kind: OriginKind
    origin_stage: str
    origin_subquery: str
    source: str
    adapted_query: str | None = None
    adaptation_strategy: str | None = None
    purpose: str | None = None
    cache_hit: bool = False
    source_skipped_reason: str | None = None
    source_rank: int | None = Field(default=None, ge=1)


class RetrievalCallTrace(BaseModel):
    origin_subquery: str
    source: str
    terminal_status: str | None = None
    adapted_query: str | None = None
    adaptation_strategy: str | None = None
    cache_hit: bool = False
    run_dedupe_hit: bool = False
    logical_call_executed: bool = True
    triggered_by: list[str] = Field(default_factory=list)
    safe_original_candidate_count: int | None = None
    safe_original_core_term_coverage: float | None = None
    safe_original_constraint_coverage: float | None = None
    sufficiency_reasons: list[str] = Field(default_factory=list)
    compact_query_executed: bool | None = None
    compact_query_skipped_reason: str | None = None
    source_skipped_reason: str | None = None
    remaining_subquery_count: int = 0
    returned_count: int = 0
    request_count: int = 0
    error_count: int = 0
    snapshot_provenance: str = "live"
    snapshot_key: str | None = None
    snapshot_hit: bool = False
    recorded_request_count: int = 0
    recorded_retry_count: int = 0
    recorded_error_count: int = 0
    recorded_rate_limit_wait_seconds: float = 0.0
    recorded_latency_seconds: float = 0.0
    query_provenance: list[QueryAdaptationProvenance] = Field(default_factory=list)


class DiagnosticCandidate(BaseModel):
    identifiers: PaperIdentifiers = Field(default_factory=PaperIdentifiers)
    title: str
    year: int | None = None
    sources: list[str] = Field(default_factory=list)
    provenance: list[CandidateProvenance] = Field(default_factory=list)
    rank: int | None = None
    judgement_score: float | None = None
    category: str | None = None
    final_score: float | None = None
    matched_terms: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    judgement_features: JudgementFeatureVector | None = None
    rrf_score: float | None = Field(default=None, ge=0.0)
    rrf_contributions: list[dict[str, object]] = Field(default_factory=list)
    original_rank: int | None = Field(default=None, ge=1)
    rrf_top_20_change: str | None = None
    rrf_rank_change_reason: str | None = None


class StageCandidateSnapshot(BaseModel):
    stage: str
    status: SnapshotStatus = "completed"
    skipped_reason: str | None = None
    candidates: list[DiagnosticCandidate] = Field(default_factory=list)
    retrieval_calls: list[RetrievalCallTrace] = Field(default_factory=list)
    identity_audit: list[dict[str, object]] = Field(default_factory=list)


@dataclass
class _TrackedCandidate:
    paper: Paper
    provenance: list[CandidateProvenance] = field(default_factory=list)


@dataclass
class _TrackedJudgement:
    paper: Paper
    score: float


class PipelineDiagnosticsCollector:
    """Collect compact snapshots without changing retrieval or ranking decisions."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.snapshots: list[StageCandidateSnapshot] = []
        self._tracked: list[_TrackedCandidate] = []
        self._tracked_profiles: list[IdentityProfile] = []
        self._judgements: list[_TrackedJudgement] = []
        self._judgement_profiles: list[IdentityProfile] = []

    def register_retrieval(
        self,
        stage: str,
        outputs: list[RetrievalOutput],
        *,
        origin_kind_by_query: dict[str, OriginKind],
    ) -> None:
        if not self.enabled:
            return
        papers: list[Paper] = []
        retrieval_calls: list[RetrievalCallTrace] = []
        for output in outputs:
            origin_kind = origin_kind_by_query.get(
                output.query,
                "initial_generated_subquery",
            )
            traced_paper = False
            for stats in output.source_stats:
                retrieval_calls.append(
                    RetrievalCallTrace(
                        origin_subquery=output.query,
                        source=stats.source,
                        terminal_status=stats.terminal_status,
                        adapted_query=stats.adapted_query,
                        adaptation_strategy=stats.adaptation_strategy,
                        cache_hit=stats.cache_hit,
                        run_dedupe_hit=stats.run_dedupe_hit,
                        logical_call_executed=stats.logical_call_executed,
                        triggered_by=list(stats.triggered_by),
                        safe_original_candidate_count=(
                            stats.safe_original_candidate_count
                        ),
                        safe_original_core_term_coverage=(
                            stats.safe_original_core_term_coverage
                        ),
                        safe_original_constraint_coverage=(
                            stats.safe_original_constraint_coverage
                        ),
                        sufficiency_reasons=list(stats.sufficiency_reasons),
                        compact_query_executed=stats.compact_query_executed,
                        compact_query_skipped_reason=(
                            stats.compact_query_skipped_reason
                        ),
                        source_skipped_reason=stats.source_skipped_reason,
                        remaining_subquery_count=stats.remaining_subquery_count,
                        returned_count=stats.returned_count,
                        request_count=stats.diagnostics.request_count,
                        error_count=stats.diagnostics.error_count,
                        snapshot_provenance=stats.snapshot_provenance,
                        snapshot_key=stats.snapshot_key,
                        snapshot_hit=stats.snapshot_hit,
                        recorded_request_count=(
                            stats.recorded_diagnostics.request_count
                            if stats.recorded_diagnostics is not None
                            else 0
                        ),
                        recorded_retry_count=(
                            stats.recorded_diagnostics.retry_count
                            if stats.recorded_diagnostics is not None
                            else 0
                        ),
                        recorded_error_count=(
                            stats.recorded_diagnostics.error_count
                            if stats.recorded_diagnostics is not None
                            else 0
                        ),
                        recorded_rate_limit_wait_seconds=(
                            stats.recorded_diagnostics.rate_limit_wait_seconds
                            if stats.recorded_diagnostics is not None
                            else 0.0
                        ),
                        recorded_latency_seconds=stats.recorded_latency_seconds,
                        query_provenance=list(stats.query_provenance),
                    )
                )
                for source_rank, paper in enumerate(
                    stats.diagnostic_papers, start=1
                ):
                    traced_paper = True
                    logical_provenance = stats.query_provenance or [
                        QueryAdaptationProvenance(
                            origin_subquery=output.query,
                            adaptation_strategy=stats.adaptation_strategy or "unknown",
                        )
                    ]
                    for query_provenance in logical_provenance:
                        self._register(
                            paper,
                            CandidateProvenance(
                                origin_kind=origin_kind_by_query.get(
                                    query_provenance.origin_subquery,
                                    origin_kind,
                                ),
                                origin_stage=stage,
                                origin_subquery=query_provenance.origin_subquery,
                                source=stats.source,
                                adapted_query=stats.adapted_query,
                                adaptation_strategy=(
                                    query_provenance.adaptation_strategy
                                ),
                                purpose=query_provenance.purpose,
                                cache_hit=stats.cache_hit,
                                source_skipped_reason=(
                                    stats.source_skipped_reason
                                    if query_provenance.origin_subquery == output.query
                                    else None
                                ),
                                source_rank=source_rank,
                            ),
                        )
                    papers.append(paper)
            if not traced_paper:
                for paper in output.papers:
                    sources = _stable_strings(
                        paper.sources or output.requested_sources
                    )
                    for source in sources:
                        self._register(
                            paper,
                            CandidateProvenance(
                                origin_kind=origin_kind,
                                origin_stage=stage,
                                origin_subquery=output.query,
                                source=source,
                            ),
                        )
                    papers.append(paper)
        self.snapshots.append(
            StageCandidateSnapshot(
                stage=stage,
                candidates=[self._paper_candidate(paper) for paper in papers],
                retrieval_calls=retrieval_calls,
            )
        )

    def register_refchain(
        self,
        stage: str,
        papers: list[Paper],
    ) -> None:
        if not self.enabled:
            return
        for paper in papers:
            for source in _stable_strings(paper.sources or ["openalex"]):
                self._register(
                    paper,
                    CandidateProvenance(
                        origin_kind="refchain",
                        origin_stage=stage,
                        origin_subquery="refchain",
                        source=source,
                    ),
                )
        self.snapshot_papers(stage, papers)

    def register_semantic_seed_expansion(
        self,
        stage: str,
        papers: list[Paper],
    ) -> None:
        if not self.enabled:
            return
        for rank, paper in enumerate(papers, start=1):
            self._register(
                paper,
                CandidateProvenance(
                    origin_kind="semantic_seed_expansion",
                    origin_stage=stage,
                    origin_subquery="semantic_seed_expansion",
                    source="semantic_scholar",
                    source_rank=rank,
                ),
            )
        self.snapshot_papers(stage, papers)

    def snapshot_papers(
        self,
        stage: str,
        papers: list[Paper],
        *,
        identity_audit: list[dict[str, object]] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.snapshots.append(
            StageCandidateSnapshot(
                stage=stage,
                candidates=[self._paper_candidate(paper) for paper in papers],
                identity_audit=identity_audit or [],
            )
        )

    def snapshot_judgements(
        self,
        stage: str,
        judgements: list[JudgementResult],
    ) -> None:
        if not self.enabled:
            return
        candidates: list[DiagnosticCandidate] = []
        for judgement in judgements:
            self._judgements.append(
                _TrackedJudgement(
                    paper=judgement.paper.model_copy(deep=True),
                    score=judgement.score,
                )
            )
            self._judgement_profiles.append(build_identity_profile(judgement.paper))
            base = self._paper_candidate(judgement.paper)
            candidates.append(
                base.model_copy(
                    update={
                        "judgement_score": judgement.score,
                        "category": judgement.category,
                        "matched_terms": list(judgement.matched_terms),
                        "warnings": list(judgement.warnings),
                        "judgement_features": judgement.feature_vector,
                    }
                )
            )
        self.snapshots.append(StageCandidateSnapshot(stage=stage, candidates=candidates))

    def snapshot_ranked(
        self,
        stage: str,
        ranked_papers: list[RankedPaper],
    ) -> None:
        if not self.enabled:
            return
        candidates: list[DiagnosticCandidate] = []
        for ranked in ranked_papers:
            ranked_profile = build_identity_profile(ranked.paper)
            base = self._paper_candidate(ranked.paper, profile=ranked_profile)
            judgement_score = next(
                (
                    self._judgements[index].score
                    for index in range(len(self._judgements) - 1, -1, -1)
                    if identity_evidence_from_profiles(
                        self._judgement_profiles[index], ranked_profile
                    ).equivalent
                ),
                None,
            )
            candidates.append(
                base.model_copy(
                    update={
                        "rank": ranked.rank,
                        "judgement_score": judgement_score,
                        "category": ranked.category,
                        "final_score": ranked.final_score,
                        "matched_terms": list(ranked.matched_terms),
                        "warnings": list(ranked.warnings),
                        "rrf_score": ranked.rrf_score,
                        "rrf_contributions": [
                            item.model_dump(mode="json")
                            for item in ranked.rrf_contributions
                        ],
                        "original_rank": ranked.original_rank,
                        "rrf_top_20_change": ranked.rrf_top_20_change,
                        "rrf_rank_change_reason": ranked.rrf_rank_change_reason,
                    }
                )
            )
        self.snapshots.append(StageCandidateSnapshot(stage=stage, candidates=candidates))

    def skip(self, stage: str, reason: str) -> None:
        if not self.enabled:
            return
        self.snapshots.append(
            StageCandidateSnapshot(
                stage=stage,
                status="skipped",
                skipped_reason=reason,
            )
        )

    def _register(self, paper: Paper, provenance: CandidateProvenance) -> None:
        profile = build_identity_profile(paper)
        for index, tracked in enumerate(self._tracked):
            if not identity_evidence_from_profiles(
                self._tracked_profiles[index], profile
            ).equivalent:
                continue
            tracked.paper = deduplicate_papers([tracked.paper, paper])[0]
            self._tracked_profiles[index] = build_identity_profile(tracked.paper)
            if provenance not in tracked.provenance:
                tracked.provenance.append(provenance)
            return
        self._tracked.append(
            _TrackedCandidate(
                paper=paper.model_copy(deep=True),
                provenance=[provenance],
            )
        )
        self._tracked_profiles.append(profile)

    def _paper_candidate(
        self, paper: Paper, *, profile: IdentityProfile | None = None
    ) -> DiagnosticCandidate:
        provenance: list[CandidateProvenance] = []
        profile = profile or build_identity_profile(paper)
        for index, tracked in enumerate(self._tracked):
            if identity_evidence_from_profiles(
                self._tracked_profiles[index], profile
            ).equivalent:
                provenance.extend(tracked.provenance)
        provenance = _stable_provenance(provenance)
        return DiagnosticCandidate(
            identifiers=paper.identifiers.model_copy(deep=True),
            title=paper.title,
            year=paper.year,
            sources=_stable_strings(
                [*paper.sources, *(item.source for item in provenance)]
            ),
            provenance=provenance,
        )


def _same_candidate(left: Paper, right: Paper) -> bool:
    return len(deduplicate_papers([left, right])) == 1


def _stable_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip().casefold()
        if not key or key in seen:
            continue
        result.append(value)
        seen.add(key)
    return result


def _stable_provenance(
    values: list[CandidateProvenance],
) -> list[CandidateProvenance]:
    result: list[CandidateProvenance] = []
    seen: set[
        tuple[
            str,
            str,
            str,
            str,
            str | None,
            str | None,
            str | None,
            bool,
            str | None,
            int | None,
        ]
    ] = set()
    for value in values:
        key = (
            value.origin_kind,
            value.origin_stage,
            value.origin_subquery,
            value.source,
            value.adapted_query,
            value.adaptation_strategy,
            value.purpose,
            value.cache_hit,
            value.source_skipped_reason,
            value.source_rank,
        )
        if key in seen:
            continue
        result.append(value)
        seen.add(key)
    return result
