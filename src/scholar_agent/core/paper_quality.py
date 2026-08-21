"""Offline, explainable paper-quality signals independent from relevance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scholar_agent.core.identity import paper_identifier_set
from scholar_agent.core.paper_schemas import Paper


QualitySignalState = Literal["present", "missing", "unknown"]


class QualitySignal(BaseModel):
    """One bounded, source-local quality observation, never a relevance label."""

    model_config = ConfigDict(extra="forbid")

    name: str
    state: QualitySignalState
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    detail: str


class PaperQualityReport(BaseModel):
    """Deterministic report for a paper without external risk assertions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["paper-quality-v1"] = "paper-quality-v1"
    quality_score: float = Field(ge=0.0, le=1.0)
    signals: list[QualitySignal]
    external_risk_data_available: bool = False
    quality_scope: Literal["metadata_only_not_relevance_or_retraction_claim"] = (
        "metadata_only_not_relevance_or_retraction_claim"
    )


def assess_paper_quality(paper: Paper) -> PaperQualityReport:
    """Build a local metadata-quality report without filtering a paper.

    The report contains only facts available on ``Paper``. Retraction and
    duplicate-risk fields deliberately remain ``unknown`` until an independent
    authoritative source is integrated by a later task.
    """

    completeness = _metadata_completeness(paper)
    unique_sources = len({source.strip().casefold() for source in paper.sources if source.strip()})
    source_score = min(unique_sources / 2, 1.0)
    identifier_count = len(paper_identifier_set(paper))
    identifier_score = min(identifier_count / 3, 1.0)
    full_text_count = len(paper.full_text_evidence)
    signals = [
        _known_signal("metadata_completeness", completeness, "title_abstract_year_venue"),
        _known_signal("source_corroboration", source_score, f"unique_sources:{unique_sources}"),
        _known_signal("stable_identifier_coverage", identifier_score, f"stable_identifiers:{identifier_count}"),
        _known_signal(
            "licensed_full_text_evidence",
            1.0 if full_text_count else 0.0,
            f"evidence_documents:{full_text_count}",
        ),
        QualitySignal(
            name="retraction_status",
            state="unknown",
            detail="external_retraction_source_not_checked",
        ),
        QualitySignal(
            name="duplicate_risk",
            state="unknown",
            detail="external_duplicate_source_not_checked",
        ),
    ]
    # Only local, directly observable signals contribute. Unknown external risk
    # states are intentionally excluded rather than treated as a penalty.
    score = sum(signal.value or 0.0 for signal in signals[:4]) / 4
    return PaperQualityReport(quality_score=score, signals=signals)


def _metadata_completeness(paper: Paper) -> float:
    values = (
        bool(paper.title.strip()),
        bool(paper.abstract.strip()),
        paper.year is not None,
        bool((paper.venue or "").strip()),
    )
    return sum(values) / len(values)


def _known_signal(name: str, value: float, detail: str) -> QualitySignal:
    return QualitySignal(
        name=name,
        state="present" if value > 0 else "missing",
        value=value,
        detail=detail,
    )
