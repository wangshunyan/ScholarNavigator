"""Offline, explainable paper-quality signals independent from relevance."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scholar_agent.core.identity import paper_identifier_set
from scholar_agent.core.paper_schemas import Paper


QualitySignalState = Literal["present", "missing", "unknown"]
QualityRiskSignal = Literal["retraction_status", "duplicate_risk"]
VerifiedRiskState = Literal["clear", "flagged"]


class VerifiedQualityEvidence(BaseModel):
    """Caller-supplied evidence envelope; this model performs no network I/O."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["paper-quality-evidence-v1"] = (
        "paper-quality-evidence-v1"
    )
    paper_identifier: str = Field(min_length=1)
    signal_name: QualityRiskSignal
    state: VerifiedRiskState
    source: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)


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
    quality_scope: Literal[
        "metadata_only_not_relevance_or_retraction_claim",
        "metadata_plus_verified_risk_not_relevance_claim",
    ] = (
        "metadata_only_not_relevance_or_retraction_claim"
    )


def assess_paper_quality(
    paper: Paper,
    *,
    verified_evidence: Sequence[VerifiedQualityEvidence] = (),
) -> PaperQualityReport:
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
    signals: list[QualitySignal] = [
        _known_signal("metadata_completeness", completeness, "title_abstract_year_venue"),
        _known_signal("source_corroboration", source_score, f"unique_sources:{unique_sources}"),
        _known_signal("stable_identifier_coverage", identifier_score, f"stable_identifiers:{identifier_count}"),
        _known_signal(
            "licensed_full_text_evidence",
            1.0 if full_text_count else 0.0,
            f"evidence_documents:{full_text_count}",
        ),
    ]
    evidence_by_signal = _matching_verified_evidence(paper, verified_evidence)
    for name in ("retraction_status", "duplicate_risk"):
        evidence = evidence_by_signal.get(name)
        if evidence is None:
            signals.append(
                QualitySignal(
                    name=name,
                    state="unknown",
                    detail=f"external_{name}_source_not_checked",
                )
            )
        else:
            signals.append(
                QualitySignal(
                    name=name,
                    state="present" if evidence.state == "clear" else "missing",
                    value=1.0 if evidence.state == "clear" else 0.0,
                    detail=(
                        f"verified_source:{evidence.source};"
                        f"record_sha256:{_digest(evidence.source_record_id)}"
                    ),
                )
            )
    known = [signal for signal in signals if signal.value is not None]
    score = sum(signal.value or 0.0 for signal in known) / len(known)
    return PaperQualityReport(
        quality_score=score,
        signals=signals,
        external_risk_data_available=bool(evidence_by_signal),
        quality_scope=(
            "metadata_plus_verified_risk_not_relevance_claim"
            if evidence_by_signal
            else "metadata_only_not_relevance_or_retraction_claim"
        ),
    )


def _matching_verified_evidence(
    paper: Paper,
    evidence: Sequence[VerifiedQualityEvidence],
) -> dict[str, VerifiedQualityEvidence]:
    identifiers = paper_identifier_set(paper)
    matched: dict[str, VerifiedQualityEvidence] = {}
    for item in evidence:
        if item.paper_identifier not in identifiers:
            continue
        previous = matched.get(item.signal_name)
        if previous is not None and previous.state != item.state:
            raise ValueError("conflicting_verified_quality_evidence")
        matched[item.signal_name] = item
    return matched


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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
