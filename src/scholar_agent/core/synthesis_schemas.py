"""Schemas for citation-backed answer synthesis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from scholar_agent.core.paper_schemas import PaperIdentifiers
from scholar_agent.core.search_schemas import EvidenceSource


class SynthesisOptions(BaseModel):
    max_cited_papers: int = Field(default=8, ge=0, le=50)
    max_evidence_rows_per_paper: int = Field(default=3, ge=1, le=20)
    max_findings: int = Field(default=5, ge=0, le=20)
    evidence_snippet_chars: int = Field(default=240, ge=40, le=1000)


class SynthesisEvidenceRow(BaseModel):
    row_id: str
    citation_key: str
    rank: int = Field(ge=1)
    paper_title: str
    year: int | None = None
    venue: str | None = None
    sources: list[str] = Field(default_factory=list)
    identifiers: PaperIdentifiers = Field(default_factory=PaperIdentifiers)
    category: str
    final_score: float = Field(ge=0.0, le=1.0)
    evidence_source: EvidenceSource
    evidence_text: str
    supported_terms: list[str] = Field(default_factory=list)
    supported_claim: str


class SynthesisFinding(BaseModel):
    text: str
    citation_keys: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_row_ids: list[str] = Field(default_factory=list)


class CitationCoverage(BaseModel):
    ranked_paper_count: int = Field(default=0, ge=0)
    cited_paper_count: int = Field(default=0, ge=0)
    evidence_row_count: int = Field(default=0, ge=0)
    cited_evidence_row_count: int = Field(default=0, ge=0)
    missing_evidence_count: int = Field(default=0, ge=0)
    source_error_count: int = Field(default=0, ge=0)
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class SynthesisOutput(BaseModel):
    answer_summary: str
    status: Literal["succeeded", "insufficient_evidence"] = "succeeded"
    key_findings: list[SynthesisFinding] = Field(default_factory=list)
    evidence_table: list[SynthesisEvidenceRow] = Field(default_factory=list)
    citation_coverage: CitationCoverage = Field(default_factory=CitationCoverage)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
