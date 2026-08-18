"""Shared paper schemas for retrieval connectors."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PaperIdentifiers(BaseModel):
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    s2orc_corpus_id: str | None = None
    openalex_id: str | None = None
    pubmed_id: str | None = None


class PaperUrls(BaseModel):
    landing_page: str | None = None
    pdf: str | None = None


class Paper(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str = ""
    identifiers: PaperIdentifiers = Field(default_factory=PaperIdentifiers)
    urls: PaperUrls = Field(default_factory=PaperUrls)
    sources: list[str] = Field(default_factory=list)
    citation_count: int = 0
