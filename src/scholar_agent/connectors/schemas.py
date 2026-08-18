"""Shared connector result schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper


class ConnectorSearchResult(BaseModel):
    papers: list[Paper] = Field(default_factory=list)
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    latency_seconds: float = 0.0
    diagnostics: ConnectorDiagnostics = Field(default_factory=ConnectorDiagnostics)
    snapshot_provenance: Literal[
        "live", "snapshot_record", "snapshot_replay", "snapshot_plan"
    ] = "live"
    snapshot_key: str | None = None
    snapshot_hit: bool = False
    recorded_diagnostics: ConnectorDiagnostics | None = None
    recorded_latency_seconds: float = 0.0
    reference_batch_status: Literal[
        "success", "partial_success", "missing_id", "failed"
    ] | None = None
    missing_reference_ids: list[str] = Field(default_factory=list)
    reference_batch_count: int = Field(default=0, ge=0)
    supplemental_request_count: int = Field(default=0, ge=0)


class ConnectorRequestSpec(BaseModel):
    """Credential-free description shared by live requests and offline audits."""

    source: str
    adapted_query: str
    endpoint_alias: str
    method: Literal["GET", "POST"]
    parameters: dict[str, str]
    timeout_seconds: float = Field(gt=0.0)
    max_retries: int = Field(ge=0)
    auth_scope_alias: str
    auth_affects_response_semantics: bool = False
    accept: str
    response_media_type: str
    page_budget: int = Field(ge=1)
    pagination_strategy: str
    response_dependent_children: list[dict[str, object]] = Field(
        default_factory=list
    )
