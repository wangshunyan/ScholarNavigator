"""统一的外部连接器请求诊断结构。"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field


class ConnectorDiagnostics(BaseModel):
    """一次或多次 connector 操作的真实请求与等待统计。"""

    request_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    cache_hit_count: int = Field(default=0, ge=0)
    rate_limit_wait_seconds: float = Field(default=0.0, ge=0.0)
    retry_after_seconds: float | None = Field(default=None, ge=0.0, exclude=True)
    latency_seconds: float = Field(default=0.0, ge=0.0)


def merge_connector_diagnostics(
    diagnostics: Iterable[ConnectorDiagnostics],
) -> ConnectorDiagnostics:
    """逐字段求和，避免以调用记录条数推断真实 HTTP 请求数。"""

    items = list(diagnostics)
    return ConnectorDiagnostics(
        request_count=sum(item.request_count for item in items),
        retry_count=sum(item.retry_count for item in items),
        error_count=sum(item.error_count for item in items),
        cache_hit_count=sum(item.cache_hit_count for item in items),
        rate_limit_wait_seconds=sum(
            item.rate_limit_wait_seconds for item in items
        ),
        retry_after_seconds=max(
            (
                item.retry_after_seconds
                for item in items
                if item.retry_after_seconds is not None
            ),
            default=None,
        ),
        latency_seconds=sum(item.latency_seconds for item in items),
    )
