"""检索查询适配与来源调度。"""

from scholar_agent.retrieval.query_adapter import (
    DEFAULT_QUERY_ADAPTER_POLICY,
    MAX_ADAPTED_QUERIES_PER_SOURCE,
    AdaptedQuery,
    QueryAdapterPolicy,
    adapt_queries_for_source,
    adapt_query_for_source,
)

__all__ = [
    "MAX_ADAPTED_QUERIES_PER_SOURCE",
    "DEFAULT_QUERY_ADAPTER_POLICY",
    "AdaptedQuery",
    "QueryAdapterPolicy",
    "adapt_queries_for_source",
    "adapt_query_for_source",
]
