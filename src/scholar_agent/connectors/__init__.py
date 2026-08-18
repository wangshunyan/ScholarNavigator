"""Academic search connectors."""

from .arxiv import search_arxiv, search_arxiv_detailed
from .local_bm25 import (
    LocalBM25Config,
    LocalBM25FieldConfig,
    configure_local_bm25,
    local_bm25_connector_version,
    local_bm25_metadata,
    search_local_bm25,
    search_local_bm25_detailed,
)
from .local_hybrid import (
    LocalHybridConfig,
    LocalHybridIndexMetadata,
    build_local_hybrid_index,
    configure_local_hybrid,
    local_hybrid_connector_version,
    local_hybrid_metadata,
    search_local_hybrid,
    search_local_hybrid_detailed,
)
from .openalex import (
    fetch_openalex_references,
    fetch_openalex_references_detailed,
    search_openalex,
    search_openalex_detailed,
)
from .pubmed import search_pubmed, search_pubmed_detailed
from .schemas import ConnectorSearchResult
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from .semantic_scholar import (
    recommend_semantic_scholar_papers_detailed,
    resolve_semantic_scholar_paper_ids_detailed,
    search_semantic_scholar,
    search_semantic_scholar_detailed,
)

__all__ = [
    "ConnectorSearchResult",
    "ConnectorDiagnostics",
    "fetch_openalex_references",
    "fetch_openalex_references_detailed",
    "LocalBM25Config",
    "LocalBM25FieldConfig",
    "configure_local_bm25",
    "local_bm25_connector_version",
    "local_bm25_metadata",
    "search_arxiv",
    "search_arxiv_detailed",
    "search_local_bm25",
    "search_local_bm25_detailed",
    "LocalHybridConfig",
    "LocalHybridIndexMetadata",
    "build_local_hybrid_index",
    "configure_local_hybrid",
    "local_hybrid_connector_version",
    "local_hybrid_metadata",
    "search_local_hybrid",
    "search_local_hybrid_detailed",
    "search_openalex",
    "search_openalex_detailed",
    "search_pubmed",
    "search_pubmed_detailed",
    "recommend_semantic_scholar_papers_detailed",
    "resolve_semantic_scholar_paper_ids_detailed",
    "search_semantic_scholar",
    "search_semantic_scholar_detailed",
]
