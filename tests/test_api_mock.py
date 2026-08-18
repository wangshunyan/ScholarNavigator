from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholar_agent.app.main import app  # noqa: E402


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["time"]


def test_runtime_config_is_real_search_only(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    _clear_local_bm25_env(monkeypatch)
    _clear_local_hybrid_env(monkeypatch)
    response = client.get("/api/v1/runtime/config")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] in {"real_search", "real"}
    assert body["llm"] == {
        "provider": "disabled",
        "model": None,
        "available": False,
        "base_url_host": None,
        "reason": "provider_disabled",
    }
    assert body["features"]["sse"] is True
    assert body["features"]["real_search"] is True
    assert body["features"]["real_search_cancel"] is True
    assert body["features"]["real_search_sse"] is True
    assert body["features"]["retrieval_cache"] is True
    assert body["features"]["batch_cli"] is True
    assert body["features"]["llm_query_understanding"] is False
    assert body["features"]["llm_judgement"] is False
    assert body["limits"]["real_search_max_workers"] >= 1
    assert body["limits"]["real_search_background_workers"] >= 1
    assert "real_search_run_ttl_seconds" in body["limits"]
    assert "real_search_max_stored_runs" in body["limits"]

    connectors = {connector["name"]: connector for connector in body["connectors"]}
    assert "mock" not in connectors
    assert connectors["openalex"]["available"] is True
    assert connectors["openalex"]["reason"] == "implemented_for_real_search"
    assert connectors["arxiv"]["available"] is True
    assert connectors["arxiv"]["reason"] == "implemented_for_real_search"
    assert connectors["semantic_scholar"]["available"] is True
    assert connectors["semantic_scholar"]["requires_key"] is False
    assert connectors["semantic_scholar"]["reason"].startswith("implemented_for_real_search")
    assert connectors["pubmed"]["available"] is True
    assert connectors["pubmed"]["requires_key"] is False
    assert connectors["pubmed"]["reason"].startswith("implemented_for_real_search")
    assert connectors["local_bm25"]["available"] is False
    assert connectors["local_bm25"]["requires_key"] is False
    assert connectors["local_bm25"]["reason"] == "local_bm25_env_not_configured"
    assert connectors["local_hybrid"]["available"] is False
    assert connectors["local_hybrid"]["requires_key"] is False
    assert connectors["local_hybrid"]["reason"] == "local_hybrid_env_not_configured"


def test_runtime_config_shows_local_bm25_from_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_llm_env(monkeypatch)
    _clear_local_bm25_env(monkeypatch)
    _clear_local_hybrid_env(monkeypatch)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"_id":"2501.00001","title":"Local paper","abstract":"retrieval"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_BM25_CORPUS", str(corpus))
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_IDENTITY", "arxiv_id")

    response = client.get("/api/v1/runtime/config")

    assert response.status_code == 200
    connectors = {
        connector["name"]: connector
        for connector in response.json()["connectors"]
    }
    assert connectors["local_bm25"]["available"] is True
    assert connectors["local_bm25"]["requires_key"] is False
    assert connectors["local_bm25"]["reason"] == "configured_from_env:1_documents"

    monkeypatch.delenv("SCHOLAR_AGENT_LOCAL_BM25_CORPUS", raising=False)
    client.get("/api/v1/runtime/config")


def test_runtime_config_shows_local_hybrid_from_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_llm_env(monkeypatch)
    _clear_local_bm25_env(monkeypatch)
    _clear_local_hybrid_env(monkeypatch)
    bm25_corpus = tmp_path / "bm25.jsonl"
    bm25_corpus.write_text(
        '{"_id":"2501.00001","title":"Local hybrid paper","abstract":""}\n',
        encoding="utf-8",
    )
    semantic_corpus = tmp_path / "semantic.jsonl"
    semantic_corpus.write_text(
        (
            '{"_id":"2501.00001","arxiv_id":"2501.00001",'
            '"title":"Local hybrid paper","abstract":"semantic retrieval"}\n'
        ),
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    model_dir = tmp_path / "model"
    index_dir.mkdir()
    model_dir.mkdir()
    np.save(index_dir / "embeddings.npy", np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    corpus_bytes = semantic_corpus.read_bytes()
    metadata = {
        "schema_version": "1",
        "semantic_corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "semantic_corpus_size_bytes": len(corpus_bytes),
        "document_count": 1,
        "abstract_document_count": 1,
        "embedding_dimension": 3,
        "model_path": str(model_dir.resolve()),
        "model_fingerprint": "test",
        "index_fingerprint": "test",
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_BM25_CORPUS", str(bm25_corpus))
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_IDENTITY", "arxiv_id")
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_HYBRID_SEMANTIC_CORPUS", str(semantic_corpus))
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_HYBRID_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_HYBRID_MODEL", str(model_dir))

    response = client.get("/api/v1/runtime/config")

    assert response.status_code == 200
    connectors = {
        connector["name"]: connector
        for connector in response.json()["connectors"]
    }
    assert connectors["local_hybrid"]["available"] is True
    assert connectors["local_hybrid"]["requires_key"] is False
    assert connectors["local_hybrid"]["reason"] == (
        "configured_from_env:1_documents:1_abstracts"
    )

    _clear_local_hybrid_env(monkeypatch)
    _clear_local_bm25_env(monkeypatch)
    client.get("/api/v1/runtime/config")


def test_runtime_config_shows_enabled_llm_without_api_key_leak(monkeypatch) -> None:
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_API_KEY", "sk-do-not-leak")
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_MODEL", "gpt-test")
    monkeypatch.setenv("SCHOLAR_AGENT_ENABLE_LLM_QUERY_UNDERSTANDING", "1")
    monkeypatch.setenv("SCHOLAR_AGENT_ENABLE_LLM_JUDGEMENT", "1")

    response = client.get("/api/v1/runtime/config")

    assert response.status_code == 200
    body = response.json()
    assert body["llm"] == {
        "provider": "openai_compatible",
        "model": "gpt-test",
        "available": True,
        "base_url_host": "api.example.test",
        "reason": None,
    }
    assert body["features"]["llm_query_understanding"] is True
    assert body["features"]["llm_judgement"] is True
    assert "sk-do-not-leak" not in response.text


def test_legacy_mock_search_run_endpoints_are_not_available() -> None:
    post_response = client.post(
        "/api/v1/search/runs",
        json={"query": "请帮我搜索关于 LLM reranking 的代表性论文"},
    )
    status_response = client.get("/api/v1/search/runs/some_id")
    result_response = client.get("/api/v1/search/runs/some_id/result")
    events_response = client.get("/api/v1/search/runs/some_id/events")

    assert post_response.status_code in {404, 405}
    assert status_response.status_code in {404, 405}
    assert result_response.status_code in {404, 405}
    assert events_response.status_code in {404, 405}


def test_legacy_mock_search_run_paths_are_not_in_openapi() -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]

    assert "/api/v1/search/runs" not in paths
    assert "/api/v1/search/runs/{run_id}" not in paths
    assert "/api/v1/search/runs/{run_id}/result" not in paths
    assert "/api/v1/search/runs/{run_id}/events" not in paths
    assert "/api/v1/real/search/runs" in paths
    assert "/api/v1/real/search/runs/{run_id}" in paths
    assert "/api/v1/real/search/runs/{run_id}/result" in paths
    assert "/api/v1/real/search/runs/{run_id}/events" in paths
    assert "/api/v1/real/search/runs/{run_id}/cancel" in paths


def _clear_llm_env(monkeypatch) -> None:
    for env_name in (
        "SCHOLAR_AGENT_LLM_PROVIDER",
        "SCHOLAR_AGENT_LLM_BASE_URL",
        "SCHOLAR_AGENT_LLM_API_KEY",
        "SCHOLAR_AGENT_LLM_MODEL",
        "SCHOLAR_AGENT_ENABLE_LLM_QUERY_UNDERSTANDING",
        "SCHOLAR_AGENT_ENABLE_LLM_JUDGEMENT",
    ):
        monkeypatch.delenv(env_name, raising=False)


def _clear_local_bm25_env(monkeypatch) -> None:
    for env_name in (
        "SCHOLAR_AGENT_LOCAL_BM25_CORPUS",
        "SCHOLAR_AGENT_LOCAL_BM25_CACHE_DIR",
        "SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_ID_FIELD",
        "SCHOLAR_AGENT_LOCAL_BM25_TITLE_FIELD",
        "SCHOLAR_AGENT_LOCAL_BM25_ABSTRACT_FIELD",
        "SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_IDENTITY",
        "SCHOLAR_AGENT_LOCAL_BM25_DOI_FIELD",
        "SCHOLAR_AGENT_LOCAL_BM25_ARXIV_ID_FIELD",
        "SCHOLAR_AGENT_LOCAL_BM25_SEMANTIC_SCHOLAR_ID_FIELD",
        "SCHOLAR_AGENT_LOCAL_BM25_S2ORC_CORPUS_ID_FIELD",
        "SCHOLAR_AGENT_LOCAL_BM25_OPENALEX_ID_FIELD",
        "SCHOLAR_AGENT_LOCAL_BM25_PUBMED_ID_FIELD",
    ):
        monkeypatch.delenv(env_name, raising=False)


def _clear_local_hybrid_env(monkeypatch) -> None:
    for env_name in (
        "SCHOLAR_AGENT_LOCAL_HYBRID_SEMANTIC_CORPUS",
        "SCHOLAR_AGENT_LOCAL_HYBRID_INDEX_DIR",
        "SCHOLAR_AGENT_LOCAL_HYBRID_MODEL",
        "SCHOLAR_AGENT_LOCAL_HYBRID_BM25_CANDIDATE_LIMIT",
        "SCHOLAR_AGENT_LOCAL_HYBRID_SEMANTIC_CANDIDATE_LIMIT",
        "SCHOLAR_AGENT_LOCAL_HYBRID_RRF_K",
    ):
        monkeypatch.delenv(env_name, raising=False)
