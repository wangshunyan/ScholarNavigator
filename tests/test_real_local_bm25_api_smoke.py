from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from scholar_agent.app.main import app


def test_real_api_runs_offline_local_bm25_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "papers.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "_id": "2401.00001",
                        "arxiv_id": "2401.00001",
                        "title": "Graph neural networks for scientific retrieval",
                        "abstract": "A study of graph retrieval systems.",
                    }
                ),
                json.dumps(
                    {
                        "_id": "2401.00002",
                        "arxiv_id": "2401.00002",
                        "title": "Unrelated optimization methods",
                        "abstract": "A study of numerical optimization.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_BM25_CORPUS", str(corpus))
    monkeypatch.setenv(
        "SCHOLAR_AGENT_LOCAL_BM25_CACHE_DIR", str(tmp_path / "cache")
    )
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_BM25_DOCUMENT_IDENTITY", "arxiv_id")
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_BM25_ARXIV_ID_FIELD", "arxiv_id")
    monkeypatch.setenv("SCHOLAR_AGENT_LOCAL_BM25_DOI_FIELD", "doi")

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/real/search/runs",
            json={
                "query": "graph neural networks scientific retrieval",
                "source_preferences": ["local_bm25"],
                "top_k": 2,
                "run_profile": "fast",
                "options": {
                    "enable_refchain": False,
                    "stream_events": False,
                    "return_markdown": False,
                    "return_json": True,
                },
                "budgets": {
                    "max_search_rounds": 1,
                    "max_candidate_papers": 10,
                    "max_llm_calls": 0,
                    "max_total_tokens": 0,
                    "max_latency_seconds": 20,
                },
            },
        )
        assert created.status_code == 201
        run_id = created.json()["run_id"]

        terminal = None
        for _ in range(80):
            status = client.get(f"/api/v1/real/search/runs/{run_id}")
            assert status.status_code == 200
            value = status.json()
            if value["status"] in {"succeeded", "failed", "cancelled"}:
                terminal = value
                break
            time.sleep(0.1)

        assert terminal is not None
        if terminal["status"] != "succeeded":
            failure = client.get(f"/api/v1/real/search/runs/{run_id}/result")
            raise AssertionError(f"local API run failed: {failure.json()}")
        assert terminal["cost_report"]["api_call_count"] == 0
        assert terminal["cost_report"]["llm_call_count"] == 0

        result = client.get(f"/api/v1/real/search/runs/{run_id}/result")
        assert result.status_code == 200
        payload = result.json()
        papers = payload["highly_relevant_papers"] + payload["partially_relevant_papers"]
        assert papers
        assert any(
            item["paper"]["identifiers"]["arxiv_id"] == "2401.00001"
            for item in papers
        )
        assert not any("source_error:local_bm25" in warning for warning in payload["warnings"])
