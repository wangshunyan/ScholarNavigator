from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholar_agent.app.main import create_app, get_cors_origins  # noqa: E402


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v1/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )


def test_default_cors_allows_common_frontend_dev_origins(monkeypatch) -> None:
    monkeypatch.delenv("SCHOLAR_AGENT_CORS_ORIGINS", raising=False)
    client = TestClient(create_app())

    for origin in [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]:
        response = _preflight(client, origin)

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin


def test_env_cors_origins_are_trimmed_and_merged_with_defaults(monkeypatch) -> None:
    monkeypatch.setenv(
        "SCHOLAR_AGENT_CORS_ORIGINS",
        " http://localhost:4321, ,http://127.0.0.1:9000 ",
    )

    origins = get_cors_origins()
    client = TestClient(create_app())
    response = _preflight(client, "http://localhost:4321")

    assert "http://localhost:3000" in origins
    assert "http://localhost:4321" in origins
    assert "http://127.0.0.1:9000" in origins
    assert "" not in origins
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:4321"


def test_disallowed_origin_does_not_return_allow_origin(monkeypatch) -> None:
    monkeypatch.delenv("SCHOLAR_AGENT_CORS_ORIGINS", raising=False)
    client = TestClient(create_app())

    response = _preflight(client, "http://localhost:9999")

    assert "access-control-allow-origin" not in response.headers
