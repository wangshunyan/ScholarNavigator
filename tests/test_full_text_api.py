from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scholar_agent.app import main as app_main  # noqa: E402
from scholar_agent.app.api import routes  # noqa: E402
from scholar_agent.core.full_text_evidence import (  # noqa: E402
    FullTextFetchResult,
    build_paragraph_evidence,
)


client = TestClient(app_main.app)


def test_full_text_endpoint_requires_exact_allowlisted_host() -> None:
    response = client.post(
        "/api/v1/full-text/fetch",
        json={
            "source_url": "https://open.example/paper",
            "license_id": "CC-BY-4.0",
            "license_verified": True,
            "allowed_hosts": ["other.example"],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "full_text_allowed_host_must_match_source_url"


def test_full_text_endpoint_returns_structured_evidence(monkeypatch) -> None:
    document = build_paragraph_evidence(
        "A licensed paragraph.",
        source_url="https://open.example/paper",
        license_id="CC-BY-4.0",
        license_verified=True,
    )

    def fake_fetch(**kwargs):  # noqa: ANN003
        assert kwargs["allowed_hosts"] == {"open.example"}
        assert kwargs["license_verified"] is True
        return FullTextFetchResult(
            status="succeeded",
            document=document,
            source_url=kwargs["source_url"],
            license_id=kwargs["license_id"],
            response_media_type="text/plain",
        )

    monkeypatch.setattr(routes, "fetch_open_full_text", fake_fetch)
    response = client.post(
        "/api/v1/full-text/fetch",
        json={
            "source_url": "https://open.example/paper",
            "license_id": "CC-BY-4.0",
            "license_verified": True,
            "allowed_hosts": ["open.example"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["document"]["source"]["license_id"] == "CC-BY-4.0"
    assert payload["document"]["paragraphs"][0]["evidence_id"].startswith(
        "paragraph:"
    )


def test_full_text_endpoint_is_fail_closed_without_license(monkeypatch) -> None:
    called = False

    def fake_fetch(**kwargs):  # noqa: ANN003
        nonlocal called
        called = True
        return FullTextFetchResult(status="license_unverified")

    monkeypatch.setattr(routes, "fetch_open_full_text", fake_fetch)
    response = client.post(
        "/api/v1/full-text/fetch",
        json={
            "source_url": "https://open.example/paper",
            "license_id": "CC-BY-4.0",
            "license_verified": False,
            "allowed_hosts": ["open.example"],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "license_unverified"
    assert called is True
