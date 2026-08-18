"""FastAPI application entry point for the ScholarNavigator backend."""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scholar_agent.core.env_loader import load_project_env


if "pytest" not in sys.modules:
    load_project_env()

from .api.routes import router  # noqa: E402


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
CORS_ORIGINS_ENV = "SCHOLAR_AGENT_CORS_ORIGINS"


def get_cors_origins() -> list[str]:
    """Return default dev origins plus optional env-configured origins."""

    origins: list[str] = []
    seen: set[str] = set()
    for origin in DEFAULT_CORS_ORIGINS:
        _append_origin(origins, seen, origin)

    configured = os.getenv(CORS_ORIGINS_ENV, "")
    for origin in configured.split(","):
        _append_origin(origins, seen, origin)

    return origins


def _append_origin(origins: list[str], seen: set[str], origin: str) -> None:
    normalized = origin.strip()
    if not normalized or normalized in seen:
        return
    origins.append(normalized)
    seen.add(normalized)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ScholarNavigator Real Search API",
        version="0.1.0",
        description=(
            "Real Search backend API for ScholarNavigator. It calls configured "
            "academic search connectors and optional backend LLM providers."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
