#!/usr/bin/env python3
"""Verify that a source-only checkout is runnable in a fresh directory.

The smoke is deliberately small and offline.  It validates packaging, Python
imports, API health/config, and the tracked local BM25 corpus when present. It
does not require semantic model assets, an LLM provider, credentials, or
network access.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXIT_READY = 0
EXIT_NOT_READY = 3
EXIT_VIOLATION = 2

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _python_env(root: Path) -> dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key in ("SystemRoot", "WINDIR"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    return environment


def _run(root: Path, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=_python_env(root),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def _health_and_config(root: Path) -> dict[str, Any]:
    code = """
import json
from fastapi.testclient import TestClient
from scholar_agent.app.main import app
client = TestClient(app)
health = client.get('/api/v1/health')
config = client.get('/api/v1/runtime/config')
if health.status_code != 200 or config.status_code != 200:
    raise SystemExit(json.dumps({'health': health.status_code, 'config': config.status_code}))
print(json.dumps({'health': health.json(), 'runtime_config': config.json()}, ensure_ascii=False, sort_keys=True))
"""
    result = _run(root, code)
    if result.returncode != 0:
        raise RuntimeError(f"api_smoke_failed:{result.stderr[-500:]}")
    return json.loads(result.stdout)


def _bm25_smoke(root: Path) -> dict[str, Any]:
    corpus = root / "datasets/local_bm25/pasa_papers.jsonl"
    if not corpus.exists():
        return {"status": "not_ready", "reason": "tracked_local_bm25_corpus_missing"}
    code = """
import json
from pathlib import Path
from scholar_agent.connectors.local_bm25 import (
    LocalBM25Config, LocalBM25FieldConfig, configure_local_bm25,
    search_local_bm25_detailed,
)
root = Path.cwd()
configure_local_bm25(LocalBM25Config(
    corpus_path=root / 'datasets/local_bm25/pasa_papers.jsonl',
    cache_dir=root / 'smoke-cache',
    fields=LocalBM25FieldConfig(
        document_id='_id', title='title', abstract='abstract',
        document_id_identity='arxiv_id', arxiv_id='arxiv_id',
    ),
))
result = search_local_bm25_detailed('graph neural networks for scientific document retrieval', 5)
if len(result.papers) != 5 or any(not paper.title for paper in result.papers):
    raise SystemExit('bm25_result_invalid')
print(json.dumps({
    'schema_version': 'offline-search-result-v1',
    'query': 'graph neural networks for scientific document retrieval',
    'results': [
        {
            'rank': index,
            'arxiv_id': paper.identifiers.arxiv_id,
            'title': paper.title,
            'sources': list(paper.sources),
        }
        for index, paper in enumerate(result.papers, start=1)
    ],
    'warnings': list(result.warnings),
}, ensure_ascii=False, sort_keys=True))
"""
    result = _run(root, code)
    if result.returncode != 0:
        raise RuntimeError(f"bm25_smoke_failed:{result.stderr[-500:]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("bm25_structured_export_invalid_json") from exc
    rows = payload.get("results")
    if (
        payload.get("schema_version") != "offline-search-result-v1"
        or not isinstance(rows, list)
        or len(rows) != 5
        or [row.get("rank") for row in rows] != [1, 2, 3, 4, 5]
        or any(not isinstance(row.get("title"), str) or not row["title"] for row in rows)
        or any("gold" in row for row in rows)
    ):
        raise RuntimeError("bm25_structured_export_schema_invalid")
    return {
        "status": "ready",
        "schema_version": payload["schema_version"],
        "query": payload["query"],
        "result_count": len(rows),
        "results": rows,
        "warnings": payload.get("warnings", []),
    }


def run_smoke(repository_root: Path, *, keep_directory: bool = False) -> dict[str, Any]:
    from scripts.build_contest_release_package import build_package

    source = repository_root.resolve()
    with tempfile.TemporaryDirectory(prefix="scholar-clean-clone-") as temporary:
        staging = Path(temporary)
        archive = staging / "release.zip"
        package = build_package(source, archive)
        extracted = staging / "checkout"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)
        compile_result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "src", "scripts"],
            cwd=extracted,
            env=_python_env(extracted),
            capture_output=True,
            text=True,
            check=False,
        )
        if compile_result.returncode != 0:
            raise RuntimeError("compileall_failed")
        api = _health_and_config(extracted)
        bm25 = _bm25_smoke(extracted)
        dependency_inputs = {
            "requirements_txt": (extracted / "requirements.txt").is_file(),
            "frontend_package_json": (extracted / "frontend/package.json").is_file(),
            "frontend_lockfile": (extracted / "frontend/package-lock.json").is_file(),
        }
        if not all(dependency_inputs.values()):
            raise RuntimeError("release_dependency_contract_incomplete")
        report = {
            "schema_version": "clean_clone_smoke_v1",
            "status": "ready",
            "exit_code": EXIT_READY,
            "package": package,
            "api": api,
            "local_bm25": bm25,
            "dependency_inputs": dependency_inputs,
            "network_request_count": 0,
            "dotenv_read": False,
            "official_metric_scope": "internal_engineering_only",
        }
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run_smoke(args.repository_root)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        report = {
            "schema_version": "clean_clone_smoke_v1",
            "status": "not_ready",
            "exit_code": EXIT_NOT_READY,
            "reason": str(exc),
            "network_request_count": 0,
            "dotenv_read": False,
        }
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
