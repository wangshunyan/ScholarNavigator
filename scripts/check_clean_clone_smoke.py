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


def _template_env_smoke(root: Path) -> dict[str, Any]:
    """Verify the documented ``.env.example`` path in a fresh subprocess."""

    template = root / ".env.example"
    if not template.is_file():
        raise RuntimeError("env_example_missing")
    env_file = root / ".env"
    env_file.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        payload = _health_and_config(root)
        connectors = {
            item["name"]: item for item in payload["runtime_config"]["connectors"]
        }
        local_bm25 = connectors.get("local_bm25")
        llm = payload["runtime_config"].get("llm", {})
        if not local_bm25 or not local_bm25.get("available"):
            raise RuntimeError("env_example_local_bm25_not_available")
        if llm.get("available") or llm.get("provider") != "disabled":
            raise RuntimeError("env_example_llm_must_remain_disabled")
        search = _template_search_smoke(root)
    finally:
        env_file.unlink(missing_ok=True)
    return {
        "status": "ready",
        "local_bm25": local_bm25,
        "llm": llm,
        "search": search,
    }


def _template_search_smoke(root: Path) -> dict[str, Any]:
    """Run one bounded real search while the copied template ``.env`` exists."""

    code = """
import json
import time
from fastapi.testclient import TestClient
from scholar_agent.app.main import app

client = TestClient(app)
runtime = client.get('/api/v1/runtime/config')
if runtime.status_code != 200:
    raise SystemExit(f'template_search_runtime_config_failed:{runtime.status_code}')
if not next(
    item for item in runtime.json()['connectors'] if item['name'] == 'local_bm25'
).get('available'):
    raise SystemExit('template_search_local_bm25_unavailable')
created = client.post('/api/v1/real/search/runs', json={
    'query': 'Keyphrase Generation for Scientific Document Retrieval',
    'source_preferences': ['local_bm25'],
    'top_k': 2,
    'run_profile': 'fast',
    'options': {
        'enable_refchain': False,
        'stream_events': False,
        'return_markdown': False,
        'return_json': True,
    },
    'budgets': {
        'max_search_rounds': 1,
        'max_candidate_papers': 10,
        'max_llm_calls': 0,
        'max_total_tokens': 0,
        'max_latency_seconds': 60,
    },
})
if created.status_code != 201:
    raise SystemExit(f'template_search_create_failed:{created.status_code}')
run_id = created.json()['run_id']
terminal = None
for _ in range(240):
    status = client.get(f'/api/v1/real/search/runs/{run_id}')
    if status.status_code != 200:
        raise SystemExit(f'template_search_status_failed:{status.status_code}')
    payload = status.json()
    if payload['status'] in {'succeeded', 'failed', 'cancelled'}:
        terminal = payload
        break
    time.sleep(0.25)
if terminal is None or terminal['status'] != 'succeeded':
    raise SystemExit(json.dumps({
        'template_search_status': terminal,
    }, ensure_ascii=False))
result = client.get(f'/api/v1/real/search/runs/{run_id}/result')
if result.status_code != 200:
    raise SystemExit(f'template_search_result_failed:{result.status_code}')
result_payload = result.json()
papers = result_payload['highly_relevant_papers'] + result_payload['partially_relevant_papers']
print(json.dumps({
    'status': 'ready',
    'result_count': len(papers),
    'candidate_paper_count': terminal['progress']['candidate_paper_count'],
    'api_call_count': terminal['cost_report']['api_call_count'],
    'llm_call_count': terminal['cost_report']['llm_call_count'],
    'warnings': result_payload.get('warnings', []),
}, ensure_ascii=False, sort_keys=True))
"""
    result = _run(root, code)
    if result.returncode != 0:
        raise RuntimeError(f"template_env_search_failed:{result.stderr[-500:]}")
    try:
        search = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("template_env_search_invalid_json") from exc
    if (
        search.get("status") != "ready"
        or not isinstance(search.get("result_count"), int)
        or not isinstance(search.get("candidate_paper_count"), int)
        or search["candidate_paper_count"] < 1
        or search.get("api_call_count") != 0
        or search.get("llm_call_count") != 0
    ):
            raise RuntimeError(
                "template_env_search_contract_invalid:" + json.dumps(search)
            )

    return {
        "status": "ready",
        "result_count": search["result_count"],
        "candidate_paper_count": search["candidate_paper_count"],
        "api_call_count": search["api_call_count"],
        "llm_call_count": search["llm_call_count"],
        "warnings": search["warnings"],
    }


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
    from scripts.verify_contest_release_package import verify_package

    source = repository_root.resolve()
    with tempfile.TemporaryDirectory(prefix="scholar-clean-clone-") as temporary:
        staging = Path(temporary)
        archive = staging / "release.zip"
        package = build_package(source, archive)
        package_verification = verify_package(archive, expected_commit=package.get("source_commit"))
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
        template_env = _template_env_smoke(extracted)
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
            "package_verification": package_verification,
            "api": api,
            "template_env": template_env,
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
