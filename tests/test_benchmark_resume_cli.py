from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_benchmark
from scholar_agent.evaluation.snapshot_resume import (
    ResumeManifest,
    ResumeRequest,
    ResumeRuntimeConfig,
    request_signature,
    stable_hash,
)
from scholar_agent.evaluation.snapshots.schemas import (
    SnapshotPlanEntry,
    SnapshotPlanRound,
)
from scholar_agent.evaluation.snapshots.store import retrieval_snapshot_key


def _runtime_config() -> ResumeRuntimeConfig:
    return ResumeRuntimeConfig(
        dataset="auto_scholar_query",
        dataset_split="test",
        offset=0,
        limit=1000,
        run_profile="balanced",
        sources=["openalex", "arxiv", "semantic_scholar", "pubmed"],
        result_policy="highly_and_partial",
        top_k=20,
        query_adapter_policy="adaptive",
        query_planning_policy="current_rules",
        ranking_policy="current_rules",
        judgement_policy="current_rules",
        enable_query_evolution=False,
        query_evolution_policy="off",
        enable_refchain=False,
        enable_semantic_seed_expansion=False,
        enable_llm_query_understanding=False,
        enable_llm_judgement=False,
        current_year=None,
        budgets={
            "max_search_rounds": 2,
            "max_candidate_papers": 200,
            "max_llm_calls": 20,
            "max_total_tokens": 50000,
            "max_latency_seconds": 90.0,
        },
    )


def _write_manifest(tmp_path: Path) -> Path:
    key, normalized = retrieval_snapshot_key(
        source="openalex",
        adapted_query="offline query",
        limit=20,
        adapter_policy="adaptive",
        connector_version="search-v1",
    )
    entry = SnapshotPlanEntry(
        key=key,
        entry_type="retrieval",
        source="openalex",
        adapted_query="offline query",
        limit=20,
        adapter_policy="adaptive",
        connector_version="search-v1",
        required_by_group="baseline",
        case_id="case-0",
        stage="initial_retrieval",
        generated_by="initial_retrieval",
        priority=2,
    )
    plan = SnapshotPlanRound(
        snapshot_name="snapshot",
        group="baseline",
        round_index=2,
        entries=[entry],
        created_at="2026-07-21T00:00:00+00:00",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    request = ResumeRequest(
        schedule_index=0,
        key=key,
        source="openalex",
        case_id="case-0",
        case_index=0,
        adapted_query="offline query",
        normalized_query=normalized,
        limit=20,
        adapter_policy="adaptive",
        connector_version="search-v1",
        stage="initial_retrieval",
        priority=2,
        initial_classification="missing",
        request_signature=request_signature(entry),
    )
    config = _runtime_config()
    requests = [request.model_dump(mode="json")]
    manifest = ResumeManifest(
        dataset="auto_scholar_query",
        snapshot_name="snapshot",
        snapshot_dir="snapshots",
        required_plan_path="plan.json",
        required_key_count=1,
        resume_key_count=1,
        classification_counts={"missing": 1},
        retry_policy={"failed": "once"},
        schedule_policy={"source_rotation": "fixed"},
        source_order=config.sources,
        runtime_config=config,
        runtime_config_sha256=config.sha256(),
        input_hashes={"frozen_plan_round": _sha(plan_path)},
        required_keys_sha256=stable_hash([key]),
        requests_sha256=stable_hash(requests),
        requests=[request],
    )
    path = tmp_path / "resume.json"
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _resume_args(tmp_path: Path, manifest: Path) -> list[str]:
    return [
        "--dataset",
        "auto_scholar_query",
        "--dataset-split",
        "test",
        "--limit",
        "1000",
        "--run-id",
        "resume-dry-run",
        "--sources",
        "openalex,arxiv,semantic_scholar,pubmed",
        "--query-evolution-policy",
        "off",
        "--retrieval-mode",
        "record-missing",
        "--snapshot-dir",
        str(tmp_path / "snapshots"),
        "--resume-manifest",
        str(manifest),
        "--resume-manifest-dry-run",
    ]


def test_resume_dry_run_does_not_load_env_network_or_write_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(run_benchmark, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        run_benchmark,
        "load_project_env",
        lambda *_: pytest.fail("dry-run must not load project environment"),
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert run_benchmark.main(_resume_args(tmp_path, manifest)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["network_request_count"] == 0
    assert report["snapshot_write_count"] == 0
    assert report["final_progress"]["pending_key_count"] == 1
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_resume_rejects_cli_config_drift_before_loading_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(run_benchmark, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        run_benchmark,
        "load_project_env",
        lambda *_: pytest.fail("invalid dry-run must not load environment"),
    )
    args = _resume_args(tmp_path, manifest)
    args.extend(["--top-k", "50"])

    assert run_benchmark.main(args) == 1
    assert "resume config drift:top_k" in capsys.readouterr().err


def test_default_cli_path_still_loads_project_env_and_runs_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        run_benchmark,
        "load_project_env",
        lambda *_: calls.append("env"),
    )
    monkeypatch.setattr(
        run_benchmark,
        "run_benchmark",
        lambda _: (
            calls.append("run")
            or SimpleNamespace(run_dir=Path("run-dir"), result_rows=[])
        ),
    )

    assert (
        run_benchmark.main(
            ["--dataset", "auto_scholar_query", "--run-id", "default-path"]
        )
        == 0
    )
    assert calls == ["env", "run"]


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
