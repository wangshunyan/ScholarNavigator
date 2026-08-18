from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.core.search_schemas import DEFAULT_SEARCH_SOURCES, SearchBudget
from scholar_agent.evaluation.query_planning_regression import (
    BASELINE_APPROVAL_TOKEN,
    QueryPlanningAuditError,
    build_planning_audit,
    check_planning_regression,
    project_query_only_manifest,
    propose_planning_baseline,
    sha256_file,
    sha256_json,
)


def test_query_projection_is_gold_blind_and_preserves_unicode_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "qid": "q-一",
                "question": "蛋白质—折叠与 G.N.N.",
                "answer": ["must never be copied"],
                "answer_arxiv_id": ["secret-id"],
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {"qid": "q-2", "question": "second", "private_gold": {"x": 1}}
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "queries.jsonl"

    result = project_query_only_manifest(source, output)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {"query_id": "q-一", "query": "蛋白质—折叠与 G.N.N."},
        {"query_id": "q-2", "query": "second"},
    ]
    assert result["gold_fields_accessed"] is False
    assert "answer" not in output.read_text()
    assert "secret-id" not in output.read_text()
    with pytest.raises(QueryPlanningAuditError, match="already exists"):
        project_query_only_manifest(source, output)


def test_query_projection_rejects_missing_query_field(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"qid":"q","answer":["ignored"]}\n', encoding="utf-8")

    with pytest.raises(QueryPlanningAuditError, match="missing_question"):
        project_query_only_manifest(source, tmp_path / "queries.jsonl")


def test_audit_handles_unicode_long_query_duplicate_terms_and_budget(
    tmp_path: Path,
) -> None:
    long_query = "graph neural network " * 800
    manifest_path, manifest = _manifest(
        tmp_path,
        [
            ("unicode", "比较 β-折叠蛋白与 G.N.N. 方法"),
            ("long", long_query),
            ("duplicate", "BERT BERT benchmark benchmark dataset dataset"),
        ],
    )

    rows, summary, runtime = build_planning_audit(manifest)

    assert [row["query_id"] for row in rows] == ["unicode", "long", "duplicate"]
    assert all(row["status"] == "success" for row in rows)
    for row in rows:
        assert row["quality"]["schema_valid"] is True
        assert row["quality"]["serialization_valid"] is True
        assert row["quality"]["budget_consistent"] is True
        assert row["quality"]["subquery_count"] <= 3
        assert row["plan"]["selected_sources"] == list(DEFAULT_SEARCH_SOURCES)
        assert all(
            item["source_hints"] == list(DEFAULT_SEARCH_SOURCES)
            for item in row["plan"]["subqueries"]
        )
    assert summary["quality"]["duplicate_subquery_total"] == 0
    assert summary["quality"]["budget_consistent_count"] == 3
    duplicate_constraints = rows[2]["plan"]["query_analysis"]["constraints"]
    for field in ("methods", "datasets", "must_include_terms"):
        values = duplicate_constraints[field]
        assert len(values) == len({value.casefold() for value in values})
    assert runtime["excluded_from_regression"] is True
    assert len(runtime["per_query"]) == 3
    assert manifest_path.exists()


def test_empty_query_has_explicit_stable_error_terminal(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path, [("empty", "   ")])

    rows, summary, _ = build_planning_audit(manifest, measure_latency=False)

    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == {"type": "ValueError", "code": "empty_query"}
    assert summary["terminal_counts"] == {"error": 1}


def test_planning_and_summary_are_byte_deterministic_and_ordered(
    tmp_path: Path,
) -> None:
    _, manifest = _manifest(
        tmp_path,
        [("q2", "second query"), ("q1", "first query"), ("q3", "third query")],
    )

    rows_1, summary_1, runtime_1 = build_planning_audit(
        manifest, measure_latency=False
    )
    rows_2, summary_2, runtime_2 = build_planning_audit(
        manifest, measure_latency=False
    )

    assert [row["query_id"] for row in rows_1] == ["q2", "q1", "q3"]
    assert json.dumps(rows_1, ensure_ascii=False, sort_keys=True) == json.dumps(
        rows_2, ensure_ascii=False, sort_keys=True
    )
    assert summary_1 == summary_2
    assert runtime_1 == runtime_2
    assert summary_1["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "connector_invoked": False,
        "evaluator_invoked": False,
        "gold_fields_accessed": False,
    }


def test_schema_round_trip_preserves_full_plan(tmp_path: Path) -> None:
    _, manifest = _manifest(
        tmp_path,
        [("q", "recent graph neural network benchmark since 2020")],
    )

    rows, _, _ = build_planning_audit(manifest, measure_latency=False)

    row = rows[0]
    assert row["status"] == "success"
    assert row["plan_sha256"] == sha256_json(row["plan"])
    assert row["plan"]["query_planning"]["planner_version"]
    assert row["plan"]["enable_query_evolution"] is False
    assert row["plan"]["enable_refchain"] is False


def test_regression_gate_reports_minimal_plan_drift(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path, [("q", "graph retrieval")])
    rows, summary, _ = build_planning_audit(manifest, measure_latency=False)
    baseline = tmp_path / "baseline.jsonl"
    baseline_summary = tmp_path / "baseline-summary.json"
    _write_jsonl(baseline, rows)
    _write_json(baseline_summary, summary)
    manifest["baseline"] = {
        "plans_path": str(baseline),
        "plans_sha256": sha256_file(baseline),
        "summary_path": str(baseline_summary),
        "summary_sha256": sha256_file(baseline_summary),
    }
    hashes_path = tmp_path / "hashes.jsonl"
    hash_rows = [
        {"query_id": row["query_id"], "plan_sha256": row["plan_sha256"]}
        for row in rows
    ]
    _write_jsonl(hashes_path, hash_rows)
    manifest["expected"] = {
        "query_plan_hashes_path": str(hashes_path),
        "query_plan_hashes_sha256": sha256_file(hashes_path),
        "query_plan_hash_count": len(hash_rows),
    }
    _write_json(manifest_path, manifest)
    # Keep the frozen fingerprints unchanged while simulating a current-plan drift.
    baseline_rows = [dict(rows[0])]
    baseline_rows[0]["plan"]["top_k"] = 99
    _write_jsonl(baseline, baseline_rows)
    manifest["baseline"]["plans_sha256"] = sha256_file(baseline)
    _write_json(manifest_path, manifest)

    report = check_planning_regression(manifest_path, tmp_path / "run")

    assert report["passed"] is False
    paths = {item["path"] for item in report["drifts"]}
    assert "$[0].plan.top_k" in paths


def test_baseline_update_requires_explicit_token_and_reason(tmp_path: Path) -> None:
    manifest_path, _ = _manifest(tmp_path, [("q", "query")])

    with pytest.raises(QueryPlanningAuditError, match="token rejected"):
        propose_planning_baseline(
            manifest_path,
            tmp_path / "bad-token",
            approval_token="wrong",
            reason="a sufficiently long audit reason",
        )
    with pytest.raises(QueryPlanningAuditError, match="too short"):
        propose_planning_baseline(
            manifest_path,
            tmp_path / "bad-reason",
            approval_token=BASELINE_APPROVAL_TOKEN,
            reason="short",
        )
    audit = propose_planning_baseline(
        manifest_path,
        tmp_path / "proposal",
        approval_token=BASELINE_APPROVAL_TOKEN,
        reason="freeze deterministic query-planning fixture",
    )
    assert audit["tracked_files_mutated"] is False
    assert audit["approval_token_verified"] is True
    assert audit["query_plan_hash_count"] == 1


def test_manifest_rejects_budget_or_experiment_drift(tmp_path: Path) -> None:
    _, manifest = _manifest(tmp_path, [("q", "query")])
    manifest["plan_config"]["budgets"]["max_candidate_papers"] = 999
    with pytest.raises(Exception):
        build_planning_audit(manifest)

    _, manifest = _manifest(tmp_path / "second", [("q", "query")])
    manifest["plan_config"]["experimental_features"]["rrf_fusion"] = True
    with pytest.raises(QueryPlanningAuditError, match="experimental"):
        build_planning_audit(manifest)


def _manifest(
    tmp_path: Path,
    queries: list[tuple[str, str]],
) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "queries.jsonl"
    _write_jsonl(
        input_path,
        [{"query_id": query_id, "query": query} for query_id, query in queries],
    )
    prompt_path = tmp_path / "prompt-manifest.json"
    _write_json(prompt_path, {"query_understanding": {"version": "test"}})
    manifest = {
        "schema_version": "1",
        "gate": "autoscholar_query_planning_regression",
        "dataset": {
            "name": "auto_scholar_query",
            "split": "test",
            "case_count": len(queries),
        },
        "query_input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "allowed_fields": ["query_id", "query"],
        },
        "prompt_state": {
            "used": False,
            "manifest_path": str(prompt_path),
            "manifest_sha256": sha256_file(prompt_path),
        },
        "plan_config": {
            "query_planning_policy": "current_rules",
            "query_planner_version": "1.9.0",
            "run_profile": "balanced",
            "top_k": 20,
            "sources": list(DEFAULT_SEARCH_SOURCES),
            "limit_per_source": 20,
            "max_subqueries": 3,
            "effective_current_year": 2026,
            "enable_llm": False,
            "enable_query_evolution": False,
            "enable_refchain": False,
            "enable_semantic_seed_expansion": False,
            "ranking_policy": "current_rules",
            "judgement_policy": "current_rules",
            "query_adapter_policy": "adaptive",
            "budgets": SearchBudget().model_dump(mode="json"),
            "experimental_features": {
                "concept_projection": False,
                "prf_v1": False,
                "rrf_fusion": False,
                "lexical_normalization_v1": False,
                "llm_constrained_rewrite": False,
                "local_bm25": False,
            },
        },
        "baseline": {},
        "expected": {},
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
