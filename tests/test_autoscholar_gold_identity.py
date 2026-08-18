from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.evaluation.autoscholar_gold_identity import (
    BASELINE_APPROVAL_TOKEN,
    GoldIdentityAuditError,
    analyze_gold_relations,
    build_gold_identity_audit,
    check_gold_identity_regression,
    classify_gold_identity,
    propose_gold_identity_baseline,
    sha256_file,
    write_gold_identity_audit,
)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("doi", "https://doi.org/10.1/ABC", "doi:10.1/abc"),
        ("arxiv_id", "arXiv:2401.00001v2", "arxiv:2401.00001"),
        ("pubmed_id", "PMID:12345", "pubmed:12345"),
        ("semantic_scholar_id", "S2:ABC", "s2:abc"),
        ("s2orc_corpus_id", 123, "s2orc:123"),
        ("openalex_id", "https://openalex.org/W123", "openalex:w123"),
    ],
)
def test_all_stable_identifier_types_are_exactly_evaluable(
    field: str, value: object, expected: str
) -> None:
    result = classify_gold_identity({field: value, "title": "Paper"})

    assert result["terminal_status"] == "stable_identifier_evaluable"
    assert expected in result["stable_identifiers"]
    assert result["current_evaluator_evaluable"] is True


def test_conflict_title_evidence_ambiguity_and_insufficient_are_mutually_exclusive() -> None:
    conflict = classify_gold_identity(
        {
            "doi": "10.1/one",
            "identifiers": {"doi": "10.1/two"},
            "title": "Paper",
        }
    )
    assert conflict["terminal_status"] == "identifier_conflict"
    assert conflict["identifier_conflict_fields"] == ["doi"]

    title = classify_gold_identity(
        {"title": "Unicode—Paper", "authors": ["A. Author"], "year": 2024}
    )
    assert title["terminal_status"] == "conservative_title_evidence_evaluable"
    assert title["title_evidence"]["complete"] is True

    ambiguous = classify_gold_identity({"title": "Title only"})
    assert ambiguous["terminal_status"] == "identity_ambiguous"
    assert classify_gold_identity({})["terminal_status"] == "insufficient_information"


def test_duplicate_relations_and_cross_query_reuse_follow_evaluator_semantics() -> None:
    shared = EvalGoldPaper(doi="10.1/shared")
    queries = [
        EvalQuery(query_id="q1", query="one", gold_papers=[shared, shared]),
        EvalQuery(
            query_id="q2",
            query="two",
            gold_papers=[shared, EvalGoldPaper(arxiv_id="2401.00001")],
        ),
    ]

    gold, query_rows, summary = analyze_gold_relations(queries)

    assert len(gold) == 4
    assert summary["global_unique_identity_count"] == 2
    assert summary["safe_global_unique_identity_count"] == 2
    assert summary["current_evaluator_global_unique_identity_count"] == 2
    assert summary["current_evaluator_global_repeated_relation_count"] == 2
    assert summary["safe_evaluator_deduplicated_query_denominator_count"] == 3
    assert summary["global_duplicate_relation_count"] == 2
    assert summary["within_query_duplicate_relation_count"] == 0
    assert summary["queries_with_duplicate_evaluator_relations"] == 0
    assert summary["cross_query_repeated_identity_count"] == 1
    assert summary["cross_query_repeated_relation_count"] == 3
    assert summary["raw_gold_per_query_distribution"] == {"2": 2}
    assert summary["safe_unique_gold_per_query_distribution"] == {"1": 1, "2": 1}
    assert query_rows[0]["current_evaluator_gold_count"] == 1
    assert query_rows[0]["current_evaluator_unique_identity_count"] == 1


def test_shared_identifier_with_same_type_conflict_stays_separate() -> None:
    queries = [
        EvalQuery(
            query_id="q1",
            query="one",
            gold_papers=[EvalGoldPaper(doi="10.1/shared", arxiv_id="2401.00001")],
        ),
        EvalQuery(
            query_id="q2",
            query="two",
            gold_papers=[EvalGoldPaper(doi="10.1/shared", arxiv_id="2401.00002")],
        ),
    ]

    gold, _queries, summary = analyze_gold_relations(queries)

    assert {row["terminal_status"] for row in gold} == {"identifier_conflict"}
    assert all(row["cross_record_conflict_relation_ids"] for row in gold)
    assert summary["global_unique_identity_count"] == 2


def test_identity_clusters_are_independent_of_query_input_order() -> None:
    first = EvalQuery(
        query_id="q1",
        query="one",
        gold_papers=[EvalGoldPaper(doi="10.1/shared")],
    )
    second = EvalQuery(
        query_id="q2",
        query="two",
        gold_papers=[
            EvalGoldPaper(doi="10.1/shared"),
            EvalGoldPaper(arxiv_id="2401.00001"),
        ],
    )

    forward, _query_rows, forward_summary = analyze_gold_relations([first, second])
    reverse, _query_rows, reverse_summary = analyze_gold_relations([second, first])

    def projection(rows: list[dict]) -> dict[str, tuple[str, str]]:
        return {
            row["relation_id"]: (
                row["terminal_status"],
                row["identity_cluster_id"],
            )
            for row in rows
        }

    assert projection(forward) == projection(reverse)
    for field in (
        "global_unique_identity_count",
        "global_duplicate_relation_count",
        "cross_query_repeated_identity_count",
    ):
        assert forward_summary[field] == reverse_summary[field]


def test_regression_gate_detects_gold_add_remove_identity_and_count_drift(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(
        tmp_path,
        [
            {
                "qid": "q1",
                "question": "query",
                "answer": ["One", "Two"],
                "answer_arxiv_id": ["2401.00001", "2401.00002"],
            }
        ],
    )
    gold, queries, summary = build_gold_identity_audit(manifest)
    baseline = tmp_path / "baseline"
    write_gold_identity_audit(baseline, gold, queries, summary)
    manifest["baseline"] = {
        "gold_rows_path": str(baseline / "gold_identity.jsonl"),
        "gold_rows_sha256": sha256_file(baseline / "gold_identity.jsonl"),
        "query_rows_path": str(baseline / "query_identity.jsonl"),
        "query_rows_sha256": sha256_file(baseline / "query_identity.jsonl"),
        "summary_path": str(baseline / "summary.json"),
        "summary_sha256": sha256_file(baseline / "summary.json"),
    }
    _write_json(manifest_path, manifest)

    dataset = Path(manifest["dataset"]["path"])
    _write_jsonl(
        dataset,
        [
            {
                "qid": "q1",
                "question": "query",
                "answer": ["Changed", "Added", "Third"],
                "answer_arxiv_id": ["2401.99999", "2401.00002", "2401.00003"],
            }
        ],
    )
    report = check_gold_identity_regression(manifest_path, tmp_path / "run")

    assert report["passed"] is False
    kinds = {item["kind"] for item in report["drifts"]}
    paths = {item["path"] for item in report["drifts"]}
    assert "gold_added" in kinds
    assert any("stable_identifiers" in path for path in paths)
    assert any(path.startswith("$.summary") for path in paths)

    _write_jsonl(
        dataset,
        [
            {
                "qid": "q1",
                "question": "query",
                "answer": ["One"],
                "answer_arxiv_id": ["2401.00001"],
            }
        ],
    )
    removed_report = check_gold_identity_regression(
        manifest_path, tmp_path / "run_removed"
    )
    assert "gold_removed" in {item["kind"] for item in removed_report["drifts"]}


def test_baseline_proposal_requires_explicit_token_and_reason(tmp_path: Path) -> None:
    manifest_path, _manifest_value = _manifest(
        tmp_path,
        [
            {
                "qid": "q1",
                "question": "query",
                "answer": ["One"],
                "answer_arxiv_id": ["2401.00001"],
            }
        ],
    )
    with pytest.raises(GoldIdentityAuditError, match="token rejected"):
        propose_gold_identity_baseline(
            manifest_path,
            tmp_path / "bad",
            approval_token="wrong",
            reason="freeze a reviewed evaluator baseline",
        )
    with pytest.raises(GoldIdentityAuditError, match="too short"):
        propose_gold_identity_baseline(
            manifest_path,
            tmp_path / "short",
            approval_token=BASELINE_APPROVAL_TOKEN,
            reason="short",
        )


def test_real_manifest_covers_all_1000_queries_and_2403_gold() -> None:
    manifest = json.loads(
        Path("benchmark/autoscholar_gold_identity_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    gold, queries, summary = build_gold_identity_audit(manifest)

    assert len(gold) == 2403
    assert len(queries) == 1000
    assert sum(summary["terminal_counts"].values()) == 2403
    assert summary["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "retrieval_invoked": False,
        "effectiveness_metrics_generated": False,
        "gold_scope": "offline_evaluator_input_only",
    }
    result = json.loads(
        Path("benchmark/autoscholar_gold_identity_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["identity"]["terminal_counts"] == summary["terminal_counts"]
    assert (
        result["evaluator_input"]["safe_deduplicated_query_denominator"]
        == summary["safe_evaluator_deduplicated_query_denominator_count"]
    )
    assert result["duplicates"]["global_unique_identity_count"] == summary[
        "global_unique_identity_count"
    ]


@pytest.mark.gold_identity_regression
def test_legacy_gold_identity_gate_preserves_and_reports_metric_migration(
    tmp_path: Path,
) -> None:
    report = check_gold_identity_regression(
        Path("benchmark/autoscholar_gold_identity_manifest.json"), tmp_path / "gate"
    )

    assert report["passed"] is False
    assert report["case_count"] == 1000
    assert report["gold_relation_count"] == 2403
    paths = {item["path"] for item in report["drifts"]}
    assert "$.fingerprints.evaluator_implementation_sha256" in paths
    assert any("AutoScholarQuery_test_752" in path for path in paths)
    assert any("AutoScholarQuery_test_773" in path for path in paths)
    assert report["execution"]["network_request_count"] == 0


def _manifest(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset, rows)
    query_manifest = tmp_path / "queries.jsonl"
    _write_jsonl(
        query_manifest,
        [
            {"query_id": row["qid"], "query": row["question"]}
            for row in rows
        ],
    )
    identity_path = Path("src/scholar_agent/core/identity.py").resolve()
    manifest = {
        "schema_version": "1",
        "gate": "autoscholar_gold_identity_regression",
        "dataset": {
            "name": "auto_scholar_query",
            "split": "test",
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "case_count": len(rows),
            "gold_relation_count": sum(len(row["answer"]) for row in rows),
        },
        "query_manifest": {
            "path": str(query_manifest),
            "sha256": sha256_file(query_manifest),
            "case_count": len(rows),
        },
        "identity_implementation": {
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
        },
        "evaluator_implementation": {
            "path": "src/scholar_agent/evaluation/metrics.py",
            "sha256": sha256_file(Path("src/scholar_agent/evaluation/metrics.py")),
        },
        "audit_implementation": {
            "path": "src/scholar_agent/evaluation/autoscholar_gold_identity.py",
            "sha256": sha256_file(
                Path("src/scholar_agent/evaluation/autoscholar_gold_identity.py")
            ),
        },
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
        },
        "baseline": {},
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
