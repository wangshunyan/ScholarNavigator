from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.evaluation.query_independence_audit import (
    _hit_concentration,
    _metric_diagnostics,
    build_independence_graph,
    check_query_independence_regression,
    lexical_jaccard,
    normalize_query_text,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads(
        (ROOT / "benchmark/autoscholar_query_independence_protocol.json").read_text(
            encoding="utf-8"
        )
    )


def _query(query_id: str, text: str) -> EvalQuery:
    return EvalQuery(
        query_id=query_id,
        query=text,
        gold_papers=[EvalGoldPaper(title=f"Gold paper for {query_id}", arxiv_id="2401.00001")],
    )


def _memberships(*query_ids: str) -> dict[str, list[str]]:
    return {query_id: [] for query_id in query_ids}


def test_unicode_normalization_and_exact_duplicate() -> None:
    assert normalize_query_text("ＡＢＣ—β＿Study") == "abc β study"
    queries = [_query("q1", "ＡＢＣ—β Study"), _query("q2", "abc β study")]
    _, edges, components = build_independence_graph(
        queries, [], _memberships("q1", "q2"), _protocol()
    )
    assert edges[0]["edge_types"] == ["normalized_exact_query"]
    assert components[0]["query_count"] == 2


def test_word_order_change_is_lexical_near_duplicate() -> None:
    queries = [
        _query(
            "q1",
            "graph neural networks molecular property prediction contrastive learning",
        ),
        _query(
            "q2",
            "contrastive learning for molecular property prediction with graph neural networks",
        ),
    ]
    _, edges, _ = build_independence_graph(
        queries, [], _memberships("q1", "q2"), _protocol()
    )
    assert edges[0]["edge_types"] == ["lexical_near_duplicate"]
    assert edges[0]["near_duplicate_similarity"] == 1.0


def test_short_query_protection_keeps_non_exact_queries_separate() -> None:
    queries = [
        _query("q1", "alpha beta gamma delta epsilon"),
        _query("q2", "epsilon delta gamma beta alpha"),
    ]
    assignments, edges, components = build_independence_graph(
        queries, [], _memberships("q1", "q2"), _protocol()
    )
    assert edges == []
    assert len(components) == 2
    assert {item["informative_token_count"] for item in assignments} == {5}


def test_jaccard_threshold_boundary_is_inclusive() -> None:
    left = "alpha beta gamma delta epsilon zeta eta theta left"
    right = "alpha beta gamma delta epsilon zeta eta theta right"
    assert lexical_jaccard(left.split(), right.split()) == pytest.approx(0.8)
    queries = [_query("q1", left), _query("q2", right)]
    _, edges, _ = build_independence_graph(
        queries, [], _memberships("q1", "q2"), _protocol()
    )
    assert edges[0]["near_duplicate_similarity"] == pytest.approx(0.8)


def test_shared_duplicate_gold_connects_different_queries() -> None:
    queries = [
        _query("q1", "completely unrelated retrieval wording alpha beta gamma"),
        _query("q2", "different scientific problem omega sigma lambda kappa"),
    ]
    identity = [
        {"query_id": query_id, "identity_cluster_id": "shared-gold"}
        for query_id in ("q1", "q2")
    ]
    _, edges, components = build_independence_graph(
        queries, identity, _memberships("q1", "q2"), _protocol()
    )
    assert edges[0]["edge_types"] == ["shared_gold_identity_cluster"]
    assert edges[0]["shared_gold_identity_cluster_ids"] == ["shared-gold"]
    assert components[0]["query_count"] == 2


def test_transitive_connectivity_and_input_order_independence() -> None:
    queries = [
        _query("q1", "alpha beta gamma delta epsilon zeta eta theta left"),
        _query("q2", "alpha beta gamma delta epsilon zeta eta theta right"),
        _query("q3", "unrelated biomedical evidence retrieval phrase tokens here"),
    ]
    identity = [
        {"query_id": "q2", "identity_cluster_id": "bridge"},
        {"query_id": "q3", "identity_cluster_id": "bridge"},
    ]
    memberships = _memberships("q1", "q2", "q3")
    forward = build_independence_graph(queries, identity, memberships, _protocol())
    reverse = build_independence_graph(
        list(reversed(queries)), list(reversed(identity)), memberships, _protocol()
    )
    assert forward == reverse
    assert len(forward[2]) == 1
    assert forward[2][0]["query_count"] == 3


def test_decontaminated_metrics_remove_auto_but_retain_scifact() -> None:
    assignments = [
        {
            "query_id": "auto",
            "component_id": "component:auto",
            "component_query_count": 2,
            "cross_stratum_contaminated": True,
        }
    ]
    common = {
        "evaluable_gold_count": 1,
        "candidate_gold_count": 1,
        "baseline": {
            "recall_at_20": 1.0,
            "f1_at_20": 0.5,
            "matched_gold_ids": ["gold"],
        },
        "experiment": {
            "recall_at_20": 1.0,
            "f1_at_20": 0.5,
            "matched_gold_ids": ["gold"],
        },
    }
    small_cases = [
        {"case_id": "auto", "dataset": "auto_dev", **common},
        {"case_id": "science", "dataset": "scifact", **common},
    ]
    rows, summary = _metric_diagnostics(assignments, small_cases, [])
    assert len(rows) == 2
    assert summary["existing65"]["full"]["case_count"] == 2
    assert summary["existing65"]["decontaminated"]["case_count"] == 1
    assert summary["existing65"]["decontaminated"]["dataset_case_counts"] == {
        "scifact": 1
    }
    concentration = _hit_concentration(rows)
    contaminated = concentration["existing65"]["by_independence"][
        "cross_stratum_contaminated"
    ]
    assert contaminated["case_count"] == 1
    assert contaminated["candidate_hit_rate"] == 1.0
    assert contaminated["baseline_final_hit_rate"] == 1.0


@pytest.mark.parametrize("section", ["dataset", "protocol"])
def test_regression_gate_detects_data_or_threshold_drift(
    tmp_path: Path, section: str
) -> None:
    manifest = json.loads(
        (ROOT / "benchmark/autoscholar_query_independence_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest[section]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = check_query_independence_regression(path, tmp_path / "gate")
    assert report["passed"] is False
    assert report["drifts"][0]["kind"] == "input_protocol_or_cluster_drift"


def test_regression_gate_detects_cluster_assignment_drift(tmp_path: Path) -> None:
    manifest = json.loads(
        (ROOT / "benchmark/autoscholar_query_independence_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        json.loads(line)
        for line in (
            ROOT
            / "benchmark/autoscholar_query_independence_baseline/query_assignments.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["component_id"] = "component:deliberate-drift"
    drifted = tmp_path / "query_assignments.jsonl"
    drifted.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest["baseline"]["query_assignments_path"] = str(drifted)
    manifest["baseline"]["query_assignments_sha256"] = sha256_file(drifted)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = check_query_independence_regression(manifest_path, tmp_path / "gate")
    assert report["passed"] is False
    assert any(
        item["path"].endswith("[0].component_id") for item in report["drifts"]
    )


@pytest.mark.query_independence_regression
def test_frozen_query_independence_gate(tmp_path: Path) -> None:
    report = check_query_independence_regression(
        ROOT / "benchmark/autoscholar_query_independence_manifest.json",
        tmp_path / "gate",
    )
    assert report["passed"] is True
    assert report["drift_count"] == 0
    summary = json.loads(
        (tmp_path / "gate/observed/summary.json").read_text(encoding="utf-8")
    )
    assert summary["query_count"] == 1000
    assert summary["component_count"] == 715
    assert summary["query_duplicates"]["component_count"] == 0
    assert summary["cross_stratum"]["contaminated_query_count"] == 237
    assert summary["execution"]["network_request_count"] == 0
    assert summary["execution"]["llm_request_count"] == 0
    assert summary["execution"]["snapshot_write_count"] == 0
