from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.evaluation.query_gold_leakage_audit import (
    LEVELS,
    _audit_auto_relations,
    check_query_gold_leakage_regression,
    detect_query_gold_leakage,
    normalize_leakage_text,
    write_query_gold_leakage_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads(
        (ROOT / "benchmark/autoscholar_query_gold_leakage_protocol.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.parametrize(
    "query",
    [
        "See arXiv:2401.00001v2 for details",
        "See https://arxiv.org/abs/2401.00001",
        "See https://arxiv.org/pdf/2401.00001v3.pdf",
        "The identifier is 2401.00001.",
    ],
)
def test_arxiv_identifier_variants_are_exact_leaks(query: str) -> None:
    result = detect_query_gold_leakage(
        query,
        EvalGoldPaper(title="Unrelated sufficiently long title for protection", arxiv_id="2401.00001"),
        _protocol(),
    )
    assert result["leakage_level"] == "identifier_or_url_exact"
    assert result["rule_hits"][0]["identifier_type"] == "arxiv"


def test_doi_url_is_exact_but_partial_identifier_is_not() -> None:
    gold = EvalGoldPaper(
        title="A sufficiently descriptive paper title for testing",
        doi="https://doi.org/10.1000/ABC.Def",
    )
    exact = detect_query_gold_leakage(
        "Discuss doi.org/10.1000/abc.def in context", gold, _protocol()
    )
    partial = detect_query_gold_leakage(
        "Discuss 10.1000/abc in context", gold, _protocol()
    )
    assert exact["leakage_level"] == "identifier_or_url_exact"
    assert partial["leakage_level"] == "no_detected_leakage"


def test_unicode_title_normalization_and_quoted_priority() -> None:
    title = "β‑VAE：Learning Basic Visual Concepts with a Constrained Variational Framework"
    full = detect_query_gold_leakage(
        "Explain β-VAE learning basic visual concepts with a constrained variational framework",
        EvalGoldPaper(title=title),
        _protocol(),
    )
    quoted = detect_query_gold_leakage(
        f"What is shown in “{title}”?", EvalGoldPaper(title=title), _protocol()
    )
    assert normalize_leakage_text("β‑VAE") == "β vae"
    assert full["leakage_level"] == "normalized_title_full"
    assert quoted["leakage_level"] == "quoted_title_exact"
    assert {item["rule"] for item in quoted["rule_hits"]} >= {
        "quoted_title_exact",
        "normalized_title_full",
    }


def test_short_title_protection_and_partial_overlap() -> None:
    short = detect_query_gold_leakage(
        "Deep Learning", EvalGoldPaper(title="Deep Learning"), _protocol()
    )
    partial = detect_query_gold_leakage(
        "robust spectral graph networks",
        EvalGoldPaper(
            title="Robust spectral graph networks for molecular prediction"
        ),
        _protocol(),
    )
    assert short["leakage_level"] == "no_detected_leakage"
    assert short["title_diagnostics"]["skip_reason"] == "short_title_protection"
    assert partial["leakage_level"] == "no_detected_leakage"


def test_preregistered_high_coverage_boundary() -> None:
    gold = EvalGoldPaper(
        title="Robust spectral graph networks for molecular prediction systems"
    )
    hit = detect_query_gold_leakage(
        "Which robust spectral graph networks improve molecular prediction?",
        gold,
        _protocol(),
    )
    miss = detect_query_gold_leakage(
        "Which robust spectral graph networks are useful?", gold, _protocol()
    )
    assert hit["leakage_level"] == "title_token_high_coverage"
    coverage = next(
        item["coverage"]
        for item in hit["rule_hits"]
        if item["rule"] == "title_token_high_coverage"
    )
    assert coverage == pytest.approx(6 / 7)
    assert miss["leakage_level"] == "no_detected_leakage"


def test_duplicate_gold_relations_are_closed_and_order_independent() -> None:
    queries = [
        EvalQuery(
            query_id="q1",
            query="See arXiv 2401.00001",
            gold_papers=[EvalGoldPaper(title="A long enough title for q1", arxiv_id="2401.00001")],
        ),
        EvalQuery(
            query_id="q2",
            query="A non-leaking question",
            gold_papers=[EvalGoldPaper(title="A long enough title for q2", arxiv_id="2401.00001")],
        ),
    ]
    identity = [
        {
            "relation_id": f"{query.query_id}::gold[0]",
            "identity_cluster_id": "shared",
            "duplicate_across_queries": True,
            "identity_cluster_query_count": 2,
        }
        for query in queries
    ]
    forward, _ = _audit_auto_relations(queries, identity, _protocol())
    reverse, _ = _audit_auto_relations(list(reversed(queries)), identity, _protocol())
    project = lambda rows: {
        row["relation_id"]: row["leakage_level"] for row in rows
    }
    assert project(forward) == project(reverse) == {
        "q1::gold[0]": "identifier_or_url_exact",
        "q2::gold[0]": "no_detected_leakage",
    }


def test_same_query_duplicate_gold_relations_keep_unique_terminals() -> None:
    query = EvalQuery(
        query_id="q1",
        query="See arXiv 2401.00001",
        gold_papers=[
            EvalGoldPaper(title="First presentation of the shared paper", arxiv_id="2401.00001"),
            EvalGoldPaper(title="Second presentation of the shared paper", arxiv_id="2401.00001"),
        ],
    )
    identity = [
        {
            "relation_id": f"q1::gold[{index}]",
            "identity_cluster_id": "shared",
            "duplicate_across_queries": False,
            "identity_cluster_query_count": 1,
        }
        for index in range(2)
    ]
    relations, queries = _audit_auto_relations([query], identity, _protocol())
    assert [row["relation_id"] for row in relations] == [
        "q1::gold[0]",
        "q1::gold[1]",
    ]
    assert queries[0]["gold_relation_count"] == 2
    assert queries[0]["leakage_level_counts"]["identifier_or_url_exact"] == 2


def test_writer_is_byte_deterministic(tmp_path: Path) -> None:
    relations = [{"relation_id": "q::gold[0]", "leakage_level": LEVELS[-1]}]
    queries = [{"query_id": "q", "leakage_level": LEVELS[-1]}]
    summary = {"relation_count": 1, "query_count": 1}
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_query_gold_leakage_audit(first, relations, queries, summary)
    write_query_gold_leakage_audit(second, relations, queries, summary)
    for name in ("relations.jsonl", "queries.jsonl", "summary.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize("section", ["protocol", "dataset"])
def test_regression_gate_detects_protocol_or_data_drift(
    tmp_path: Path, section: str
) -> None:
    manifest = json.loads(
        (ROOT / "benchmark/autoscholar_query_gold_leakage_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest[section]["sha256"] = "0" * 64
    drifted = tmp_path / "manifest.json"
    drifted.write_text(json.dumps(manifest), encoding="utf-8")
    report = check_query_gold_leakage_regression(drifted, tmp_path / "gate")
    assert report["passed"] is False
    assert report["drifts"][0]["kind"] == "input_or_protocol_drift"


@pytest.mark.query_gold_leakage_regression
def test_frozen_autoscholar_query_gold_leakage_gate(tmp_path: Path) -> None:
    report = check_query_gold_leakage_regression(
        ROOT / "benchmark/autoscholar_query_gold_leakage_manifest.json",
        tmp_path / "gate",
    )
    assert report == {
        "schema_version": "1",
        "gate": "autoscholar_query_gold_leakage_regression",
        "passed": True,
        "drift_count": 0,
        "drifts": [],
        "execution": {
            "network_request_count": 0,
            "llm_request_count": 0,
            "snapshot_write_count": 0,
        },
    }
    summary = json.loads(
        (
            tmp_path
            / "gate"
            / "observed"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["relation_count"] == 2403
    assert sum(summary["relation_level_counts"].values()) == 2403
    assert summary["query_count"] == 1000
    assert sum(summary["query_level_counts"].values()) == 1000
    assert summary["execution"]["network_request_count"] == 0
    assert summary["execution"]["llm_request_count"] == 0
    assert summary["execution"]["snapshot_write_count"] == 0
