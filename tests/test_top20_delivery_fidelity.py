from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers, PaperUrls
from scholar_agent.core.search_schemas import RankedPaper, RerankScoreBreakdown
from scholar_agent.evaluation.top20_delivery_fidelity import (
    Top20DeliveryError,
    assert_same_delivery,
    audit_export_eligibility,
    audit_frontend_contract,
    csv_roundtrip_row,
    delivery_projection,
    paginate_delivery,
    roundtrip_json,
    roundtrip_jsonl,
    validate_authority_mapping,
    validate_frontend_keys,
)
from scholar_agent.services.api_mapper import map_final_ranked_papers


ROOT = Path(__file__).resolve().parents[1]


def _ranked(
    title: str,
    *,
    rank: int,
    doi: str,
    category: str = "partially_relevant",
    year: int | None = 2024,
    url: str | None = "https://example.test/paper",
) -> RankedPaper:
    paper = Paper(
        title=title,
        authors=["A. Author"],
        year=year,
        venue="Venue",
        abstract="Unicode β abstract",
        identifiers=PaperIdentifiers(doi=doi),
        urls=PaperUrls(landing_page=url),
        sources=["arxiv"],
    )
    return RankedPaper(
        rank=rank,
        paper=paper,
        final_score=0.7,
        category=category,
        score_breakdown=RerankScoreBreakdown(
            relevance_score=0.7,
            authority_score=0.1,
            timeliness_score=0.2,
            metadata_score=0.3,
            final_score=0.7,
            relevance_weight=0.65,
            authority_weight=0.1,
            timeliness_weight=0.2,
            metadata_weight=0.05,
        ),
        ranking_reason="frozen",
    )


def test_api_json_jsonl_and_pagination_preserve_order_and_unicode() -> None:
    authoritative = [
        _ranked("Unicode β", rank=1, doi="10.1/one", year=None),
        _ranked("Second", rank=2, doi="10.1/two"),
    ]
    mapped = map_final_ranked_papers(authoritative)
    expected = delivery_projection(mapped)

    validate_authority_mapping(authoritative, mapped, query_identity="query")
    assert_same_delivery(
        expected, roundtrip_json(mapped), export_name="json", query_identity="query"
    )
    assert_same_delivery(
        expected,
        roundtrip_jsonl(mapped),
        export_name="jsonl",
        query_identity="query",
    )
    assert paginate_delivery(expected, page_size=1) == expected
    assert expected[0]["paper"]["year"] is None
    assert expected[0]["paper"]["title"] == "Unicode β"


def test_formal_selector_excludes_weak_and_never_pads_below_twenty() -> None:
    values = [
        _ranked("Returned", rank=1, doi="10.1/returned"),
        _ranked(
            "Weak", rank=2, doi="10.1/weak", category="weakly_relevant"
        ),
    ]
    mapped = map_final_ranked_papers(values)
    assert [item.paper.title for item in mapped] == ["Returned"]
    assert len(paginate_delivery(delivery_projection(mapped), page_size=20)) == 1


def test_order_member_and_field_drift_are_located() -> None:
    mapped = map_final_ranked_papers(
        [
            _ranked("One", rank=1, doi="10.1/one"),
            _ranked("Two", rank=2, doi="10.1/two"),
        ]
    )
    expected = delivery_projection(mapped)
    reversed_values = list(reversed(copy.deepcopy(expected)))
    with pytest.raises(Top20DeliveryError, match=r"\$\.authority_digest"):
        assert_same_delivery(
            expected,
            reversed_values,
            export_name="frontend",
            query_identity="query",
        )
    missing = copy.deepcopy(expected[:-1])
    with pytest.raises(Top20DeliveryError, match="delivery_count_drift"):
        assert_same_delivery(
            expected, missing, export_name="api", query_identity="query"
        )
    changed = copy.deepcopy(expected)
    changed[0]["paper"]["authors"] = ["Wrong paper"]
    with pytest.raises(Top20DeliveryError, match=r"\$\.paper\.authors"):
        assert_same_delivery(
            expected, changed, export_name="json", query_identity="query"
        )


def test_frontend_key_rejects_duplicate_or_missing_identity() -> None:
    mapped = delivery_projection(
        map_final_ranked_papers([_ranked("One", rank=1, doi="10.1/one")])
    )
    validate_frontend_keys(mapped, query_identity="query")
    duplicate = [mapped[0], copy.deepcopy(mapped[0])]
    with pytest.raises(Top20DeliveryError, match="frontend_key_collision"):
        validate_frontend_keys(duplicate, query_identity="query")
    missing = copy.deepcopy(mapped)
    missing[0]["result_identity"] = ""
    with pytest.raises(Top20DeliveryError, match="frontend_key_missing"):
        validate_frontend_keys(missing, query_identity="query")


def test_dangerous_url_is_not_clickable_but_authority_digest_remains() -> None:
    internal = _ranked(
        "Unsafe URL", rank=1, doi="10.1/unsafe", url="javascript:alert(1)"
    )
    mapped = map_final_ranked_papers([internal])
    assert mapped[0].paper.urls.landing_page is None
    validate_authority_mapping([internal], mapped, query_identity="query")
    assert len(mapped[0].authority_digest) == 64


def test_csv_formula_protection_is_rfc4180_roundtrip_only_not_an_export() -> None:
    assert csv_roundtrip_row(["=SUM(A1:A2)", "+cmd", "plain,quoted"]) == [
        "'=SUM(A1:A2)",
        "'+cmd",
        "plain,quoted",
    ]
    eligibility = audit_export_eligibility(
        ROOT,
        ROOT / "outputs/benchmark_runs/autoscholar_current_rules_full1000_3cd47c1_record_r1",
    )
    assert eligibility["csv_table"]["status"] == "unsupported_export"


def test_frontend_contract_uses_stable_identity_and_default_v1_path() -> None:
    result = audit_frontend_contract(ROOT)
    assert result["status"] == "passed"
    assert result["key"] == "result_identity"
    contract = json.loads(
        (ROOT / "benchmark/top20_delivery_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["policy_isolation"]["production_default"] == "original_index_v1"
    assert (
        contract["policy_isolation"]["deterministic_tiebreak_v2"]["default_enabled"]
        is False
    )


def test_explicit_order_inputs_roundtrip_independently_without_v1_v2_mix() -> None:
    first = _ranked("First", rank=1, doi="10.1/first")
    second = _ranked("Second", rank=2, doi="10.1/second")
    audit_second = _ranked("Second", rank=1, doi="10.1/second")
    audit_first = _ranked("First", rank=2, doi="10.1/first")
    current = delivery_projection(map_final_ranked_papers([first, second]))
    explicit_audit = delivery_projection(
        map_final_ranked_papers([audit_second, audit_first])
    )
    assert [item["result_identity"] for item in current] != [
        item["result_identity"] for item in explicit_audit
    ]
    assert roundtrip_json(map_final_ranked_papers([first, second])) == current
    assert (
        roundtrip_json(map_final_ranked_papers([audit_second, audit_first]))
        == explicit_audit
    )
