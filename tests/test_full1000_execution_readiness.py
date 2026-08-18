from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.full1000_execution_readiness import (
    EXIT_READY,
    EXIT_VIOLATION,
    Full1000ReadinessError,
    build_plan,
    canonical_json,
    dry_run,
    preflight,
    verify_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/full1000_execution_readiness_v1_protocol.json"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def plan(protocol: dict[str, object]) -> dict[str, object]:
    return build_plan(ROOT, protocol)


def test_population_and_shards_close_exactly(plan: dict[str, object]) -> None:
    population = plan["population"]
    assert population["count"] == 1000
    assert population["component_query_count"] == 1000
    identities = population["identities"]
    assert len(identities) == len(set(identities)) == 1000
    shards = plan["sharding"]["shards"]
    flattened = [identity for shard in shards for identity in shard["query_identities"]]
    assert len(flattened) == len(set(flattened)) == 1000
    assert set(flattened) == set(identities)
    assert {shard["query_count"] for shard in shards} == {50}


def test_legacy_partial_run_is_never_reused(plan: dict[str, object]) -> None:
    assert plan["resume"]["start_mode"] == "full_restart_all_1000"
    assert plan["resume"]["legacy_completed_query_count"] == 0
    assert plan["legacy_artifacts"]["reuse_as_completed"] is False
    assert plan["legacy_artifacts"]["record160_or_162_completed_query_count"] == 162


def test_resource_upper_bounds_are_derived_from_frozen_plan(plan: dict[str, object]) -> None:
    resources = plan["resource_upper_bounds"]
    assert resources["query_count"] == 1000
    assert resources["subquery_count"] == 2410
    assert resources["logical_source_request_upper"] == 9640
    assert resources["source_logical_request_upper"] == {
        "arxiv": 2410,
        "openalex": 2410,
        "pubmed": 2410,
        "semantic_scholar": 2410,
    }
    assert resources["retry_upper"] == 7230
    assert resources["http_request_attempt_upper"] == 19280
    assert resources["candidate_records_before_global_budget_upper"] == 192800
    assert resources["checkpoint_generation_selected_attempt_upper"] == 1040
    assert resources["checkpoint_generation_all_attempts_upper"] == 2080
    assert resources["checkpoint_file_all_attempts_upper"] == 12600
    assert resources["provider_token_upper"] == "not_available"
    assert resources["provider_cost_upper"] == "not_available"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value["population"]["identities"].pop(), "query_population_not_closed"),
        (
            lambda value: value["sharding"]["shards"][1]["query_identities"].append(
                value["sharding"]["shards"][0]["query_identities"][0]
            ),
            "shard_partition_not_closed",
        ),
        (
            lambda value: value["resume"].update({"start_mode": "resume_record160"}),
            "legacy_partial_run_reuse_forbidden",
        ),
        (
            lambda value: value["execution_contract"].update({"top_k": 21}),
            "plan_content_or_digest_drift",
        ),
        (
            lambda value: value["sharding"]["shards"][0]["attempts"][1].update(
                {"supersedes": "another-attempt"}
            ),
            "plan_content_or_digest_drift",
        ),
    ],
)
def test_plan_drift_fails_closed(
    plan: dict[str, object],
    protocol: dict[str, object],
    mutation,
    expected: str,
) -> None:
    changed = copy.deepcopy(plan)
    mutation(changed)
    report = verify_plan(ROOT, protocol, changed)
    assert report["exit_code"] == EXIT_VIOLATION
    assert expected in report["violations"]


def test_forbidden_evaluation_fields_are_rejected(protocol: dict[str, object]) -> None:
    changed = copy.deepcopy(protocol)
    changed["gold"] = "forbidden"
    with pytest.raises(Full1000ReadinessError, match="frozen_input_hash_drift|forbidden"):
        build_plan(ROOT, changed)


def test_plan_and_preflight_are_byte_deterministic(
    protocol: dict[str, object], plan: dict[str, object]
) -> None:
    second = build_plan(ROOT, protocol)
    assert canonical_json(plan) == canonical_json(second)
    first_report = preflight(ROOT, protocol, plan)
    second_report = preflight(ROOT, protocol, second)
    assert first_report["exit_code"] == EXIT_READY
    assert canonical_json(first_report) == canonical_json(second_report)
    assert first_report["network_status"] == "network_not_checked"


def test_fake_full1000_chain_is_closed_and_deterministic(plan: dict[str, object]) -> None:
    first = dry_run(plan)
    second = dry_run(plan)
    assert first["exit_code"] == EXIT_READY
    assert first["query_count"] == 1000
    assert all(first["stages"].values())
    assert sum(first["terminal_counts"].values()) == 1000
    assert first["execution"] == {
        "gold_or_qrels_loaded": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }
    assert canonical_json(first) == canonical_json(second)
