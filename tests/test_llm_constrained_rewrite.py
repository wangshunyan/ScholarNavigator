from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scholar_agent.agents.query_understanding import analyze_query
from scholar_agent.core.search_schemas import QueryConstraint
from scholar_agent.evaluation.llm_planning_snapshots import (
    LLMPlanningSnapshotRuntime,
    LLMPlanningSnapshotStore,
)


QUERY = (
    "Please find recent BERT studies for question answering without retrieval "
    "after 2020"
)
VALID_REWRITE = (
    "recent BERT studies question answering without retrieval after 2020 review"
)


def _response(rewrite: str = VALID_REWRITE) -> dict[str, object]:
    return {
        "input_summary": "BERT question answering with explicit exclusions",
        "rewritten_query": rewrite,
        "preserved_terms": ["BERT", "without retrieval after 2020"],
        "generic_synonyms_used": ["review"],
        "warnings": [],
    }


class Client:
    provider = "test_provider"
    model = "rewrite-test-v1"

    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = _response() if response is None else response
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.token_usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def chat_json(self, messages, *, temperature=0, timeout=None):  # noqa: ANN001
        self.calls.append(
            {"messages": messages, "temperature": temperature, "timeout": timeout}
        )
        if self.error is not None:
            raise self.error
        self.token_usage.prompt_tokens += 17
        self.token_usage.completion_tokens += 9
        self.token_usage.total_tokens += 26
        return deepcopy(self.response)


def _plan(client: Client | None, **kwargs):  # noqa: ANN003, ANN202
    return analyze_query(
        QUERY,
        query_planning_policy="llm_constrained_rewrite",
        llm_client=client,
        **kwargs,
    )


def test_rewrite_keeps_original_first_and_replaces_only_lowest_priority_query() -> None:
    baseline = analyze_query(QUERY)
    client = Client()

    rewritten = _plan(client)

    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0
    assert len(rewritten.subqueries) == len(baseline.subqueries)
    assert rewritten.subqueries[0] == baseline.subqueries[0]
    replace_index = rewritten.query_planning.constrained_rewrite_replaced_index
    assert replace_index == len(baseline.subqueries) - 1
    assert rewritten.subqueries[replace_index].query == VALID_REWRITE
    assert rewritten.subqueries[replace_index].priority == baseline.subqueries[replace_index].priority
    assert rewritten.subqueries[replace_index].source_hints == baseline.subqueries[replace_index].source_hints
    assert rewritten.subqueries[replace_index].combination_mode == baseline.subqueries[replace_index].combination_mode
    assert rewritten.query_planning.original_query_retained is True
    assert rewritten.query_planning.fallback_used is False


def test_rewrite_prompt_excludes_evaluation_candidates_and_answers() -> None:
    client = Client()
    _plan(client)
    rendered = str(client.calls[0]["messages"]).casefold()

    for forbidden in ("gold", "qrels", "case_id", "candidate_papers", "crosswalk"):
        assert forbidden not in rendered
    assert QUERY in str(client.calls[0]["messages"])


@pytest.mark.parametrize(
    ("rewrite", "reason"),
    [
        ("recent studies question answering without retrieval after 2020", "missing_protected_term"),
        ("recent BERT studies question answering review", "missing_protected_term"),
        (
            "recent BERT RoBERTa studies question answering without retrieval after 2020",
            "introduced_entity_or_term",
        ),
        (
            "recent BERT studies question answering without retrieval after 2020 10.1000/x",
            "suspicious_identifier",
        ),
        (QUERY, "duplicate_query"),
    ],
)
def test_rewrite_quality_gate_rejects_unsafe_or_constraint_losing_output(
    rewrite: str,
    reason: str,
) -> None:
    baseline = analyze_query(QUERY)
    plan = _plan(Client(_response(rewrite)))

    assert plan.subqueries == baseline.subqueries
    assert plan.query_planning.fallback_used is True
    assert plan.query_planning.fallback_reason == "rewrite_rejected"
    assert plan.query_planning.rejection_reasons == {reason: 1}
    assert plan.query_planning.constrained_rewrite_validation_rejections == [reason]


def test_rewrite_preserves_explicit_must_have_and_excluded_terms() -> None:
    constraints = QueryConstraint(
        must_include_terms=["clinical evidence"],
        exclude_terms=["animal studies"],
    )
    response = _response(
        "recent BERT question answering without retrieval after 2020 clinical evidence review"
    )

    plan = _plan(Client(response), explicit_constraints=constraints)

    assert plan.query_planning.fallback_used is True
    assert plan.query_planning.rejection_reasons == {"missing_protected_term": 1}


def test_invalid_schema_and_timeout_restore_current_rules_exactly() -> None:
    baseline = analyze_query(QUERY)

    invalid = _plan(Client({"rewritten_query": VALID_REWRITE}))
    timeout = _plan(Client(error=TimeoutError("request timeout")))

    assert invalid.subqueries == baseline.subqueries
    assert invalid.query_planning.fallback_reason == "invalid_schema"
    assert timeout.subqueries == baseline.subqueries
    assert timeout.query_planning.fallback_reason == "llm_timeout"


def test_no_replaceable_query_skips_model_and_keeps_budget() -> None:
    client = Client()

    plan = analyze_query(
        "x",
        query_planning_policy="llm_constrained_rewrite",
        llm_client=client,
    )

    assert client.calls == []
    assert len(plan.subqueries) == 1
    assert plan.query_planning.fallback_reason == "no_derived_query_to_replace"


def test_default_current_rules_never_calls_rewrite_model() -> None:
    client = Client()

    plan = analyze_query(QUERY, llm_client=client)

    assert client.calls == []
    assert plan.query_planning.policy == "current_rules"


def test_unicode_rewrite_preserves_ordered_constraints() -> None:
    query = "请查找近三年 BERT 多语言问答研究，不含检索"
    client = Client(
        {
            "input_summary": "多语言问答",
            "rewritten_query": "近三年 BERT 多语言问答 不含检索 综述",
            "preserved_terms": ["近三年", "BERT", "不含检索"],
            "generic_synonyms_used": ["综述"],
            "warnings": [],
        }
    )

    first = analyze_query(
        query,
        query_planning_policy="llm_constrained_rewrite",
        llm_client=client,
    )
    second = analyze_query(
        query,
        query_planning_policy="llm_constrained_rewrite",
        llm_client=Client(client.response),
    )

    assert first.query_planning.fallback_used is False
    assert first.subqueries == second.subqueries
    assert first.query_planning.constrained_rewrite_query == (
        "近三年 BERT 多语言问答 不含检索 综述"
    )


def test_record_then_replay_is_deterministic_and_offline(tmp_path) -> None:  # noqa: ANN001
    store = LLMPlanningSnapshotStore(tmp_path)
    record_runtime = LLMPlanningSnapshotRuntime(
        store,
        mode="record",
        group_name="llm_constrained_rewrite",
    )
    record_runtime.begin_case("case-1")
    client = Client()
    recorded = analyze_query(
        QUERY,
        query_planning_policy="llm_constrained_rewrite",
        llm_client=client,
        llm_planning_runtime=record_runtime,
    )

    replay_runtime = LLMPlanningSnapshotRuntime(
        store,
        mode="replay",
        group_name="llm_constrained_rewrite",
    )
    replay_runtime.begin_case("case-1")
    replayed = analyze_query(
        QUERY,
        query_planning_policy="llm_constrained_rewrite",
        llm_planning_runtime=replay_runtime,
    )

    assert client.calls and len(client.calls) == 1
    assert recorded.subqueries == replayed.subqueries
    assert replayed.query_planning.replayed is True
    assert replayed.query_planning.llm_call_attempted is False
    report = replay_runtime.finish_case()
    assert report.replay_execution_request_count == 0
    assert report.replay_execution_retry_count == 0
    assert report.replay_execution_network_wait_seconds == 0


def test_snapshot_plan_entry_uses_constrained_rewrite_policy(tmp_path) -> None:  # noqa: ANN001
    runtime = LLMPlanningSnapshotRuntime(
        LLMPlanningSnapshotStore(tmp_path),
        mode="replay",
        group_name="llm_constrained_rewrite",
    )
    runtime.begin_case("case-1")

    plan = analyze_query(
        QUERY,
        query_planning_policy="llm_constrained_rewrite",
        llm_planning_runtime=runtime,
    )
    entries = runtime.plan_entries()

    assert plan.query_planning.fallback_reason == "snapshot_missing"
    assert len(entries) == 1
    assert entries[0].query_planning_policy == "llm_constrained_rewrite"
    assert entries[0].llm_request["query_planning_policy"] == "llm_constrained_rewrite"
