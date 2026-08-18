from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from scholar_agent.evaluation import llm_relevance_judging as judging
from scholar_agent.evaluation.llm_relevance_judging import (
    EXIT_COMPLETED,
    EXIT_INCOMPLETE,
    EXIT_INTEGRITY_VIOLATION,
    HARDENED_CONTRACT_VERSION,
    LLMRelevanceJudgingError,
    load_protocol,
    prepare_run,
    run_adjudication,
    run_judge_round,
    score_run,
    verify_run,
)
from scholar_agent.llm.provider import LLMResponseError


ROOT = Path(__file__).resolve().parents[1]
V1_PROTOCOL_PATH = ROOT / "benchmark" / "llm_relevance_judging_v1_protocol.json"
PROTOCOL_PATH = ROOT / "benchmark" / "llm_relevance_judging_v1_1_protocol.json"
INCOMPLETE_EVIDENCE = (
    ROOT / "benchmark" / "llm_relevance_judging_v1_1_record160_incomplete"
)
ITEM_PATTERN = re.compile(r'"item_id":"(item:[0-9a-f]{64})"')


class SingleItemFakeLLM:
    base_url = "https://offline.invalid/v1"
    model = "offline-fake-model"

    def __init__(
        self,
        *,
        failures_before_success: int = 0,
        failure_mode: str = "schema",
        label: str = "not_relevant",
    ) -> None:
        self.failures_before_success = failures_before_success
        self.failure_mode = failure_mode
        self.label = label
        self.call_count = 0
        self.item_counts: list[int] = []
        self.last_call_usage_fields: dict[str, int] | None = None
        self.last_call_diagnostics: dict[str, Any] | None = None

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        assert temperature == 0
        assert [item["role"] for item in messages] == ["system", "user"]
        item_ids = ITEM_PATTERN.findall(messages[1]["content"])
        self.item_counts.append(len(item_ids))
        self.call_count += 1
        self.last_call_usage_fields = {
            "prompt_tokens": 7,
            "completion_tokens": 5,
            "total_tokens": 12,
        }
        self.last_call_diagnostics = {
            "mode": "structured_json",
            "http_attempts": 1,
            "latency_ms": 0,
            "fallback_reason": None,
        }
        if self.call_count <= self.failures_before_success:
            if self.failure_mode == "extra_text":
                raise LLMResponseError("llm_invalid_json_content")
            if self.failure_mode == "missing_key":
                return {"labels": [{"item_id": item_ids[0]}]}
            if self.failure_mode == "illegal_enum":
                return {
                    "labels": [
                        {
                            "item_id": item_ids[0],
                            "label": "maybe",
                            "evidence": "Visible evidence only.",
                        }
                    ]
                }
            if self.failure_mode == "duplicate":
                row = {
                    "item_id": item_ids[0],
                    "label": self.label,
                    "evidence": "Visible evidence only.",
                }
                return {"labels": [row, row]}
            return {"labels": []}
        if "adjudicator" in messages[0]["content"]:
            return {
                "decisions": [
                    {
                        "item_id": item_ids[0],
                        "final_label": self.label,
                        "evidence": "Visible evidence resolves the rubric disagreement.",
                    }
                ]
            }
        return {
            "labels": [
                {
                    "item_id": item_ids[0],
                    "label": self.label,
                    "evidence": "Visible evidence supports the frozen rubric label.",
                }
            ]
        }


def _protocol() -> dict[str, Any]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _binding(*, model: str = "offline-fake-model") -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract": HARDENED_CONTRACT_VERSION,
        "provider": "offline_fake",
        "model": model,
        "provider_endpoint_sha256": "0" * 64,
        "request_options": {"max_tokens": 1024, "timeout_seconds": 60.0},
    }


@pytest.mark.llm_relevance_judging_regression
def test_v1_1_protocol_is_single_item_and_preserves_frozen_semantics(
    tmp_path: Path,
) -> None:
    hardened = _protocol()
    legacy = load_protocol(V1_PROTOCOL_PATH, repository_root=ROOT)
    assert hardened["contract"] == HARDENED_CONTRACT_VERSION
    assert hardened["judge"]["batch_size"] == 1
    assert hardened["judge"]["structured_output"] == {
        "client_validation": "strict_pydantic_extra_forbid",
        "malformed_output_recovery": "forbidden",
        "provider_mode": "native_json_object",
        "top_level_adjudication_key": "decisions",
        "top_level_judge_key": "labels",
    }
    assert hardened["rubric"] == legacy["rubric"]
    assert hardened["statistics"] == legacy["statistics"]
    assert hardened["blinding"]["allowed_item_fields"] == legacy["blinding"][
        "allowed_item_fields"
    ]
    hardened_dir = tmp_path / "hardened"
    legacy_dir = tmp_path / "legacy"
    prepare_run(hardened, repository_root=ROOT, run_dir=hardened_dir)
    prepare_run(legacy, repository_root=ROOT, run_dir=legacy_dir)
    hardened_ids = {
        json.loads(line)["item_id"]
        for line in (hardened_dir / "blind_view.jsonl").read_text().splitlines()
    }
    legacy_ids = {
        json.loads(line)["item_id"]
        for line in (legacy_dir / "blind_view.jsonl").read_text().splitlines()
    }
    assert len(hardened_ids) == len(legacy_ids) == 471
    assert hardened_ids.isdisjoint(legacy_ids)


@pytest.mark.llm_relevance_judging_regression
def test_v1_1_isolates_every_item_and_resume_does_not_rebill(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    run_dir = tmp_path / "run"
    prepare_run(protocol, repository_root=ROOT, run_dir=run_dir)
    clients: list[SingleItemFakeLLM] = []

    def factory() -> SingleItemFakeLLM:
        client = SingleItemFakeLLM()
        clients.append(client)
        return client

    report = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=SingleItemFakeLLM(),
        runtime_binding=_binding(),
        client_factory=factory,
    )
    assert report["exit_code"] == EXIT_COMPLETED
    assert report["details"]["batch_count"] == 471
    assert len(clients) == 471
    assert all(client.item_counts == [1] for client in clients)

    before = len(clients)
    repeated = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=SingleItemFakeLLM(),
        runtime_binding=_binding(),
        client_factory=factory,
    )
    assert repeated["details"]["called_batch_count"] == 0
    assert len(clients) == before


@pytest.mark.llm_relevance_judging_regression
def test_v1_1_applies_uniform_retry_and_locks_only_strict_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol()
    run_dir = tmp_path / "run"
    prepare_run(protocol, repository_root=ROOT, run_dir=run_dir)
    sleeps: list[float] = []
    monkeypatch.setattr(judging.time, "sleep", sleeps.append)
    clients: list[SingleItemFakeLLM] = []

    def factory() -> SingleItemFakeLLM:
        client = SingleItemFakeLLM(failures_before_success=1)
        clients.append(client)
        return client

    report = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=SingleItemFakeLLM(),
        runtime_binding=_binding(),
        max_batches=3,
        client_factory=factory,
    )
    assert report["exit_code"] == EXIT_INCOMPLETE
    assert report["details"]["locked_item_count"] == 3
    assert len(clients) == 3
    assert all(client.call_count == 2 for client in clients)
    assert sleeps == [1.0, 1.0, 1.0]
    for path in sorted((run_dir / "rounds/independent_1").glob("*.json")):
        value = json.loads(path.read_text())
        assert [row["status"] for row in value["attempts"]] == [
            "schema_failure",
            "success",
        ]
        assert value["status"] == "locked_success"


@pytest.mark.llm_relevance_judging_regression
@pytest.mark.parametrize(
    ("failure_mode", "expected_status"),
    [
        ("schema", "schema_failure"),
        ("missing_key", "schema_failure"),
        ("illegal_enum", "schema_failure"),
        ("duplicate", "schema_failure"),
        ("extra_text", "provider_failure"),
    ],
)
def test_v1_1_never_repairs_malformed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    expected_status: str,
) -> None:
    protocol = _protocol()
    run_dir = tmp_path / failure_mode
    prepare_run(protocol, repository_root=ROOT, run_dir=run_dir)
    monkeypatch.setattr(judging.time, "sleep", lambda _seconds: None)
    client = SingleItemFakeLLM(
        failures_before_success=99,
        failure_mode=failure_mode,
    )
    report = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=client,
        runtime_binding=_binding(),
        max_batches=1,
    )
    assert report["exit_code"] == (
        EXIT_INTEGRITY_VIOLATION
        if expected_status == "schema_failure"
        else EXIT_INCOMPLETE
    )
    value = json.loads(
        (run_dir / "rounds/independent_1/batch_00000.json").read_text()
    )
    assert [row["status"] for row in value["attempts"]] == [
        expected_status,
        expected_status,
    ]
    assert value["locked_response"] is None


@pytest.mark.llm_relevance_judging_regression
def test_v1_1_rejects_cross_version_and_runtime_binding_mixing(
    tmp_path: Path,
) -> None:
    hardened = _protocol()
    legacy = load_protocol(V1_PROTOCOL_PATH, repository_root=ROOT)
    run_dir = tmp_path / "run"
    prepare_run(hardened, repository_root=ROOT, run_dir=run_dir)
    client = SingleItemFakeLLM()
    with pytest.raises(LLMRelevanceJudgingError):
        run_judge_round(
            legacy,
            repository_root=ROOT,
            run_dir=run_dir,
            round_id="independent_1",
            client=client,
            runtime_binding={**_binding(), "contract": legacy["contract"]},
            max_batches=1,
        )
    assert client.call_count == 0

    run_judge_round(
        hardened,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=client,
        runtime_binding=_binding(),
        max_batches=1,
    )
    with pytest.raises(LLMRelevanceJudgingError, match="runtime_binding_drift"):
        run_judge_round(
            hardened,
            repository_root=ROOT,
            run_dir=run_dir,
            round_id="independent_1",
            client=SingleItemFakeLLM(),
            runtime_binding=_binding(model="different-model"),
            max_batches=1,
        )


@pytest.mark.llm_relevance_judging_regression
def test_v1_1_completes_two_rounds_adjudication_and_scoring(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    run_dir = tmp_path / "complete"
    prepare_run(protocol, repository_root=ROOT, run_dir=run_dir)

    def relevant_factory() -> SingleItemFakeLLM:
        return SingleItemFakeLLM(label="relevant")

    def negative_factory() -> SingleItemFakeLLM:
        return SingleItemFakeLLM(label="not_relevant")

    first = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=SingleItemFakeLLM(label="relevant"),
        runtime_binding=_binding(),
        client_factory=relevant_factory,
    )
    second = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_2",
        client=SingleItemFakeLLM(label="not_relevant"),
        runtime_binding=_binding(),
        client_factory=negative_factory,
    )
    assert first["exit_code"] == second["exit_code"] == EXIT_COMPLETED
    adjudication = run_adjudication(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        client=SingleItemFakeLLM(label="relevant"),
        runtime_binding=_binding(),
        client_factory=relevant_factory,
    )
    assert adjudication["exit_code"] == EXIT_COMPLETED
    assert adjudication["details"]["disagreement_count"] == 471
    assert verify_run(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
    )["exit_code"] == EXIT_COMPLETED
    result = score_run(protocol, repository_root=ROOT, run_dir=run_dir)
    assert result["contract"] == HARDENED_CONTRACT_VERSION
    assert result["review"]["coverage"]["resolved_item_count"] == 471
    assert result["review"]["agreement"]["adjudicated_count"] == 471
    assert result["review"]["absolute_precision_at_20"] == {
        "baseline": None,
        "candidate": None,
        "reason": "shared_top20_items_not_present_in_frozen_change_only_package",
    }


def test_v1_1_batch_truncation_is_rejected_without_partial_lock() -> None:
    item_ids = ["item:" + f"{index:064x}" for index in range(8)]
    rows = [
        {
            "item_id": item_id,
            "label": "relevant",
            "evidence": "Visible evidence.",
        }
        for item_id in item_ids[:-1]
    ]
    with pytest.raises(LLMRelevanceJudgingError, match="batch_mismatch"):
        judging._validate_response(
            {"labels": rows},
            expected_ids=item_ids,
            mode="judge",
        )


@pytest.mark.llm_relevance_judging_regression
def test_v1_1_actual_incomplete_evidence_never_publishes_partial_labels() -> None:
    manifest = json.loads((INCOMPLETE_EVIDENCE / "manifest.json").read_text())
    status = json.loads((INCOMPLETE_EVIDENCE / "status.json").read_text())
    calls = [
        json.loads(line)
        for line in (INCOMPLETE_EVIDENCE / "calls.jsonl").read_text().splitlines()
        if line
    ]

    assert manifest["contract"] == status["contract"] == HARDENED_CONTRACT_VERSION
    assert manifest["state"] == "incomplete_no_unblinding"
    assert manifest["labels_file"] is None
    assert manifest["statistics_file"] is None
    assert status["labels_locked"] is False
    assert status["private_mapping_opened"] is False
    assert status["statistics"] is None
    assert len(calls) == 471
    assert all(row["phase"] == "independent_1" for row in calls)
    assert all(row["item_count"] == 1 for row in calls)
    assert all("response" not in row for row in calls)
    assert status["usage"]["logical_call_count"] == 689
    assert status["usage"]["successful_logical_call_count"] == 291
    assert status["usage"]["failed_logical_call_count"] == 398
