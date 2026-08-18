from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from scholar_agent.evaluation.llm_relevance_judging import (
    CONTRACT_VERSION,
    EXIT_COMPLETED,
    EXIT_INCOMPLETE,
    EXIT_INTEGRITY_VIOLATION,
    LLMRelevanceJudgingError,
    load_protocol,
    prepare_run,
    publish_incomplete_audit,
    run_adjudication,
    run_judge_round,
    score_run,
    verify_run,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark" / "llm_relevance_judging_v1_protocol.json"
ITEM_PATTERN = re.compile(r'"item_id":"(item:[0-9a-f]{64})"')


class FakeLLM:
    base_url = "https://offline.invalid/v1"
    model = "offline-fake-model"

    def __init__(
        self,
        *,
        flip_first_per_batch: bool = False,
        malformed: bool = False,
        fail_calls: int = 0,
    ) -> None:
        self.flip_first_per_batch = flip_first_per_batch
        self.malformed = malformed
        self.fail_calls = fail_calls
        self.call_count = 0
        self.last_call_usage: dict[str, int] | None = None
        self.last_call_diagnostics: dict[str, Any] | None = None
        self.observed_payloads: list[dict[str, Any]] = []

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        assert temperature == 0
        assert [message["role"] for message in messages] == ["system", "user"]
        payload = json.loads(messages[1]["content"].split("\n\n", 1)[1])
        self.observed_payloads.append(payload)
        ids = ITEM_PATTERN.findall(messages[1]["content"])
        self.call_count += 1
        self.last_call_usage = None
        self.last_call_diagnostics = None
        if self.call_count <= self.fail_calls:
            raise RuntimeError("controlled fake provider failure")
        self.last_call_usage = {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        }
        self.last_call_diagnostics = {
            "mode": "offline_fake_json",
            "http_attempts": 1,
            "latency_ms": 0,
            "fallback_reason": None,
        }
        if self.malformed:
            return {"labels": []}
        if "adjudicator" in messages[0]["content"]:
            return {
                "decisions": [
                    {
                        "item_id": item_id,
                        "final_label": "relevant",
                        "evidence": "Visible metadata directly addresses the query.",
                    }
                    for item_id in ids
                ]
            }
        return {
            "labels": [
                {
                    "item_id": item_id,
                    "label": (
                        "relevant"
                        if self.flip_first_per_batch and index == 0
                        else "not_relevant"
                    ),
                    "evidence": "Visible metadata does not materially address the query.",
                }
                for index, item_id in enumerate(ids)
            ]
        }


def _protocol() -> dict[str, Any]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _binding() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "contract": CONTRACT_VERSION,
        "provider": "offline_fake",
        "model": "offline-fake-model",
        "provider_endpoint_sha256": "0" * 64,
        "request_options": {"max_tokens": 512, "timeout_seconds": 15.0},
    }


@pytest.mark.llm_relevance_judging_regression
def test_prepare_is_deterministic_and_strictly_blinded(tmp_path: Path) -> None:
    protocol = _protocol()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_report = prepare_run(protocol, repository_root=ROOT, run_dir=first)
    second_report = prepare_run(protocol, repository_root=ROOT, run_dir=second)

    assert first_report == second_report
    assert first_report["details"]["item_count"] == 471
    assert first_report["details"]["current_package_item_count"] == 439
    assert first_report["details"]["prior_package_item_count"] == 32
    assert (first / "blind_view.jsonl").read_bytes() == (
        second / "blind_view.jsonl"
    ).read_bytes()
    rows = [
        json.loads(line)
        for line in (first / "blind_view.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        set(row) == {"item_id", "query", "title", "abstract", "year"}
        for row in rows
    )
    assert all(row["item_id"].startswith("item:") for row in rows)
    assert not (first / "private").exists()
    manifest = json.loads(
        (first / "prepared_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["blinding"]["private_mapping_written"] is False


@pytest.mark.llm_relevance_judging_regression
def test_fake_llm_full_workflow_is_resumable_locked_and_deterministic(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    run_dir = tmp_path / "run"
    prepare_run(protocol, repository_root=ROOT, run_dir=run_dir)
    first = FakeLLM()
    second = FakeLLM(flip_first_per_batch=True)
    first_clients: list[FakeLLM] = []
    second_clients: list[FakeLLM] = []

    def first_factory() -> FakeLLM:
        current = FakeLLM()
        first_clients.append(current)
        return current

    def second_factory() -> FakeLLM:
        current = FakeLLM(flip_first_per_batch=True)
        second_clients.append(current)
        return current

    first_report = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=first,
        runtime_binding=_binding(),
        client_factory=first_factory,
    )
    second_report = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_2",
        client=second,
        runtime_binding=_binding(),
        client_factory=second_factory,
    )
    assert first_report["exit_code"] == EXIT_COMPLETED
    assert second_report["exit_code"] == EXIT_COMPLETED
    assert len(first_clients) == len(second_clients) == 59
    assert all(client.call_count == 1 for client in [*first_clients, *second_clients])
    assert all(
        set(item) == {"item_id", "query", "title", "abstract", "year"}
        for client in first_clients
        for payload in client.observed_payloads
        for item in payload["payload"]["items"]
    )

    locked_client_count = len(first_clients)
    repeated = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=first,
        runtime_binding=_binding(),
        client_factory=first_factory,
    )
    assert repeated["details"]["called_batch_count"] == 0
    assert len(first_clients) == locked_client_count

    adjudicator = FakeLLM()
    adjudicator_clients: list[FakeLLM] = []

    def adjudicator_factory() -> FakeLLM:
        current = FakeLLM()
        adjudicator_clients.append(current)
        return current

    adjudication = run_adjudication(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        client=adjudicator,
        runtime_binding=_binding(),
        client_factory=adjudicator_factory,
    )
    assert adjudication["exit_code"] == EXIT_COMPLETED
    assert adjudication["details"]["disagreement_count"] == 59
    assert len(adjudicator_clients) == 8
    verification = verify_run(
        protocol, repository_root=ROOT, run_dir=run_dir
    )
    assert verification["exit_code"] == EXIT_COMPLETED

    publish_dir = tmp_path / "published"
    first_score = score_run(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        publish_dir=publish_dir,
    )
    score_bytes = (run_dir / "score.json").read_bytes()
    second_score = score_run(
        protocol, repository_root=ROOT, run_dir=run_dir
    )
    assert first_score == second_score
    assert score_bytes == (run_dir / "score.json").read_bytes()
    assert first_score["score_scope"] == "internal_llm_proxy_not_human_or_official"
    assert first_score["review"]["coverage"]["resolved_item_count"] == 471
    assert first_score["review"]["agreement"]["adjudicated_count"] == 59
    assert first_score["review"]["absolute_precision_at_20"]["baseline"] is None
    calls = [
        json.loads(line)
        for line in (publish_dir / "calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(calls) == 126
    assert all(
        "input_sha256" in attempt
        for row in calls
        for attempt in row["attempts"]
    )
    assert all("response" not in row for row in calls)
    assert first_score["usage"]["provider_cost"]["status"] == "not_available"


@pytest.mark.llm_relevance_judging_regression
def test_failures_resume_and_schema_mismatch_is_not_silently_dropped(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    run_dir = tmp_path / "resume"
    prepare_run(protocol, repository_root=ROOT, run_dir=run_dir)
    client = FakeLLM(fail_calls=1)

    first = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=client,
        runtime_binding=_binding(),
        max_batches=1,
    )
    assert first["exit_code"] == EXIT_INCOMPLETE
    batch = json.loads(
        (run_dir / "rounds/independent_1/batch_00000.json").read_text(
            encoding="utf-8"
        )
    )
    assert batch["attempts"][0]["status"] == "provider_failure"
    assert batch["attempts"][0]["usage"]["status"] == "not_available"

    second = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        round_id="independent_1",
        client=client,
        runtime_binding=_binding(),
        max_batches=1,
    )
    assert second["exit_code"] == EXIT_INCOMPLETE
    batch = json.loads(
        (run_dir / "rounds/independent_1/batch_00000.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["status"] for item in batch["attempts"]] == [
        "provider_failure",
        "success",
    ]
    assert batch["status"] == "locked_success"

    malformed_dir = tmp_path / "malformed"
    prepare_run(protocol, repository_root=ROOT, run_dir=malformed_dir)
    malformed = FakeLLM(malformed=True)
    run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=malformed_dir,
        round_id="independent_1",
        client=malformed,
        runtime_binding=_binding(),
        max_batches=1,
    )
    terminal = run_judge_round(
        protocol,
        repository_root=ROOT,
        run_dir=malformed_dir,
        round_id="independent_1",
        client=malformed,
        runtime_binding=_binding(),
        max_batches=1,
    )
    assert terminal["exit_code"] == EXIT_INTEGRITY_VIOLATION
    assert terminal["details"]["terminal_schema_failure_count"] == 1
    published = tmp_path / "incomplete-published"
    incomplete = publish_incomplete_audit(
        protocol,
        repository_root=ROOT,
        run_dir=malformed_dir,
        publish_dir=published,
    )
    assert incomplete["statistics"] is None
    assert incomplete["labels_locked"] is False
    assert (published / "calls.jsonl").is_file()
    assert not (published / "labels.jsonl").exists()


@pytest.mark.llm_relevance_judging_regression
def test_unblinding_requires_lock_and_locked_response_tampering_is_detected(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    run_dir = tmp_path / "run"
    prepare_run(protocol, repository_root=ROOT, run_dir=run_dir)

    with pytest.raises(LLMRelevanceJudgingError, match="labels_not_locked"):
        score_run(protocol, repository_root=ROOT, run_dir=run_dir)

    first = FakeLLM()
    for round_id in ("independent_1", "independent_2"):
        run_judge_round(
            protocol,
            repository_root=ROOT,
            run_dir=run_dir,
            round_id=round_id,
            client=first,
            runtime_binding=_binding(),
        )
    run_adjudication(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        client=first,
        runtime_binding=_binding(),
    )
    path = run_dir / "rounds/independent_1/batch_00000.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["locked_response"]["labels"][0]["label"] = "relevant"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(LLMRelevanceJudgingError):
        verify_run(protocol, repository_root=ROOT, run_dir=run_dir)


@pytest.mark.llm_relevance_judging_regression
def test_verify_after_prepare_reports_incomplete_not_integrity_failure(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    run_dir = tmp_path / "run"
    prepare_run(protocol, repository_root=ROOT, run_dir=run_dir)

    report = verify_run(protocol, repository_root=ROOT, run_dir=run_dir)

    assert report["exit_code"] == EXIT_INCOMPLETE
    assert report["details"]["reason"] == "runtime_binding_missing"


@pytest.mark.parametrize(
    "rows,error",
    [
        ([], "llm_response_batch_mismatch"),
        (
            [
                {
                    "item_id": "item:" + "1" * 64,
                    "label": "relevant",
                    "evidence": "Visible evidence.",
                },
                {
                    "item_id": "item:" + "1" * 64,
                    "label": "not_relevant",
                    "evidence": "Visible evidence.",
                },
            ],
            "llm_response_duplicate_item",
        ),
    ],
)
def test_response_batch_omission_and_duplicate_are_rejected(
    rows: list[dict[str, str]], error: str
) -> None:
    from scholar_agent.evaluation.llm_relevance_judging import _validate_response

    with pytest.raises(LLMRelevanceJudgingError, match=error):
        _validate_response(
            {"labels": rows},
            expected_ids=["item:" + "1" * 64],
            mode="judge",
        )
