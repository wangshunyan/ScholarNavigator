from __future__ import annotations

import json
from pathlib import Path

from scholar_agent.evaluation.llm_feedback_snapshots import (
    LLMFeedbackRequest,
    LLMFeedbackSnapshotRuntime,
    LLMFeedbackSnapshotStore,
    llm_feedback_snapshot_key,
)


class Client:
    provider = "test_provider"
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.last_call_diagnostics = None

    def chat_json(self, messages, *, temperature: float, timeout: float):  # noqa: ANN001
        del messages, temperature, timeout
        self.calls += 1
        self.token_usage = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }
        return {"supplemental_queries": []}


def _request() -> LLMFeedbackRequest:
    return LLMFeedbackRequest(
        provider="test_provider",
        model="test-model",
        base_url_host="loopback",
        prompt_name="llm_feedback_evolution",
        prompt_version="1",
        prompt_hash="a" * 64,
        request_identity={
            "original_query_hash": "b" * 64,
            "constraints_hash": "c" * 64,
            "coverage_gap_hash": "d" * 64,
            "candidate_hashes": ["e" * 64],
            "candidate_count": 1,
            "max_supplemental_queries": 1,
        },
        temperature=0,
        max_tokens=128,
        max_supplemental_queries=1,
    )


def test_feedback_snapshot_key_is_stable_and_deidentified() -> None:
    request = _request()

    assert llm_feedback_snapshot_key(request) == llm_feedback_snapshot_key(request)
    assert "graph neural" not in json.dumps(request.model_dump(mode="json"))


def test_feedback_record_then_replay_is_offline(tmp_path: Path) -> None:
    request = _request()
    store = LLMFeedbackSnapshotStore(tmp_path)
    client = Client()
    record = LLMFeedbackSnapshotRuntime(store, mode="record")

    recorded = record.execute(request, [{"role": "user", "content": "irrelevant"}], client, timeout=1)
    replay = LLMFeedbackSnapshotRuntime(store, mode="replay")
    replayed = replay.execute(request, [{"role": "user", "content": "must not be sent"}], None, timeout=1)

    assert recorded.snapshot_status == "record"
    assert replayed.snapshot_status == "replay"
    assert replayed.replayed is True
    assert replayed.llm_call_attempted is False
    assert client.calls == 1
    assert replay.finish_case().replay_execution_request_count == 0


def test_feedback_replay_missing_fails_closed_without_a_client(tmp_path: Path) -> None:
    runtime = LLMFeedbackSnapshotRuntime(LLMFeedbackSnapshotStore(tmp_path), mode="replay")

    try:
        runtime.execute(_request(), [], None, timeout=1)
    except Exception as exc:  # The public error name is part of the fail-closed contract.
        assert "llm_feedback_snapshot_missing" in str(exc)
    else:  # pragma: no cover - protects the test assertion itself.
        raise AssertionError("missing feedback snapshot did not fail closed")
    assert runtime.failure_diagnostics()["llm_call_attempted"] is False


def test_feedback_snapshot_stores_hashes_not_request_text(tmp_path: Path) -> None:
    request = _request()
    runtime = LLMFeedbackSnapshotRuntime(LLMFeedbackSnapshotStore(tmp_path), mode="record")
    runtime.execute(request, [{"role": "user", "content": "private query"}], Client(), timeout=1)

    text = next((tmp_path / "llm_feedback").glob("*.json")).read_text(encoding="utf-8")
    assert "private query" not in text
    assert "graph neural" not in text
    assert "input_payload" not in text
