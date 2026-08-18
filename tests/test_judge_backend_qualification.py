from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import check_judge_backend_qualification as qualification_cli
from scholar_agent.evaluation.judge_backend_qualification import (
    EXIT_NOT_ELIGIBLE,
    EXIT_QUALIFIED,
    EXIT_VIOLATION,
    QualificationError,
    analyze_frozen_evidence,
    candidate_from_runtime,
    load_protocol,
    qualify_run,
    run_probe,
    verify_published,
    write_frozen_analysis,
)
from scholar_agent.llm.provider import (
    LLMCallDiagnostics,
    LLMErrorDetails,
    LLMProviderError,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark" / "judge_backend_qualification_v1_protocol.json"


class FakeProvider:
    def __init__(self, fault: str | None = None) -> None:
        self.fault = fault
        self.calls = 0
        self.last_call_diagnostics: LLMCallDiagnostics | None = None
        self.last_call_usage_fields: dict[str, int] | None = None

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        assert temperature == 0
        assert timeout == 60.0
        self.calls += 1
        payload = _payload(messages)
        item = payload["item"]
        self.last_call_diagnostics = LLMCallDiagnostics(
            mode="structured_json",
            http_attempts=1,
            latency_ms=1,
        )
        self.last_call_usage_fields = {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }
        if self.fault in {"429", "503"}:
            status = int(self.fault)
            raise LLMProviderError(
                "safe_provider_failure",
                details=LLMErrorDetails(http_status=status),
            )
        if self.fault == "missing_usage":
            self.last_call_usage_fields = None
        if self.fault == "retry":
            self.last_call_diagnostics = LLMCallDiagnostics(
                mode="structured_json",
                http_attempts=2,
                latency_ms=1,
            )
        if self.fault == "fallback":
            self.last_call_diagnostics = LLMCallDiagnostics(
                mode="json_only_prompt",
                http_attempts=2,
                latency_ms=1,
                fallback_reason="unsupported_parameters:response_format",
            )
        label = {
            "item_id": item["item_id"],
            "label": item["required_label"],
            "evidence": item["expected_evidence"],
        }
        if self.fault == "item_mismatch":
            label["item_id"] = "canary:" + "0" * 64
        if self.fault == "illegal_enum":
            label["label"] = "maybe"
        response: dict[str, Any] = {"labels": [label]}
        if self.fault == "extra_text":
            response["commentary"] = "not allowed"
        if self.fault == "missing_key":
            del label["evidence"]
        if self.fault == "truncated_batch":
            response["labels"] = []
        if self.fault == "duplicate_item":
            response["labels"] = [label, dict(label)]
        return response


def _protocol() -> dict[str, Any]:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


def _candidate(model: str = "fake-model") -> Any:
    return candidate_from_runtime(
        provider="openai_compatible",
        model=model,
        available=True,
        reason=None,
        request_options={"max_tokens": 1024, "timeout_seconds": 60.0},
    )


def _payload(messages: list[dict[str, str]]) -> dict[str, Any]:
    marker = '{"item":'
    content = messages[1]["content"]
    return json.loads(content[content.index(marker) :])


def _prepare(run_dir: Path) -> dict[str, Any]:
    protocol = _protocol()
    write_frozen_analysis(
        run_dir,
        analyze_frozen_evidence(protocol, repository_root=ROOT),
    )
    return protocol


def _calls_path(run_dir: Path, candidate: Any) -> Path:
    return run_dir / "calls" / f"{candidate.candidate_id.removeprefix('candidate:')}.jsonl"


def test_frozen_analysis_is_descriptive_complete_and_deterministic() -> None:
    protocol = _protocol()

    first = analyze_frozen_evidence(protocol, repository_root=ROOT)
    second = analyze_frozen_evidence(protocol, repository_root=ROOT)

    assert first == second
    assert first["response_content_accessed"] is False
    assert first["labels_generated_or_read"] is False
    assert [item["attempt_count"] for item in first["evidence"]] == [65, 689]
    assert first["evidence"][1]["http_statuses"] == {"429": 155, "503": 8}
    assert first["causal_interpretation"] == "descriptive_association_only"


def test_all_24_strict_canaries_qualify_without_persisting_labels(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    publish_dir = tmp_path / "published"
    protocol = _prepare(run_dir)
    candidate = _candidate()
    fake = FakeProvider()

    probe = run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        candidates=[candidate],
        client_factory=lambda _candidate: fake,
    )
    result = qualify_run(
        protocol,
        repository_root=ROOT,
        run_dir=run_dir,
        publish_dir=publish_dir,
    )

    assert fake.calls == 24
    assert probe["status"] == "qualified"
    assert probe["exit_code"] == EXIT_QUALIFIED
    assert result["qualified_candidate_count"] == 1
    assert result["followup_run_recommendation"]["full_run_started"] is False
    assert result["relevance_labels_generated"] is False
    assert result["quality_statistics_generated"] is False
    persisted = _calls_path(run_dir, candidate).read_text(encoding="utf-8")
    for forbidden in ('"label"', '"required_label"', '"response"'):
        assert forbidden not in persisted
    assert verify_published(protocol, publish_dir=publish_dir)["status"] == "qualified"


@pytest.mark.parametrize(
    ("fault", "expected_code"),
    [
        ("extra_text", "response_schema_invalid"),
        ("missing_key", "response_schema_invalid"),
        ("illegal_enum", "response_schema_invalid"),
        ("truncated_batch", "response_item_count_mismatch"),
        ("duplicate_item", "response_item_count_mismatch"),
        ("item_mismatch", "response_item_binding_mismatch"),
        ("missing_usage", "supplier_usage_missing"),
        ("retry", "native_mode_or_rate_violation"),
        ("fallback", "native_mode_or_rate_violation"),
    ],
)
def test_schema_binding_usage_and_transport_drift_never_auto_repair(
    tmp_path: Path,
    fault: str,
    expected_code: str,
) -> None:
    protocol = _prepare(tmp_path)
    candidate = _candidate(f"fake-{fault}")
    fake = FakeProvider(fault)

    report = run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=tmp_path,
        candidates=[candidate],
        client_factory=lambda _candidate: fake,
    )

    assert fake.calls == 24
    assert report["status"] == "conformance_violation"
    assert report["exit_code"] == EXIT_VIOLATION
    matrix = report["candidates"][0]
    assert matrix["qualified"] is False
    assert matrix["failure_codes"][expected_code] == 24


@pytest.mark.parametrize("fault", ["429", "503"])
def test_provider_failure_is_counted_and_not_selectively_retried(
    tmp_path: Path,
    fault: str,
) -> None:
    protocol = _prepare(tmp_path)
    candidate = _candidate(f"fake-http-{fault}")
    fake = FakeProvider(fault)

    report = run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=tmp_path,
        candidates=[candidate],
        client_factory=lambda _candidate: fake,
    )

    assert fake.calls == 24
    assert report["candidates"][0]["counts"]["provider_failure_count"] == 24
    assert report["candidates"][0]["counts"]["unexpected_http_retry_count"] == 0
    assert report["candidates"][0]["transport_diagnostics"] == {
        "http_attempts_unavailable_count": 0
    }
    assert report["candidates"][0]["failure_codes"] == {"provider_failure": 24}
    assert report["exit_code"] == EXIT_VIOLATION


def test_resume_reuses_all_locked_calls_and_never_appends_attempts(
    tmp_path: Path,
) -> None:
    protocol = _prepare(tmp_path)
    candidate = _candidate()
    first = FakeProvider()
    run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=tmp_path,
        candidates=[candidate],
        client_factory=lambda _candidate: first,
    )
    before = _calls_path(tmp_path, candidate).read_bytes()
    client_factory_calls = 0

    def forbidden_factory(_candidate: Any) -> FakeProvider:
        nonlocal client_factory_calls
        client_factory_calls += 1
        raise AssertionError("a complete resume must not instantiate a client")

    report = run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=tmp_path,
        candidates=[candidate],
        client_factory=forbidden_factory,
    )

    assert first.calls == 24
    assert client_factory_calls == 0
    assert report["status"] == "qualified"
    assert _calls_path(tmp_path, candidate).read_bytes() == before


def test_duplicate_attempt_and_cross_candidate_mixing_are_rejected(
    tmp_path: Path,
) -> None:
    protocol = _prepare(tmp_path)
    candidate = _candidate()
    run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=tmp_path,
        candidates=[candidate],
        client_factory=lambda _candidate: FakeProvider(),
    )
    path = _calls_path(tmp_path, candidate)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in [*rows, rows[0]]) + "\n"
    )
    with pytest.raises(QualificationError, match="duplicate_canary_attempt"):
        run_probe(
            protocol,
            repository_root=ROOT,
            run_dir=tmp_path,
            candidates=[candidate],
            client_factory=lambda _candidate: FakeProvider(),
        )

    path.write_text(
        "\n".join(
            json.dumps(
                ({**row, "candidate_id": "candidate:" + "f" * 64} if index == 0 else row),
                sort_keys=True,
            )
            for index, row in enumerate(rows)
        )
        + "\n"
    )
    with pytest.raises(QualificationError, match="cross_candidate_call_mixing"):
        run_probe(
            protocol,
            repository_root=ROOT,
            run_dir=tmp_path,
            candidates=[candidate],
            client_factory=lambda _candidate: FakeProvider(),
        )


def test_candidate_limit_request_mismatch_and_unavailable_are_bounded(
    tmp_path: Path,
) -> None:
    protocol = _prepare(tmp_path)
    unavailable = candidate_from_runtime(
        provider="disabled",
        model=None,
        available=False,
        reason="provider_disabled",
        request_options={"max_tokens": 1024, "timeout_seconds": 60.0},
    )
    report = run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=tmp_path,
        candidates=[unavailable],
        client_factory=lambda _candidate: pytest.fail("client must not be created"),
    )
    assert report["exit_code"] == EXIT_NOT_ELIGIBLE
    assert report["candidates"][0]["logical_call_count"] == 0

    mismatched = candidate_from_runtime(
        provider="openai_compatible",
        model="other",
        available=True,
        reason=None,
        request_options={"max_tokens": 1024, "timeout_seconds": 30.0},
    )
    second = tmp_path / "second"
    _prepare(second)
    report = run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=second,
        candidates=[mismatched],
        client_factory=lambda _candidate: pytest.fail("client must not be created"),
    )
    assert report["exit_code"] == EXIT_NOT_ELIGIBLE
    assert report["candidates"][0]["reason"] == "request_options_mismatch"


def test_published_report_tampering_is_detected(tmp_path: Path) -> None:
    protocol = _prepare(tmp_path / "run")
    candidate = _candidate()
    run_probe(
        protocol,
        repository_root=ROOT,
        run_dir=tmp_path / "run",
        candidates=[candidate],
        client_factory=lambda _candidate: FakeProvider(),
    )
    publish = tmp_path / "publish"
    qualify_run(
        protocol,
        repository_root=ROOT,
        run_dir=tmp_path / "run",
        publish_dir=publish,
    )
    report = json.loads((publish / "qualification.json").read_text())
    report["qualified_candidate_count"] = 0
    (publish / "qualification.json").write_text(json.dumps(report))

    with pytest.raises(QualificationError, match="published_file_(size|hash)_mismatch"):
        verify_published(protocol, publish_dir=publish)


def test_cli_analyze_probe_qualify_and_verify_with_fake_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    publish_dir = tmp_path / "publish"
    fake = FakeProvider()
    runtime = type(
        "Runtime",
        (),
        {
            "provider": "openai_compatible",
            "model": "fake-model",
            "available": True,
            "reason": None,
        },
    )()
    monkeypatch.setattr(qualification_cli, "load_project_env", lambda root: True)
    monkeypatch.setattr(
        qualification_cli, "get_llm_runtime_config", lambda: runtime
    )
    monkeypatch.setattr(
        qualification_cli,
        "get_llm_request_options",
        lambda: {"max_tokens": 1024, "timeout_seconds": 60.0},
    )
    monkeypatch.setattr(
        qualification_cli.OpenAICompatibleLLMClient,
        "from_env",
        lambda: fake,
    )
    common = [
        "--repository-root",
        str(ROOT),
        "--protocol",
        str(PROTOCOL_PATH),
        "--run-dir",
        str(run_dir),
        "--publish-dir",
        str(publish_dir),
    ]

    for command in ("analyze-frozen", "probe", "qualify", "verify"):
        assert qualification_cli.main([*common, command]) == EXIT_QUALIFIED
        report = json.loads(capsys.readouterr().out)
        assert report["exit_code"] == EXIT_QUALIFIED
    assert fake.calls == 24
