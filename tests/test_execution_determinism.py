from __future__ import annotations

import copy
import json
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.check_execution_determinism as determinism_cli  # noqa: E402
from scholar_agent.evaluation.execution_determinism import (
    EXIT_INVARIANT_VIOLATION,
    CanonicalizationRule,
    ExecutionDeterminismError,
    ExecutionEvent,
    ExecutionRecord,
    ExecutionRuntime,
    FixtureNotEligible,
    QueryFixture,
    build_checkpoint,
    canonicalize_execution_record,
    load_protocol,
    load_query_fixtures,
    merge_checkpoint_resume,
    run_execution_determinism,
    write_json,
)


PROTOCOL = ROOT / "benchmark" / "execution_determinism_v1_protocol.json"


def _record(
    query: QueryFixture,
    *,
    status: str = "succeeded",
    marker: Any = None,
    ranked: list[str] | None = None,
    elapsed: float = 1.0,
) -> ExecutionRecord:
    return ExecutionRecord(
        query_identity=query.identity,
        status=status,
        result=(
            {
                "deduplicated_count": 2,
                "ranked_papers": ranked or ["paper-a", "paper-b"],
                "stage_snapshots": [{"stage": "final", "status": "completed"}],
                "semantic_marker": marker,
                "latency_seconds": elapsed,
                "stage_latencies": {"retrieval": elapsed},
                "budget_status": {"elapsed_seconds": elapsed},
            }
            if status == "succeeded"
            else None
        ),
        events=[
            ExecutionEvent(
                event="completed" if status == "succeeded" else status,
                payload={"stage": "final", "latency_seconds": elapsed},
            )
        ],
        error_type=None if status == "succeeded" else status.title(),
        runtime=ExecutionRuntime(
            run_id=f"run-{elapsed}",
            process_id=999,
            invocation_index=max(1, int(elapsed)),
            elapsed_seconds=elapsed,
        ),
    )


class StableBackend:
    def __init__(self) -> None:
        self.calls = 0

    def execute(
        self,
        query: QueryFixture,
        *,
        execution_label: str,
        should_cancel: Callable[[], bool],
    ) -> ExecutionRecord:
        self.calls += 1
        if should_cancel():
            return _record(query, status="cancelled", elapsed=float(self.calls))
        return _record(query, marker=query.identity, elapsed=float(self.calls))


class OrderLeakingBackend(StableBackend):
    """Mutable call position leaks into semantics and makes order observable."""

    def execute(
        self,
        query: QueryFixture,
        *,
        execution_label: str,
        should_cancel: Callable[[], bool],
    ) -> ExecutionRecord:
        self.calls += 1
        if should_cancel():
            return _record(query, status="cancelled", elapsed=float(self.calls))
        return _record(query, marker=self.calls, elapsed=float(self.calls))


class DeterministicRaceBackend(StableBackend):
    def __init__(self, parties: int) -> None:
        super().__init__()
        self._barrier = threading.Barrier(parties)
        self._shared = 0

    def execute(
        self,
        query: QueryFixture,
        *,
        execution_label: str,
        should_cancel: Callable[[], bool],
    ) -> ExecutionRecord:
        if should_cancel():
            return _record(query, status="cancelled")
        if execution_label == "concurrent":
            observed = self._shared
            self._barrier.wait()
            self._shared = observed + 1
            return _record(query, marker=observed)
        return _record(query, marker=query.identity)


class UnstableTieBackend(StableBackend):
    def execute(
        self,
        query: QueryFixture,
        *,
        execution_label: str,
        should_cancel: Callable[[], bool],
    ) -> ExecutionRecord:
        if should_cancel():
            return _record(query, status="cancelled")
        ranked = ["paper-b", "paper-a"] if execution_label == "repeat_second" else None
        return _record(query, marker=query.identity, ranked=ranked)


class CancellationLeakingBackend(StableBackend):
    def __init__(self) -> None:
        super().__init__()
        self._cancelled = False

    def execute(
        self,
        query: QueryFixture,
        *,
        execution_label: str,
        should_cancel: Callable[[], bool],
    ) -> ExecutionRecord:
        self._cancelled = self._cancelled or should_cancel()
        if self._cancelled:
            return _record(query, status="cancelled")
        return _record(query, marker=query.identity)


def _protocol() -> dict[str, Any]:
    return load_protocol(PROTOCOL, repository_root=ROOT)


def _rules(protocol: dict[str, Any]) -> list[CanonicalizationRule]:
    return [
        CanonicalizationRule.model_validate(item)
        for item in protocol["canonicalization"]["excluded_fields"]
    ]


@pytest.mark.execution_determinism_regression
def test_repository_execution_determinism_gate_passes_offline() -> None:
    report = run_execution_determinism(_protocol(), repository_root=ROOT)
    assert report["status"] == "passed"
    assert report["violation_count"] == 0
    assert {row["invariant"] for row in report["invariants"]} == {
        "repeat_same_configuration",
        "single_vs_batch",
        "batch_reorder",
        "serial_vs_controlled_concurrent",
        "checkpoint_resume",
        "cancellation_isolation",
    }
    assert report["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "quality_metric_count": 0,
        "fault_injection": None,
    }


def test_gate_detects_mutable_order_state_and_unstable_tie_break(
    tmp_path: Path,
) -> None:
    order_report = run_execution_determinism(
        _protocol(),
        repository_root=ROOT,
        backend_factory=OrderLeakingBackend,
        snapshot_root=tmp_path,
    )
    tie_report = run_execution_determinism(
        _protocol(),
        repository_root=ROOT,
        backend_factory=UnstableTieBackend,
        snapshot_root=tmp_path,
    )
    assert any(
        item["invariant"] == "batch_reorder"
        and item["first_difference_path"].endswith("semantic_marker")
        for item in order_report["violations"]
    )
    assert any(
        item["invariant"] == "repeat_same_configuration"
        and "ranked_papers" in item["first_difference_path"]
        for item in tie_report["violations"]
    )


def test_gate_detects_controlled_concurrent_race_without_sleep(
    tmp_path: Path,
) -> None:
    report = run_execution_determinism(
        _protocol(),
        repository_root=ROOT,
        backend_factory=lambda: DeterministicRaceBackend(3),
        snapshot_root=tmp_path,
    )
    assert any(
        item["invariant"] == "serial_vs_controlled_concurrent"
        for item in report["violations"]
    )


def test_gate_detects_cancellation_token_leak_and_semantic_fault(
    tmp_path: Path,
) -> None:
    cancellation = run_execution_determinism(
        _protocol(),
        repository_root=ROOT,
        backend_factory=CancellationLeakingBackend,
        snapshot_root=tmp_path,
    )
    fault = run_execution_determinism(
        _protocol(),
        repository_root=ROOT,
        backend_factory=StableBackend,
        fault="semantic_result_change",
        snapshot_root=tmp_path,
    )
    assert any(
        item["invariant"] == "cancellation_isolation"
        for item in cancellation["violations"]
    )
    assert fault["exit_code"] == EXIT_INVARIANT_VIOLATION
    assert fault["violations"][0]["first_difference_path"].endswith(
        "deduplicated_count"
    )


def test_only_registered_transient_fields_are_excluded() -> None:
    protocol = _protocol()
    rules = _rules(protocol)
    query = load_query_fixtures(protocol, repository_root=ROOT)[0][0]
    left = _record(query, marker="same", elapsed=1.0)
    right = _record(query, marker="same", elapsed=9.0)
    left_value, left_counts = canonicalize_execution_record(left, rules)
    right_value, right_counts = canonicalize_execution_record(right, rules)
    assert left_value == right_value
    assert left_counts == right_counts
    assert left_counts["runtime.elapsed_seconds"] == 1
    right.result["new_semantic_field"] = "changed"
    changed, _ = canonicalize_execution_record(right, rules)
    assert changed != left_value


def test_checkpoint_resume_rejects_duplicates_and_omissions() -> None:
    protocol = _protocol()
    queries = load_query_fixtures(protocol, repository_root=ROOT)[0]
    records = [_record(query, marker=query.identity) for query in queries]
    checkpoint = build_checkpoint(queries, records[:1], config_sha256="a" * 64)
    merged = merge_checkpoint_resume(
        checkpoint, records[1:], config_sha256="a" * 64
    )
    assert [item.query_identity for item in merged] == [
        item.identity for item in queries
    ]
    with pytest.raises(ExecutionDeterminismError, match="duplicate"):
        merge_checkpoint_resume(
            checkpoint, [records[0], *records[1:]], config_sha256="a" * 64
        )
    with pytest.raises(ExecutionDeterminismError, match="omission"):
        merge_checkpoint_resume(
            checkpoint, records[2:], config_sha256="a" * 64
        )


def test_insufficient_fixture_is_not_eligible() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["fixture"]["query_selection"]["count"] = 100
    with pytest.raises(FixtureNotEligible, match="insufficient_fixture"):
        load_query_fixtures(protocol, repository_root=ROOT)


def test_gate_output_is_byte_deterministic(tmp_path: Path) -> None:
    first = run_execution_determinism(
        _protocol(),
        repository_root=ROOT,
        backend_factory=StableBackend,
        snapshot_root=tmp_path / "snapshots",
    )
    second = run_execution_determinism(
        _protocol(),
        repository_root=ROOT,
        backend_factory=StableBackend,
        snapshot_root=tmp_path / "snapshots",
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(first_path, first)
    write_json(second_path, second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text(encoding="utf-8"))["status"] == "passed"


def test_cli_uses_contract_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert determinism_cli.main([]) == 4

    monkeypatch.setattr(
        determinism_cli,
        "run_execution_determinism",
        lambda *_args, **_kwargs: {
            "schema_version": "1",
            "contract": "execution_determinism_v1",
            "gate": "execution_determinism_gate",
            "status": "invariant_violation",
            "exit_code": 2,
        },
    )
    assert (
        determinism_cli.main(
            [
                "--repository-root",
                str(ROOT),
                "--protocol",
                str(PROTOCOL),
                "check",
            ]
        )
        == 2
    )
    assert (
        determinism_cli.main(
            [
                "--repository-root",
                str(ROOT),
                "--protocol",
                str(PROTOCOL),
                "audit-frozen",
            ]
        )
        == 3
    )
