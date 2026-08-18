"""Two-stage, fail-closed launch control for the frozen Full1000 plan.

The module never loads credentials or starts retrieval.  It seals the already
reviewed execution plan into deterministic preparation/authorization records
and validates an append-only operational audit chain.  A run without these
bindings may still be an ordinary benchmark run, but it is not an authoritative
Full1000 run.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash
from scholar_agent.evaluation.validation_evidence_freshness import (
    load_contract as load_freshness_contract,
    verify_current as verify_freshness,
)


PROTOCOL = "full1000_launch_control_v1"
SCHEMA_VERSION = "1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

EXECUTION = {
    "gold_or_qrels_loaded": False,
    "llm_request_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
    "snapshot_write_count": 0,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ATTEMPT = re.compile(r"^shard-(?P<shard>[0-9]{2})-attempt-(?P<attempt>[01])$")
_EVENTS = frozenset(
    {
        "authorized",
        "started",
        "paused",
        "resumed",
        "shard_failed",
        "attempt_superseded",
        "shard_completed",
        "aggregate_requested",
        "cancelled",
        "revoked",
    }
)
_PROHIBITED_TEXT = (
    ".env",
    "api_key",
    "authorization:",
    "bearer ",
    "http://",
    "https://",
)


class LaunchControlError(RuntimeError):
    """Authorization, state, or audit invariants were violated."""


class LaunchControlNotReady(LaunchControlError):
    """A required frozen input is unavailable."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchControlNotReady("required_json_unavailable") from exc
    if not isinstance(value, dict):
        raise LaunchControlError("json_root_not_object")
    return value


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise LaunchControlError("unsafe_relative_path")
    if path.parts[0] == "third_party" or path.name == ".env":
        raise LaunchControlError("prohibited_path")
    return path.as_posix()


def _contract_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("protocol_sha256", None)
    return stable_hash(payload)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_object(path)
    required = {
        "activation",
        "audit",
        "authorization",
        "bindings",
        "execution_contract",
        "launch_command",
        "output",
        "population",
        "protocol",
        "protocol_sha256",
        "schema_version",
        "source_commit",
        "state_machine",
    }
    if set(protocol) != required:
        raise LaunchControlError("protocol_schema_invalid")
    if protocol["protocol"] != PROTOCOL or protocol["schema_version"] != SCHEMA_VERSION:
        raise LaunchControlError("protocol_version_invalid")
    if not _COMMIT.fullmatch(str(protocol["source_commit"])):
        raise LaunchControlError("protocol_source_commit_invalid")
    if _contract_digest(protocol) != protocol["protocol_sha256"]:
        raise LaunchControlError("protocol_digest_invalid")
    if protocol["activation"] != {
        "credential_status": "not_checked",
        "network_status": "not_checked",
        "real_launch_allowed_by_this_protocol": False,
    }:
        raise LaunchControlError("activation_contract_drift")
    population = protocol["population"]
    if (
        population.get("query_count") != 1000
        or population.get("shard_count") != 20
        or protocol["authorization"].get("allowed_shards") != list(range(20))
    ):
        raise LaunchControlError("population_or_shard_contract_drift")
    if set(protocol["audit"].get("allowed_events") or []) != _EVENTS:
        raise LaunchControlError("audit_event_contract_drift")
    for binding in protocol["bindings"].values():
        _safe_relative(str(binding["path"]))
    for key in ("authoritative_run_root", "aggregate_directory", "snapshot_directory"):
        _safe_relative(str(protocol["output"][key]))
    return protocol


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if completed.returncode != 0:
        raise LaunchControlNotReady("git_identity_unavailable")
    return completed.stdout.strip()


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    return completed.returncode == 0


def _validate_bound_inputs(root: Path, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    bindings = protocol["bindings"]
    observed: dict[str, str] = {}
    values: dict[str, dict[str, Any]] = {}
    for name, binding in bindings.items():
        path = root / _safe_relative(str(binding["path"]))
        if not path.is_file():
            raise LaunchControlNotReady(f"bound_input_missing:{name}")
        if "sha256" in binding:
            digest = sha256_file(path)
            observed[name] = digest
            if digest != binding["sha256"]:
                raise LaunchControlError(f"bound_input_hash_drift:{name}")
        values[name] = _read_object(path)
    plan = values["execution_plan"]
    addendum = values["capture_addendum"]
    if plan.get("contract") != "full1000_execution_plan_v1":
        raise LaunchControlError("execution_plan_contract_drift")
    if plan.get("plan_sha256") != bindings["execution_plan"]["embedded_plan_sha256"]:
        raise LaunchControlError("execution_plan_embedded_digest_drift")
    if addendum.get("execution_plan", {}).get("embedded_plan_sha256") != plan.get("plan_sha256"):
        raise LaunchControlError("capture_addendum_plan_binding_drift")
    if addendum.get("capture_contract", {}).get("protocol") != "provider_ingest_provenance_v1":
        raise LaunchControlError("capture_protocol_binding_drift")
    population = plan.get("population") or {}
    expected_population = protocol["population"]
    checks = {
        "query_count": population.get("count"),
        "query_stable_identity_sha256": population.get("stable_identity_sha256"),
        "query_order_sha256": population.get("order_sha256"),
        "shard_assignment_sha256": plan.get("sharding", {}).get("assignment_sha256"),
        "shard_count": plan.get("sharding", {}).get("shard_count"),
    }
    if checks != expected_population:
        raise LaunchControlError("plan_population_binding_drift")
    if stable_hash(plan.get("execution_contract")) != protocol["execution_contract"]["configuration_sha256"]:
        raise LaunchControlError("execution_configuration_drift")
    features = plan["execution_contract"]["experimental_features"]
    if (
        plan["execution_contract"].get("query_planning_policy") != "current_rules"
        or plan["execution_contract"].get("judgement_policy") != "current_rules"
        or plan["execution_contract"].get("ranking_policy") != "current_rules"
        or any(features.values())
        or features.get("deterministic_tiebreak_v2") is not False
    ):
        raise LaunchControlError("default_policy_drift")
    if (
        plan.get("resume", {}).get("start_mode") != "full_restart_all_1000"
        or plan.get("legacy_artifacts", {}).get("reuse_as_completed") is not False
    ):
        raise LaunchControlError("legacy_checkpoint_reuse_forbidden")
    return plan, {"hashes": observed, "values": values}


def _validate_freshness(root: Path, protocol: Mapping[str, Any]) -> None:
    binding = protocol["bindings"]["freshness"]
    contract = load_freshness_contract(root / binding["path"], repository_root=root)
    report = verify_freshness(contract, repository_root=root)
    if report.get("status") != binding["expected_status"] or report.get("exit_code") != 0:
        raise LaunchControlError("freshness_not_closed")


def _run_root_is_empty(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and next(path.iterdir(), None) is None)


def build_preparation(
    root: Path,
    protocol: Mapping[str, Any],
    *,
    authoritative_root: Path | None = None,
    check_freshness: bool = True,
) -> dict[str, Any]:
    plan, bound = _validate_bound_inputs(root, protocol)
    if check_freshness:
        _validate_freshness(root, protocol)
    output_relative = _safe_relative(str(protocol["output"]["authoritative_run_root"]))
    output_path = authoritative_root or (root / output_relative)
    if not _run_root_is_empty(output_path):
        raise LaunchControlError("authoritative_output_root_not_empty")
    observed_head = _git_head(root)
    if not _git_is_ancestor(root, str(protocol["source_commit"]), observed_head):
        raise LaunchControlError("source_commit_not_ancestor")
    shards = plan["sharding"]["shards"]
    allowed_attempts = {
        str(shard["shard_index"]): [
            item["attempt_id"] for item in shard["attempts"]
        ]
        for shard in shards
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "state": "prepared",
        "source_commit": protocol["source_commit"],
        "observed_head": observed_head,
        "protocol_sha256": protocol["protocol_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": bound["hashes"]["execution_plan"],
        "capture_addendum_sha256": bound["hashes"]["capture_addendum"],
        "provider_ingest_protocol_sha256": bound["hashes"]["provider_ingest_protocol"],
        "query_count": 1000,
        "query_order_sha256": protocol["population"]["query_order_sha256"],
        "configuration_sha256": protocol["execution_contract"]["configuration_sha256"],
        "shard_count": 20,
        "allowed_attempts": allowed_attempts,
        "authoritative_output_root": output_relative,
        "output_root_empty": True,
        "launch_command_template_sha256": stable_hash(protocol["launch_command"]["argv_template"]),
        "required_observability": protocol["execution_contract"]["required_observability"],
        "legacy_checkpoint_reuse": False,
        "credential_status": "not_checked",
        "network_status": "not_checked",
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }
    payload["preparation_sha256"] = stable_hash(payload)
    return payload


def validate_preparation(prepared: Mapping[str, Any]) -> None:
    if prepared.get("state") != "prepared":
        raise LaunchControlError("preparation_state_invalid")
    payload = dict(prepared)
    digest = payload.pop("preparation_sha256", None)
    if digest != stable_hash(payload):
        raise LaunchControlError("preparation_digest_invalid")
    if prepared.get("query_count") != 1000 or prepared.get("output_root_empty") is not True:
        raise LaunchControlError("preparation_scope_invalid")
    if prepared.get("legacy_checkpoint_reuse") is not False:
        raise LaunchControlError("legacy_checkpoint_reuse_forbidden")


def build_authorization(prepared: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    validate_preparation(prepared)
    if (
        prepared.get("protocol_sha256") != protocol["protocol_sha256"]
        or prepared.get("source_commit") != protocol["source_commit"]
        or prepared.get("configuration_sha256")
        != protocol["execution_contract"]["configuration_sha256"]
        or prepared.get("query_order_sha256")
        != protocol["population"]["query_order_sha256"]
    ):
        raise LaunchControlError("preparation_protocol_binding_drift")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "state": "authorized",
        "preparation_sha256": prepared["preparation_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "source_commit": prepared["source_commit"],
        "observed_head": prepared["observed_head"],
        "plan_sha256": prepared["plan_sha256"],
        "configuration_sha256": prepared["configuration_sha256"],
        "query_order_sha256": prepared["query_order_sha256"],
        "authoritative_output_root": prepared["authoritative_output_root"],
        "allowed_attempts": prepared["allowed_attempts"],
        "allowed_shards": list(range(20)),
        "launch_argv_template": protocol["launch_command"]["argv_template"],
        "private_key_used": False,
        "activation": "external_activation_blocked",
        "credential_status": "not_checked",
        "network_status": "not_checked",
        "formal_validation_complete": False,
    }
    payload["authorization_sha256"] = stable_hash(payload)
    return payload


def validate_authorization(
    prepared: Mapping[str, Any],
    authorization: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    validate_preparation(prepared)
    if authorization.get("state") != "authorized":
        raise LaunchControlError("authorization_state_invalid")
    payload = dict(authorization)
    digest = payload.pop("authorization_sha256", None)
    if digest != stable_hash(payload):
        raise LaunchControlError("authorization_digest_invalid")
    expected = build_authorization(prepared, protocol)
    if authorization != expected:
        raise LaunchControlError("authorization_binding_drift")


def validate_authorization_context(
    root: Path,
    prepared: Mapping[str, Any],
    authorization: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Revalidate the immutable repository and output context before launch."""

    validate_authorization(prepared, authorization, protocol)
    observed_head = _git_head(root)
    if prepared.get("observed_head") != observed_head:
        raise LaunchControlError("authorization_commit_drift")
    if not _git_is_ancestor(root, str(protocol["source_commit"]), observed_head):
        raise LaunchControlError("source_commit_not_ancestor")
    plan, _ = _validate_bound_inputs(root, protocol)
    _validate_freshness(root, protocol)
    if prepared.get("plan_sha256") != plan.get("plan_sha256"):
        raise LaunchControlError("authorization_plan_drift")
    output_path = root / _safe_relative(
        str(protocol["output"]["authoritative_run_root"])
    )
    if not _run_root_is_empty(output_path):
        raise LaunchControlError("authoritative_output_root_not_empty")


class AuditEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    event: str
    state_before: Literal["prepared", "authorized", "started", "revoked", "invalid"]
    state_after: Literal["prepared", "authorized", "started", "revoked", "invalid"]
    shard_index: int | None = Field(default=None, ge=0, le=19)
    attempt_id: str | None = None
    supersedes_attempt_id: str | None = None
    previous_entry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    details_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_digest(self) -> "AuditEntry":
        if self.event not in _EVENTS:
            raise ValueError("audit event not allowed")
        payload = self.model_dump(mode="json", exclude={"entry_sha256"})
        if stable_hash(payload) != self.entry_sha256:
            raise ValueError("audit entry digest mismatch")
        return self


class OperationAuditLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    protocol: Literal["full1000_launch_control_v1"] = PROTOCOL
    preparation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: list[AuditEntry]
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_chain(self) -> "OperationAuditLog":
        previous = None
        for index, entry in enumerate(self.entries):
            if entry.index != index or entry.previous_entry_sha256 != previous:
                raise ValueError("audit chain discontinuity")
            previous = entry.entry_sha256
        payload = self.model_dump(mode="json", exclude={"audit_sha256"})
        if stable_hash(payload) != self.audit_sha256:
            raise ValueError("audit summary digest mismatch")
        return self


class LaunchOperationMachine:
    """Deterministic state machine used by the launcher and offline fixture."""

    def __init__(self, prepared: Mapping[str, Any], authorization: Mapping[str, Any]) -> None:
        self.prepared = dict(prepared)
        self.authorization = dict(authorization)
        self.state = "prepared"
        self.entries: list[dict[str, Any]] = []
        self.paused = False
        self.cancelled = False
        self.failed_attempts: set[tuple[int, str]] = set()
        self.selected_attempts = {index: f"shard-{index:02d}-attempt-0" for index in range(20)}
        self.completed_shards: set[int] = set()
        self.aggregate_requested = False

    def _append(
        self,
        event: str,
        *,
        state_after: str | None = None,
        shard_index: int | None = None,
        attempt_id: str | None = None,
        supersedes_attempt_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        before = self.state
        after = state_after or before
        payload = {
            "index": len(self.entries),
            "event": event,
            "state_before": before,
            "state_after": after,
            "shard_index": shard_index,
            "attempt_id": attempt_id,
            "supersedes_attempt_id": supersedes_attempt_id,
            "previous_entry_sha256": (
                self.entries[-1]["entry_sha256"] if self.entries else None
            ),
            "details_sha256": stable_hash(dict(details or {})),
        }
        payload["entry_sha256"] = stable_hash(payload)
        self.entries.append(payload)
        self.state = after

    def authorize(self) -> None:
        if self.state != "prepared":
            raise LaunchControlError("authorization_transition_invalid")
        self._append("authorized", state_after="authorized")

    def start(self) -> None:
        if self.state != "authorized":
            raise LaunchControlError("start_requires_authorization")
        self._append("started", state_after="started")

    def pause(self) -> None:
        if self.state != "started" or self.paused:
            raise LaunchControlError("pause_transition_invalid")
        self.paused = True
        self._append("paused")

    def resume(self) -> None:
        if self.state != "started" or not self.paused or self.cancelled:
            raise LaunchControlError("resume_not_authorized")
        self.paused = False
        self._append("resumed")

    def fail_shard(self, shard_index: int) -> None:
        self._active_shard(shard_index)
        attempt = self.selected_attempts[shard_index]
        self.failed_attempts.add((shard_index, attempt))
        self._append("shard_failed", shard_index=shard_index, attempt_id=attempt)

    def supersede(self, shard_index: int) -> None:
        self._active_shard(shard_index)
        old = self.selected_attempts[shard_index]
        if (shard_index, old) not in self.failed_attempts or not old.endswith("attempt-0"):
            raise LaunchControlError("attempt_supersession_not_authorized")
        new = f"shard-{shard_index:02d}-attempt-1"
        self.selected_attempts[shard_index] = new
        self._append(
            "attempt_superseded",
            shard_index=shard_index,
            attempt_id=new,
            supersedes_attempt_id=old,
        )

    def complete_shard(self, shard_index: int, attempt_id: str | None = None) -> None:
        self._active_shard(shard_index)
        selected = self.selected_attempts[shard_index]
        if attempt_id is not None and attempt_id != selected:
            raise LaunchControlError("stale_attempt_completion")
        if shard_index in self.completed_shards:
            raise LaunchControlError("duplicate_shard_completion")
        if (shard_index, selected) in self.failed_attempts:
            raise LaunchControlError("failed_attempt_cannot_complete")
        self.completed_shards.add(shard_index)
        self._append("shard_completed", shard_index=shard_index, attempt_id=selected)

    def aggregate(self) -> None:
        if self.state != "started" or self.completed_shards != set(range(20)):
            raise LaunchControlError("aggregate_before_all_shards_complete")
        if self.aggregate_requested:
            raise LaunchControlError("duplicate_aggregate_request")
        self.aggregate_requested = True
        self._append(
            "aggregate_requested",
            details={"selected_attempts_sha256": stable_hash(self.selected_attempts)},
        )

    def cancel(self) -> None:
        if self.state != "started" or self.cancelled:
            raise LaunchControlError("cancel_transition_invalid")
        self.cancelled = True
        self._append("cancelled")

    def revoke(self) -> None:
        if self.state not in {"prepared", "authorized", "started"}:
            raise LaunchControlError("revoke_transition_invalid")
        self._append("revoked", state_after="revoked")

    def _active_shard(self, shard_index: int) -> None:
        if (
            self.state != "started"
            or self.paused
            or self.cancelled
            or shard_index not in range(20)
        ):
            raise LaunchControlError("shard_operation_not_authorized")

    def audit_log(self) -> OperationAuditLog:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "preparation_sha256": self.prepared["preparation_sha256"],
            "authorization_sha256": self.authorization["authorization_sha256"],
            "entries": self.entries,
        }
        payload["audit_sha256"] = stable_hash(payload)
        return OperationAuditLog.model_validate(payload)


def _fixture_protocol(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(protocol))
    value["output"]["authoritative_run_root"] = "synthetic/authoritative"
    value["output"]["aggregate_directory"] = "synthetic/authoritative/aggregate"
    value["output"]["snapshot_directory"] = "synthetic/snapshots"
    value["protocol_sha256"] = _contract_digest(value)
    return value


def simulate_operations(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="full1000-launch-control-") as temporary:
        temp = Path(temporary)
        fixture = _fixture_protocol(root, protocol)
        prepared = build_preparation(
            root,
            fixture,
            authoritative_root=temp / "authoritative",
            check_freshness=False,
        )
        authorization = build_authorization(prepared, fixture)
        validate_authorization(prepared, authorization, fixture)
        machine = LaunchOperationMachine(prepared, authorization)
        machine.authorize()
        machine.start()
        machine.pause()
        machine.resume()
        machine.fail_shard(7)
        machine.supersede(7)
        for shard in range(20):
            machine.complete_shard(shard)
        machine.aggregate()
        audit = machine.audit_log()
        validate_launch_evidence(
            prepared,
            authorization,
            audit.model_dump(mode="json"),
            fixture,
        )

        scenarios: dict[str, bool] = {}

        tampered = copy.deepcopy(authorization)
        tampered["plan_sha256"] = "0" * 64
        try:
            validate_authorization(prepared, tampered, fixture)
        except LaunchControlError:
            scenarios["authorization_tamper"] = True

        non_empty = temp / "non-empty"
        non_empty.mkdir()
        (non_empty / "legacy-checkpoint.json").write_text("{}", encoding="utf-8")
        try:
            build_preparation(
                root,
                fixture,
                authoritative_root=non_empty,
                check_freshness=False,
            )
        except LaunchControlError:
            scenarios["non_empty_or_legacy_output"] = True

        duplicate = LaunchOperationMachine(prepared, authorization)
        duplicate.authorize()
        duplicate.start()
        try:
            duplicate.start()
        except LaunchControlError:
            scenarios["duplicate_start"] = True

        revoked = LaunchOperationMachine(prepared, authorization)
        revoked.authorize()
        revoked.revoke()
        try:
            revoked.start()
        except LaunchControlError:
            scenarios["revoked_start"] = True

        resume = LaunchOperationMachine(prepared, authorization)
        resume.authorize()
        resume.start()
        try:
            resume.resume()
        except LaunchControlError:
            scenarios["unauthorized_resume"] = True

        early = LaunchOperationMachine(prepared, authorization)
        early.authorize()
        early.start()
        early.complete_shard(0)
        try:
            early.aggregate()
        except LaunchControlError:
            scenarios["partial_aggregate"] = True

        stale = LaunchOperationMachine(prepared, authorization)
        stale.authorize()
        stale.start()
        stale.fail_shard(1)
        stale.supersede(1)
        try:
            stale.complete_shard(1, "shard-01-attempt-0")
        except LaunchControlError:
            scenarios["stale_attempt"] = True

        broken = audit.model_dump(mode="json")
        broken["entries"][1]["previous_entry_sha256"] = "0" * 64
        try:
            OperationAuditLog.model_validate(broken)
        except ValidationError:
            scenarios["audit_chain_break"] = True

        plan_drift = copy.deepcopy(fixture)
        plan_drift["bindings"]["execution_plan"]["sha256"] = "0" * 64
        plan_drift["protocol_sha256"] = _contract_digest(plan_drift)
        try:
            build_preparation(
                root,
                plan_drift,
                authoritative_root=temp / "drift",
                check_freshness=False,
            )
        except LaunchControlError:
            scenarios["plan_or_commit_drift"] = True

        try:
            validate_launch_evidence(
                prepared,
                authorization,
                {
                    "schema_version": "1",
                    "protocol": PROTOCOL,
                    "entries": [],
                },
                fixture,
            )
        except LaunchControlError:
            scenarios["direct_runner_without_seal"] = True

    violations = sorted(name for name, passed in scenarios.items() if not passed)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "launch_controls_ready" if not violations else "authorization_or_operation_violation",
        "exit_code": EXIT_READY if not violations else EXIT_VIOLATION,
        "query_count": 1000,
        "shard_count": 20,
        "operation_count": len(audit.entries),
        "selected_attempts_sha256": stable_hash(machine.selected_attempts),
        "audit_sha256": audit.audit_sha256,
        "scenario_count": len(scenarios),
        "scenarios": [{"scenario": name, "blocked": scenarios[name]} for name in sorted(scenarios)],
        "violations": violations,
        "fixture_only": True,
        "network_status": "not_checked",
        "credential_status": "not_checked",
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }


def audit_readiness(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    plan, _ = _validate_bound_inputs(root, protocol)
    _validate_freshness(root, protocol)
    checks = {
        "query_population_closed": plan["population"]["count"] == 1000,
        "twenty_shards_bound": plan["sharding"]["shard_count"] == 20,
        "full_restart_required": plan["resume"]["start_mode"] == "full_restart_all_1000",
        "legacy_record_reuse_rejected": plan["legacy_artifacts"]["reuse_as_completed"] is False,
        "current_rules_only": protocol["execution_contract"]["current_rules_only"] is True,
        "tiebreak_v2_disabled": protocol["execution_contract"]["deterministic_tiebreak_v2_enabled"] is False,
        "observability_closed": all(protocol["execution_contract"]["required_observability"].values()),
        "network_not_checked": True,
        "credentials_not_checked": True,
    }
    if not all(checks.values()):
        raise LaunchControlError("readiness_check_failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "external_activation_blocked",
        "exit_code": EXIT_BLOCKED,
        "controls_ready": True,
        "checks": checks,
        "network_status": "not_checked",
        "credential_status": "not_checked",
        "official_run_started": False,
        "full1000_completed": False,
        "formal_validation_complete": False,
        "execution": EXECUTION,
    }


def verify_audit_log(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parsed = OperationAuditLog.model_validate(value)
    except ValidationError as exc:
        raise LaunchControlError("audit_log_invalid") from exc
    text = canonical_json(parsed.model_dump(mode="json")).decode("utf-8").casefold()
    if any(token in text for token in _PROHIBITED_TEXT):
        raise LaunchControlError("sensitive_audit_content")
    return {
        "status": "launch_controls_ready",
        "exit_code": EXIT_READY,
        "entry_count": len(parsed.entries),
        "audit_sha256": parsed.audit_sha256,
    }


def validate_launch_evidence(
    prepared: Mapping[str, Any],
    authorization: Mapping[str, Any],
    audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Reject runner output that was not preceded by the sealed two-stage flow."""

    validate_authorization(prepared, authorization, protocol)
    try:
        parsed = OperationAuditLog.model_validate(audit)
    except ValidationError as exc:
        raise LaunchControlError("audit_log_invalid") from exc
    if (
        parsed.preparation_sha256 != prepared["preparation_sha256"]
        or parsed.authorization_sha256 != authorization["authorization_sha256"]
    ):
        raise LaunchControlError("audit_authorization_binding_drift")
    if [entry.event for entry in parsed.entries[:2]] != ["authorized", "started"]:
        raise LaunchControlError("runner_started_without_two_stage_authorization")
