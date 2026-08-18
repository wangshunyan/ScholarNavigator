"""Offline runtime hermeticity gate for the production Replay seam.

``runtime_hermeticity_v1`` launches a pre-imported worker in a controlled
subprocess, then audits only the business-execution boundary.  The worker uses
the repository's canonical SearchService Replay fixture; it does not load
runtime configuration, gold, Snapshot writers, or quality metrics.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import locale
import os
import socket
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scholar_agent.evaluation.current_rules_regression import compare_profiles
from scholar_agent.evaluation.execution_determinism import (
    ExecutionDeterminismError,
    FixtureNotEligible,
    load_protocol as load_execution_protocol,
    replay_canonical_fixture,
)
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash


CONTRACT_VERSION = "runtime_hermeticity_v1"
SCHEMA_VERSION = "1"
GATE_NAME = "runtime_hermeticity_gate"
EXIT_PASSED = 0
EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION = 2
EXIT_NOT_ELIGIBLE = 3
EXIT_USAGE_ERROR = 4
SCORE_SCOPE = "runtime_hermeticity_only_not_quality_or_official_score"

_ORIGINAL_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_ENVIRON = os.environ
_ORIGINAL_MUTATIONS = {
    name: getattr(os, name)
    for name in ("mkdir", "makedirs", "remove", "rename", "replace", "rmdir", "unlink")
    if hasattr(os, name)
}
_SENSITIVE_ENV_KEYS = frozenset(
    {
        "API_KEY",
        "AUTHORIZATION",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
    }
)
_FAULTS = frozenset(
    {
        "dotenv_read",
        "network_attempt",
        "forbidden_file_read",
        "forbidden_file_write",
        "cache_residue",
        "subprocess_attempt",
        "sensitive_environment_read",
        "sensitive_sentinel_echo",
        "hash_seed_semantic_drift",
        "timezone_semantic_drift",
        "cwd_semantic_drift",
        "home_semantic_drift",
    }
)


class RuntimeHermeticityError(RuntimeError):
    """Malformed protocol or unsafe invocation."""


class RuntimeHermeticityNotEligible(RuntimeHermeticityError):
    """Fixture or frozen run cannot satisfy the declared contract."""


class HermeticityBlocked(RuntimeError):
    """A business operation was blocked by the worker audit boundary."""


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_protocol(path: Path, *, repository_root: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeHermeticityError("protocol_unreadable") from exc
    if not isinstance(value, dict):
        raise RuntimeHermeticityError("protocol_root_must_be_object")
    if value.get("contract") != CONTRACT_VERSION:
        raise RuntimeHermeticityError("protocol_contract_invalid")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeHermeticityError("protocol_schema_version_invalid")
    if value.get("score_scope") != SCORE_SCOPE:
        raise RuntimeHermeticityError("protocol_score_scope_invalid")
    execution_protocol = _file_spec(value, "execution_protocol")
    execution_path = _repository_file(repository_root, execution_protocol["path"])
    _verify_file(execution_path, execution_protocol)
    execution = load_execution_protocol(execution_path, repository_root=repository_root)
    fixture_path = _repository_file(
        repository_root, str(execution["fixture"]["retrieval_outputs_path"])
    )
    allowed_inputs = value.get("allowed_inputs")
    if not isinstance(allowed_inputs, list) or len(allowed_inputs) != 2:
        raise RuntimeHermeticityError("allowed_inputs_must_be_two_exact_files")
    indexed = {
        str(item.get("role")): item
        for item in allowed_inputs
        if isinstance(item, dict)
    }
    if set(indexed) != {"execution_protocol", "retrieval_fixture"}:
        raise RuntimeHermeticityError("allowed_input_roles_invalid")
    for role, expected_path in (
        ("execution_protocol", execution_path),
        ("retrieval_fixture", fixture_path),
    ):
        spec = indexed[role]
        path_value = _repository_file(repository_root, str(spec.get("path") or ""))
        if path_value != expected_path:
            raise RuntimeHermeticityError(f"allowed_input_path_mismatch:{role}")
        _verify_file(path_value, spec)
    profiles = value.get("environment_profiles")
    if not isinstance(profiles, list) or len(profiles) < 7:
        raise RuntimeHermeticityError("environment_profiles_incomplete")
    profile_ids = [
        str(item.get("profile_id") or "")
        for item in profiles
        if isinstance(item, dict)
    ]
    if len(profile_ids) != len(profiles) or len(profile_ids) != len(set(profile_ids)):
        raise RuntimeHermeticityError("environment_profile_identity_invalid")
    required_kinds = {
        "minimal_environment",
        "hash_seed",
        "working_directory_home_tmpdir",
        "timezone",
        "locale",
        "thread_environment",
        "polluted_environment",
    }
    if required_kinds - {str(item.get("kind")) for item in profiles}:
        raise RuntimeHermeticityError("environment_profile_kind_missing")
    canonicalization = value.get("canonicalization")
    if not isinstance(canonicalization, dict) or canonicalization.get("policy") != (
        "reuse_execution_determinism_explicit_paths_unknown_fields_preserved"
    ):
        raise RuntimeHermeticityError("canonicalization_policy_invalid")
    if (
        value.get("audit_boundary")
        != "after_dependency_import_before_business_execution"
    ):
        raise RuntimeHermeticityError("audit_boundary_invalid")
    if value.get("allowed_outputs") != [
        {
            "role": "gate_worker_report",
            "scope": "isolated_profile_output_directory",
        }
    ]:
        raise RuntimeHermeticityError("allowed_output_contract_invalid")
    if value.get("business_execution") != {
        "dynamic_configuration_loading": False,
        "entrypoint": "execution_determinism.replay_canonical_fixture",
        "llm_enabled": False,
        "snapshot_writes_enabled": False,
    }:
        raise RuntimeHermeticityError("business_execution_contract_invalid")
    return value


def run_runtime_hermeticity_gate(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    fault: str | None = None,
    profile_ids: Sequence[str] | None = None,
    worker_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Execute the declared Replay fixture in isolated environment profiles."""

    if fault is not None and fault not in _FAULTS:
        raise RuntimeHermeticityError("unsupported_fault")
    root = repository_root.resolve()
    declared_profiles = list(protocol["environment_profiles"])
    if profile_ids is not None:
        requested = list(profile_ids)
        if len(requested) != len(set(requested)):
            raise RuntimeHermeticityError("duplicate_requested_profile")
        index = {item["profile_id"]: item for item in declared_profiles}
        unknown = sorted(set(requested) - set(index))
        if unknown:
            raise RuntimeHermeticityError("unknown_requested_profile")
        declared_profiles = [index[item] for item in requested]
    if not declared_profiles:
        raise RuntimeHermeticityError("no_environment_profile_selected")

    worker = root / "scripts" / "runtime_hermeticity_worker.py"
    if not worker.is_file():
        raise RuntimeHermeticityNotEligible("worker_entrypoint_missing")
    execution_spec = _file_spec(protocol, "execution_protocol")
    execution_path = _repository_file(root, execution_spec["path"])
    execution = load_execution_protocol(execution_path, repository_root=root)
    fixture_path = _repository_file(
        root, str(execution["fixture"]["retrieval_outputs_path"])
    )
    allowed_inputs = [
        {
            "role": "execution_protocol",
            "path": str(execution_path),
            "sha256": sha256_file(execution_path),
        },
        {
            "role": "retrieval_fixture",
            "path": str(fixture_path),
            "sha256": sha256_file(fixture_path),
        },
    ]
    reports: list[dict[str, Any]] = []
    semantic_payloads: dict[str, dict[str, Any]] = {}
    fault_position = 1 if fault is not None and len(declared_profiles) > 1 else 0
    with tempfile.TemporaryDirectory(prefix="runtime-hermeticity-") as temporary:
        sandbox = Path(temporary)
        for position, profile in enumerate(declared_profiles):
            controlled_fault = fault if position == fault_position else None
            report, semantic = _run_profile(
                profile,
                repository_root=root,
                worker=worker,
                execution_protocol=execution_path,
                allowed_inputs=allowed_inputs,
                sandbox_root=sandbox,
                controlled_fault=controlled_fault,
                timeout_seconds=worker_timeout_seconds,
                allowed_environment_keys=list(
                    protocol["allowed_environment_variables"]
                ),
            )
            reports.append(report)
            if semantic is not None:
                semantic_payloads[str(profile["profile_id"])] = semantic

    violations: list[dict[str, Any]] = []
    supported = [item for item in reports if item["status"] != "profile_not_supported"]
    baseline_profile = next(
        (item for item in supported if item["profile_id"] == "minimal"),
        supported[0] if supported else None,
    )
    if baseline_profile is None:
        return _report(
            status="profile_not_supported",
            exit_code=EXIT_NOT_ELIGIBLE,
            profiles=reports,
            violations=[],
            comparison_rows=[],
            fault=fault,
        )
    baseline_id = str(baseline_profile["profile_id"])
    baseline = semantic_payloads.get(baseline_id)
    if baseline is None:
        violations.append(
            _violation(
                profile_id=baseline_id,
                operation_type="business_execution",
                invariant="baseline_semantic_output_available",
                path="$.semantic_output",
            )
        )
    comparison_rows: list[dict[str, Any]] = []
    for profile in supported:
        profile_id = str(profile["profile_id"])
        for item in profile.get("violations", []):
            violations.append(item)
        semantic = semantic_payloads.get(profile_id)
        differences = (
            compare_profiles(baseline, semantic, max_diffs=1)
            if baseline is not None and semantic is not None
            else [{"path": "$.semantic_output"}]
        )
        comparison_rows.append(
            {
                "profile_id": profile_id,
                "baseline_profile_id": baseline_id,
                "status": "passed" if not differences else "semantic_violation",
                "semantic_sha256": stable_hash(semantic),
                "first_difference_path": (
                    differences[0]["path"] if differences else None
                ),
            }
        )
        if differences:
            violations.append(
                _violation(
                    profile_id=profile_id,
                    operation_type="semantic_comparison",
                    invariant="normalized_semantics_equal_across_profiles",
                    path=str(differences[0]["path"]),
                    left=baseline,
                    right=semantic,
                )
            )
    violations = sorted(
        violations,
        key=lambda item: (
            item["profile_id"],
            item["operation_type"],
            item["invariant"],
            item["first_difference_path"],
        ),
    )
    unsupported = [
        item for item in reports if item["status"] == "profile_not_supported"
    ]
    if violations:
        status = "hermeticity_or_semantic_violation"
        exit_code = EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION
    elif unsupported:
        status = "profile_not_supported"
        exit_code = EXIT_NOT_ELIGIBLE
    else:
        status = "passed"
        exit_code = EXIT_PASSED
    return _report(
        status=status,
        exit_code=exit_code,
        profiles=reports,
        violations=violations,
        comparison_rows=comparison_rows,
        fault=fault,
    )


def audit_frozen_baseline_eligibility(
    protocol: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    spec = _file_spec(protocol, "frozen_baseline_eligibility")
    path = _repository_file(repository_root, spec["path"])
    _verify_file(path, spec)
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = []
    for item in sorted(payload.get("profiles", []), key=lambda row: row["profile_id"]):
        profiles.append(
            {
                "profile_id": item["profile_id"],
                "status": "not_eligible",
                "reason": (
                    "declarative_io_contract_and_self_contained_replay_unavailable"
                ),
                "observed_record_count": item.get("observed_record_count"),
                "expected_query_count": item.get("expected_query_count"),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": "not_eligible",
        "exit_code": EXIT_NOT_ELIGIBLE,
        "score_scope": SCORE_SCOPE,
        "profiles": profiles,
        "eligible_count": 0,
        "profile_count": len(profiles),
        "execution": _zero_execution(),
    }


class AuditedEnvironment(MutableMapping[str, str]):
    """Environment proxy that attributes business-stage key access."""

    def __init__(self, audit: "BusinessIOAudit", backing: MutableMapping[str, str]):
        self._audit = audit
        self._backing = backing

    def __getitem__(self, key: str) -> str:
        self._audit.environment_read(str(key))
        return self._backing[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._audit.block("environment_write", f"environment:{key}")

    def __delitem__(self, key: str) -> None:
        self._audit.block("environment_write", f"environment:{key}")

    def __iter__(self) -> Iterator[str]:
        self._audit.block("environment_enumeration", "environment:all")
        return iter(())

    def __len__(self) -> int:
        return len(self._backing)


class BusinessIOAudit:
    """Content-I/O hooks enabled only around the Replay business call."""

    def __init__(
        self,
        *,
        allowed_inputs: Sequence[Mapping[str, Any]],
        allowed_environment_keys: Sequence[str],
        output_root: Path,
        sandbox_root: Path,
    ) -> None:
        self.allowed_inputs = {
            Path(str(item["path"])).resolve(): str(item["role"])
            for item in allowed_inputs
        }
        self.allowed_environment_keys = frozenset(allowed_environment_keys)
        self.output_root = output_root.resolve()
        self.sandbox_root = sandbox_root.resolve()
        self.operations: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []

    @contextmanager
    def activate(self):
        environment = AuditedEnvironment(self, _ORIGINAL_ENVIRON)
        with ExitStack() as stack:
            stack.enter_context(patch.object(builtins, "open", self.open))
            stack.enter_context(patch.object(io, "open", self.io_open))
            stack.enter_context(patch.object(os, "open", self.os_open))
            stack.enter_context(patch.object(os, "scandir", self.scandir))
            stack.enter_context(patch.object(os, "listdir", self.listdir))
            for name in _ORIGINAL_MUTATIONS:
                stack.enter_context(
                    patch.object(os, name, self._mutation_wrapper(name))
                )
            stack.enter_context(patch.object(os, "environ", environment))
            stack.enter_context(patch.object(socket, "create_connection", self.network))
            stack.enter_context(patch.object(socket.socket, "connect", self.network))
            stack.enter_context(patch.object(socket.socket, "connect_ex", self.network))
            for name in (
                "getaddrinfo",
                "gethostbyname",
                "gethostbyname_ex",
                "gethostbyaddr",
            ):
                stack.enter_context(patch.object(socket, name, self.network))
            stack.enter_context(patch.object(subprocess, "Popen", self.subprocess))
            for name in ("system", "posix_spawn", "posix_spawnp"):
                if hasattr(os, name):
                    stack.enter_context(patch.object(os, name, self.subprocess))
            for name in ("fork", "forkpty"):
                if hasattr(os, name):
                    stack.enter_context(patch.object(os, name, self.subprocess))
            yield self

    def open(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any):
        self._authorize_file(file, mode)
        return _ORIGINAL_OPEN(file, mode, *args, **kwargs)

    def io_open(self, file: Any, mode: str = "r", *args: Any, **kwargs: Any):
        self._authorize_file(file, mode)
        return _ORIGINAL_IO_OPEN(file, mode, *args, **kwargs)

    def os_open(self, path: Any, flags: int, *args: Any, **kwargs: Any):
        write = bool(
            flags
            & (
                os.O_WRONLY
                | os.O_RDWR
                | os.O_APPEND
                | os.O_CREAT
                | os.O_TRUNC
            )
        )
        self._authorize_file(path, "w" if write else "r")
        return _ORIGINAL_OS_OPEN(path, flags, *args, **kwargs)

    def scandir(self, path: Any = "."):
        self.block("directory_enumeration", self.resource_identity(path))

    def listdir(self, path: Any = "."):
        self.block("directory_enumeration", self.resource_identity(path))

    def _mutation_wrapper(self, name: str):
        original = _ORIGINAL_MUTATIONS[name]

        def mutate(path: Any, *args: Any, **kwargs: Any):
            paths = [path]
            if name in {"rename", "replace"} and args:
                paths.append(args[0])
            for item in paths:
                resolved = Path(os.fspath(item)).resolve()
                try:
                    resolved.relative_to(self.output_root)
                except ValueError:
                    self.block("file_mutation", self.resource_identity(item))
            self.operations.append(
                {
                    "operation_type": "file_mutation",
                    "resource": self.resource_identity(path),
                }
            )
            return original(path, *args, **kwargs)

        return mutate

    def network(self, *_args: Any, **_kwargs: Any):
        self.block("network_attempt", "network:blocked")

    def subprocess(self, *_args: Any, **_kwargs: Any):
        self.block("subprocess_attempt", "subprocess:unregistered")

    def environment_read(self, key: str) -> None:
        identity = f"environment:{key}"
        if key not in self.allowed_environment_keys:
            self.block("environment_read", identity)
        self.operations.append(
            {"operation_type": "environment_read", "resource": identity}
        )

    def block(self, operation_type: str, resource: str) -> None:
        violation = {
            "operation_type": operation_type,
            "resource_identity": self._sanitize_resource(resource),
            "invariant": "business_io_must_be_declared",
        }
        self.violations.append(violation)
        raise HermeticityBlocked(operation_type)

    def resource_identity(self, value: Any) -> str:
        if isinstance(value, int):
            return "file_descriptor"
        path = Path(os.fspath(value)).resolve()
        role = self.allowed_inputs.get(path)
        if role:
            return f"input:{role}"
        try:
            relative = path.relative_to(self.sandbox_root).as_posix()
        except ValueError:
            return f"external:{hashlib.sha256(path.name.encode()).hexdigest()}"
        return f"sandbox:{relative}"

    def summary(self) -> dict[str, Any]:
        counts = Counter(item["operation_type"] for item in self.operations)
        resources = Counter(item["resource"] for item in self.operations)
        return {
            "operation_counts": dict(sorted(counts.items())),
            "allowed_resource_counts": dict(sorted(resources.items())),
            "violation_count": len(self.violations),
            "violations": list(self.violations),
        }

    def _authorize_file(self, value: Any, mode: str) -> None:
        identity = self.resource_identity(value)
        write = any(marker in mode for marker in ("w", "a", "x", "+"))
        operation = "file_write" if write else "file_read"
        if isinstance(value, int):
            self.block(operation, identity)
        path = Path(os.fspath(value)).resolve()
        if write:
            try:
                path.relative_to(self.output_root)
            except ValueError:
                self.block(operation, identity)
        elif path not in self.allowed_inputs:
            self.block(operation, identity)
        self.operations.append({"operation_type": operation, "resource": identity})

    @staticmethod
    def _sanitize_resource(value: str) -> str:
        lowered = value.casefold()
        if ".env" in lowered:
            return "forbidden:dotenv"
        if any(key.casefold() in lowered for key in _SENSITIVE_ENV_KEYS):
            return "forbidden:sensitive_environment"
        if value.startswith(("input:", "sandbox:", "network:", "subprocess:")):
            return value
        return f"opaque:{hashlib.sha256(value.encode()).hexdigest()}"


def run_worker_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Worker-side entry after all dependencies have been imported."""

    profile_id = str(request["profile_id"])
    locale_name = str(request.get("locale") or "")
    if locale_name:
        try:
            locale.setlocale(locale.LC_ALL, locale_name)
        except locale.Error:
            return {
                "schema_version": SCHEMA_VERSION,
                "contract": CONTRACT_VERSION,
                "status": "profile_not_supported",
                "profile_id": profile_id,
                "reason": "locale_profile_not_supported",
            }
    audit = BusinessIOAudit(
        allowed_inputs=request["allowed_inputs"],
        allowed_environment_keys=request["allowed_environment_keys"],
        output_root=Path(str(request["output_root"])),
        sandbox_root=Path(str(request["sandbox_root"])),
    )
    semantic: dict[str, Any] | None = None
    error_type: str | None = None
    fault = request.get("fault")
    try:
        with audit.activate():
            _inject_pre_replay_fault(fault, request)
            protocol = load_execution_protocol(
                Path(str(request["execution_protocol"])),
                repository_root=Path(str(request["repository_root"])),
            )
            semantic = replay_canonical_fixture(
                protocol,
                repository_root=Path(str(request["repository_root"])),
                snapshot_root=Path(str(request["snapshot_root"])),
                collect_result_lineage=True,
                install_network_guard=False,
            )
            _inject_post_replay_fault(fault, semantic, request)
    except HermeticityBlocked:
        error_type = "HermeticityBlocked"
    except (ExecutionDeterminismError, FixtureNotEligible, OSError, ValueError):
        error_type = "BusinessExecutionError"
    summary = audit.summary()
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "status": (
            "passed"
            if semantic is not None and not summary["violations"] and error_type is None
            else "hermeticity_or_semantic_violation"
        ),
        "profile_id": profile_id,
        "semantic_payload": semantic,
        "semantic_sha256": stable_hash(semantic),
        "business_io": summary,
        "error_type": error_type,
    }


def _run_profile(
    profile: Mapping[str, Any],
    *,
    repository_root: Path,
    worker: Path,
    execution_protocol: Path,
    allowed_inputs: Sequence[Mapping[str, Any]],
    sandbox_root: Path,
    controlled_fault: str | None,
    timeout_seconds: float,
    allowed_environment_keys: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    profile_id = str(profile["profile_id"])
    profile_root = sandbox_root / profile_id
    cwd = profile_root / str(profile.get("cwd_variant") or "cwd")
    home = profile_root / str(profile.get("home_variant") or "home")
    temp = profile_root / str(profile.get("tmp_variant") or "tmp")
    output = profile_root / "output"
    bootstrap = profile_root / "bootstrap"
    for path in (cwd, home, temp, output, bootstrap):
        path.mkdir(parents=True, exist_ok=True)
    sentinel_values = _write_sentinel_inputs(cwd, home)
    env = _profile_environment(profile, home=home, temp=temp)
    sentinel_values.extend(_polluted_environment(env, profile))
    request_path = bootstrap / "request.json"
    response_path = output / "worker-response.json"
    request = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "profile_id": profile_id,
        "repository_root": str(repository_root),
        "execution_protocol": str(execution_protocol),
        "allowed_inputs": list(allowed_inputs),
        "allowed_environment_keys": list(allowed_environment_keys),
        "output_root": str(output),
        "sandbox_root": str(profile_root),
        "snapshot_root": str(output / "snapshot-state"),
        "locale": profile.get("locale"),
        "fault": controlled_fault,
        "sentinel_dotenv": str(cwd / ".env"),
        "sentinel_home_file": str(home / ".config" / "agent" / "credentials.json"),
        "forbidden_write": str(profile_root / "forbidden-output.txt"),
        "cache_residue": str(output / "undeclared-cache.bin"),
    }
    request_path.write_bytes(stable_json_bytes(request))
    command = [
        sys.executable,
        "-I",
        str(worker),
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    timed_out = False
    stdout = b""
    stderr = b""
    return_code: int | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
    leaked = any(
        sentinel.encode("utf-8") in payload
        for sentinel in sentinel_values
        for payload in (
            stdout,
            stderr,
            response_path.read_bytes() if response_path.exists() else b"",
        )
    )
    child: dict[str, Any] | None = None
    if response_path.is_file() and not leaked:
        try:
            child = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            child = None
    expected_files = {"worker-response.json"}
    residual_files = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).as_posix() not in expected_files
    )
    violations: list[dict[str, Any]] = []
    if timed_out:
        violations.append(
            _violation(
                profile_id,
                "subprocess_timeout",
                "worker_exits_within_bound",
                "$.worker",
            )
        )
    if return_code not in (0, None):
        violations.append(
            _violation(
                profile_id,
                "worker_exit",
                "worker_exit_code_zero",
                "$.worker.exit_code",
            )
        )
    if stdout or stderr:
        violations.append(
            _violation(
                profile_id,
                "worker_output",
                "worker_stdio_empty",
                "$.worker.stdio",
            )
        )
    if leaked:
        violations.append(
            _violation(
                profile_id,
                "sensitive_output",
                "sensitive_sentinel_never_emitted",
                "$.worker.output",
            )
        )
    if child is None:
        violations.append(
            _violation(
                profile_id,
                "worker_protocol",
                "worker_response_valid",
                "$.worker.response",
            )
        )
    elif child.get("status") != "profile_not_supported":
        for item in child.get("business_io", {}).get("violations", []):
            violations.append(
                _violation(
                    profile_id,
                    str(item.get("operation_type") or "business_io"),
                    str(item.get("invariant") or "business_io_must_be_declared"),
                    "$.business_io",
                    resource_identity=str(item.get("resource_identity") or "opaque"),
                )
            )
    if residual_files:
        violations.append(
            _violation(
                profile_id,
                "file_residue",
                "no_undeclared_output_or_cache_residue",
                "$.output_files",
                right=residual_files,
            )
        )
    if child and child.get("status") == "profile_not_supported":
        return (
            {
                "profile_id": profile_id,
                "kind": profile["kind"],
                "status": "profile_not_supported",
                "reason": child.get("reason"),
                "violations": violations,
            },
            None,
        )
    semantic = child.get("semantic_payload") if child else None
    business_io = child.get("business_io", {}) if child else {}
    report = {
        "profile_id": profile_id,
        "kind": profile["kind"],
        "status": (
            "passed"
            if not violations and semantic is not None
            else "hermeticity_or_semantic_violation"
        ),
        "semantic_sha256": stable_hash(semantic),
        "records_sha256": (
            semantic.get("records_sha256") if isinstance(semantic, dict) else None
        ),
        "lineage_sha256": (
            semantic.get("result_lineage", {}).get("summaries_sha256")
            if isinstance(semantic, dict)
            else None
        ),
        "business_io": {
            "operation_counts": business_io.get("operation_counts", {}),
            "allowed_resource_counts": business_io.get("allowed_resource_counts", {}),
            "network_request_count": sum(
                item.get("operation_type") == "network_attempt"
                for item in business_io.get("violations", [])
            ),
            "subprocess_request_count": sum(
                item.get("operation_type") == "subprocess_attempt"
                for item in business_io.get("violations", [])
            ),
            "snapshot_write_count": 0,
            "llm_request_count": 0,
        },
        "residual_file_count": len(residual_files),
        "violations": violations,
    }
    return report, semantic


def _profile_environment(
    profile: Mapping[str, Any], *, home: Path, temp: Path
) -> dict[str, str]:
    values = {
        "HOME": str(home),
        "TMPDIR": str(temp),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": str(profile.get("python_hash_seed") or "1"),
        "PYTHONUNBUFFERED": "1",
        "TZ": str(profile.get("timezone") or "UTC"),
        "LC_ALL": str(profile.get("locale") or "C"),
        "LANG": str(profile.get("locale") or "C"),
        "OMP_NUM_THREADS": str(profile.get("thread_count") or "1"),
        "OPENBLAS_NUM_THREADS": str(profile.get("thread_count") or "1"),
        "MKL_NUM_THREADS": str(profile.get("thread_count") or "1"),
        "NUMEXPR_NUM_THREADS": str(profile.get("thread_count") or "1"),
    }
    return values


def _polluted_environment(env: dict[str, str], profile: Mapping[str, Any]) -> list[str]:
    if profile.get("kind") != "polluted_environment":
        return []
    sentinels = [f"HERMETICITY_SENTINEL_{index:03d}" for index in range(32)]
    for index, value in enumerate(sentinels):
        env[f"UNRELATED_VARIABLE_{index:03d}"] = value
    sensitive = {
        "API_KEY": "HERMETICITY_SENTINEL_API_KEY",
        "AUTHORIZATION": "HERMETICITY_SENTINEL_AUTHORIZATION",
        "HF_TOKEN": "HERMETICITY_SENTINEL_HF_TOKEN",
        "OPENAI_API_KEY": "HERMETICITY_SENTINEL_OPENAI_KEY",
        "SEMANTIC_SCHOLAR_API_KEY": "HERMETICITY_SENTINEL_S2_KEY",
    }
    env.update(sensitive)
    return [*sentinels, *sensitive.values()]


def _write_sentinel_inputs(cwd: Path, home: Path) -> list[str]:
    dotenv = "HERMETICITY_SENTINEL_DOTENV_VALUE"
    home_value = "HERMETICITY_SENTINEL_HOME_VALUE"
    (cwd / ".env").write_text(f"API_KEY={dotenv}\n", encoding="utf-8")
    config = home / ".config" / "agent"
    config.mkdir(parents=True, exist_ok=True)
    (config / "credentials.json").write_text(
        json.dumps({"token": home_value}, sort_keys=True) + "\n", encoding="utf-8"
    )
    return [
        dotenv,
        home_value,
        "HERMETICITY_SENTINEL_API_KEY",
        "HERMETICITY_SENTINEL_AUTHORIZATION",
        "HERMETICITY_SENTINEL_HF_TOKEN",
        "HERMETICITY_SENTINEL_OPENAI_KEY",
        "HERMETICITY_SENTINEL_S2_KEY",
    ]


def _inject_pre_replay_fault(fault: Any, request: Mapping[str, Any]) -> None:
    if fault == "dotenv_read":
        Path(str(request["sentinel_dotenv"])).read_text(encoding="utf-8")
    elif fault == "forbidden_file_read":
        Path(str(request["sentinel_home_file"])).read_text(encoding="utf-8")
    elif fault == "network_attempt":
        socket.getaddrinfo("invalid.example", 443)
    elif fault == "forbidden_file_write":
        Path(str(request["forbidden_write"])).write_text("blocked", encoding="utf-8")
    elif fault == "subprocess_attempt":
        subprocess.run(["unregistered-child"], check=False)
    elif fault == "sensitive_environment_read":
        os.environ["API_KEY"]


def _inject_post_replay_fault(
    fault: Any, semantic: dict[str, Any], request: Mapping[str, Any]
) -> None:
    if fault == "cache_residue":
        Path(str(request["cache_residue"])).write_bytes(b"cache")
    elif fault == "sensitive_sentinel_echo":
        semantic["controlled_sensitive_echo"] = _ORIGINAL_ENVIRON.get(
            "API_KEY", "HERMETICITY_SENTINEL_API_KEY"
        )
    elif fault == "hash_seed_semantic_drift":
        semantic["controlled_environment_dependency"] = _ORIGINAL_ENVIRON.get(
            "PYTHONHASHSEED"
        )
    elif fault == "timezone_semantic_drift":
        semantic["controlled_environment_dependency"] = _ORIGINAL_ENVIRON.get("TZ")
    elif fault == "cwd_semantic_drift":
        semantic["controlled_environment_dependency"] = Path.cwd().name
    elif fault == "home_semantic_drift":
        semantic["controlled_environment_dependency"] = Path(
            _ORIGINAL_ENVIRON.get("HOME", "")
        ).name


def _report(
    *,
    status: str,
    exit_code: int,
    profiles: Sequence[Mapping[str, Any]],
    violations: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    fault: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_VERSION,
        "gate": GATE_NAME,
        "status": status,
        "exit_code": exit_code,
        "score_scope": SCORE_SCOPE,
        "audit_boundary": "after_dependency_import_before_business_execution",
        "canonicalization": {
            "policy": (
                "reuse_execution_determinism_explicit_paths_unknown_fields_preserved"
            ),
            "list_order_policy": "preserved",
        },
        "profile_count": len(profiles),
        "supported_profile_count": sum(
            item["status"] != "profile_not_supported" for item in profiles
        ),
        "profiles": list(profiles),
        "semantic_comparisons": list(comparison_rows),
        "violation_count": len(violations),
        "violations": list(violations),
        "execution": {
            "network_request_count": sum(
                int(item.get("business_io", {}).get("network_request_count") or 0)
                for item in profiles
            ),
            "subprocess_request_count": sum(
                int(
                    item.get("business_io", {}).get("subprocess_request_count")
                    or 0
                )
                for item in profiles
            ),
            "llm_request_count": sum(
                int(item.get("business_io", {}).get("llm_request_count") or 0)
                for item in profiles
            ),
            "snapshot_write_count": sum(
                int(item.get("business_io", {}).get("snapshot_write_count") or 0)
                for item in profiles
            ),
            "quality_metric_count": 0,
            "controlled_fault": fault,
        },
    }


def _zero_execution() -> dict[str, int]:
    return {
        "network_request_count": 0,
        "subprocess_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "quality_metric_count": 0,
    }


def _violation(
    profile_id: str,
    operation_type: str,
    invariant: str,
    path: str,
    *,
    resource_identity: str | None = None,
    left: Any = None,
    right: Any = None,
) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "operation_type": operation_type,
        "resource_identity": resource_identity,
        "invariant": invariant,
        "first_difference_path": path,
        "left_sha256": stable_hash(left),
        "right_sha256": stable_hash(right),
    }


def _file_spec(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    item = value.get(field)
    if not isinstance(item, dict):
        raise RuntimeHermeticityError(f"{field}_missing")
    if not isinstance(item.get("path"), str):
        raise RuntimeHermeticityError(f"{field}_path_invalid")
    return item


def _repository_file(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise RuntimeHermeticityError("repository_path_invalid")
    resolved_root = root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeHermeticityError("repository_path_escape") from exc
    return resolved


def _verify_file(path: Path, spec: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeHermeticityNotEligible("declared_file_missing")
    if path.stat().st_size != spec.get("size_bytes"):
        raise RuntimeHermeticityNotEligible("declared_file_size_drift")
    if sha256_file(path) != spec.get("sha256"):
        raise RuntimeHermeticityNotEligible("declared_file_hash_drift")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(payload))
