"""Hermetic, immutable handoff gate for a future official scorer.

The gate deliberately knows nothing about an official scorer schema or metric.
It proves the transport and isolation chain with strict synthetic packages, and
keeps the real readiness result blocked while the official package and the
complete Full1000 run are unavailable.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any
from unittest.mock import patch


CONTRACT_VERSION = "external_scorer_handoff_v1"
PACKAGE_VERSION = "scorer_package_manifest_v1"
HANDOFF_VERSION = "canonical_scorer_handoff_v1"
OUTPUT_VERSION = "synthetic_scorer_output_v1"
SCHEMA_VERSION = "1"
EXIT_VERIFIED = 0
EXIT_VIOLATION = 2
EXIT_BLOCKED = 3
EXIT_USAGE = 4
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_NAMESPACE = "synthetic_handoff"
SENSITIVE_MARKER = "SYNTHETIC_SECRET_SENTINEL"


class ExternalScorerError(RuntimeError):
    """Package, handoff, sandbox, or scorer output violated the contract."""


class ExternalScorerBlocked(ExternalScorerError):
    """Official scorer or complete authoritative input is unavailable."""


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalScorerError("json_input_invalid") from exc
    if not isinstance(value, dict):
        raise ExternalScorerError("json_root_not_object")
    return value


def _repo_file(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ExternalScorerError("unsafe_repository_path")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ExternalScorerError("repository_path_escape") from exc
    return path


def load_protocol(path: str | Path, *, repository_root: str | Path = REPOSITORY_ROOT) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    value = _read_json(Path(path).resolve())
    if value.get("analysis") != CONTRACT_VERSION or value.get("schema_version") != SCHEMA_VERSION:
        raise ExternalScorerError("protocol_version_invalid")
    expected_execution = {
        "gold_or_qrels_loaded": False,
        "llm_request_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "snapshot_write_count": 0,
    }
    if value.get("execution") != expected_execution:
        raise ExternalScorerError("offline_execution_contract_drift")
    official = value.get("official_package") or {}
    if official != {
        "entrypoint": "not_provided",
        "input_schema": "not_provided",
        "metric_namespace": "not_provided",
        "output_schema": "not_provided",
        "package_sha256": "not_provided",
        "runtime": "not_provided",
        "scorer_name": "unknown",
        "scorer_version": "unknown",
    }:
        raise ExternalScorerError("official_unknown_fields_were_inferred")
    for spec in (value.get("frozen_readiness_inputs") or {}).values():
        file_path = _repo_file(root, str(spec.get("path") or ""))
        if not file_path.is_file() or sha256_file(file_path) != spec.get("sha256"):
            raise ExternalScorerBlocked("frozen_readiness_input_unavailable")
    matrix = value.get("synthetic_matrix")
    if not isinstance(matrix, list) or len(matrix) != len(set(matrix)):
        raise ExternalScorerError("synthetic_matrix_invalid")
    return value


def canonical_handoff(
    queries: Sequence[Mapping[str, Any]],
    *,
    run_manifest_sha256: str,
    commit_generation_sha256: str,
    source_scope: str = "synthetic_conformance_fixture",
) -> dict[str, Any]:
    if len(run_manifest_sha256) != 64 or len(commit_generation_sha256) != 64:
        raise ExternalScorerError("handoff_lineage_binding_invalid")
    normalized: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    for expected_order, query in enumerate(queries):
        if set(query) != {"query_identity", "query_order", "results"}:
            raise ExternalScorerError("handoff_query_schema_invalid")
        identity = str(query["query_identity"])
        if not identity or identity in seen_queries or int(query["query_order"]) != expected_order:
            raise ExternalScorerError("handoff_query_identity_or_order_invalid")
        seen_queries.add(identity)
        results: list[dict[str, Any]] = []
        seen_results: set[str] = set()
        for expected_rank, result in enumerate(query["results"], start=1):
            if set(result) != {"authority_digest", "rank", "result_identity"}:
                raise ExternalScorerError("handoff_result_schema_invalid")
            result_identity = str(result["result_identity"])
            if not result_identity or result_identity in seen_results or int(result["rank"]) != expected_rank:
                raise ExternalScorerError("handoff_result_identity_or_rank_invalid")
            seen_results.add(result_identity)
            results.append(
                {
                    "authority_digest": str(result["authority_digest"]),
                    "rank": expected_rank,
                    "result_identity": result_identity,
                }
            )
        if len(results) > 20:
            raise ExternalScorerError("handoff_top20_limit_exceeded")
        normalized.append(
            {
                "query_identity": identity,
                "query_order": expected_order,
                "results": results,
            }
        )
    payload: dict[str, Any] = {
        "handoff_version": HANDOFF_VERSION,
        "lineage": {
            "commit_generation_sha256": commit_generation_sha256,
            "run_manifest_sha256": run_manifest_sha256,
        },
        "query_count": len(normalized),
        "queries": normalized,
        "schema_version": SCHEMA_VERSION,
        "scope": source_scope,
    }
    payload["content_sha256"] = sha256_bytes(stable_json_bytes(payload))
    return payload


def validate_handoff(value: Mapping[str, Any], limits: Mapping[str, Any]) -> None:
    expected = {
        "content_sha256",
        "handoff_version",
        "lineage",
        "query_count",
        "queries",
        "schema_version",
        "scope",
    }
    if set(value) != expected or value.get("handoff_version") != HANDOFF_VERSION:
        raise ExternalScorerError("handoff_schema_invalid")
    content = dict(value)
    claimed = str(content.pop("content_sha256"))
    if sha256_bytes(stable_json_bytes(content)) != claimed:
        raise ExternalScorerError("handoff_content_hash_mismatch")
    queries = value.get("queries")
    if not isinstance(queries, list) or int(value.get("query_count", -1)) != len(queries):
        raise ExternalScorerError("handoff_query_count_invalid")
    if len(queries) > int(limits["maximum_queries"]):
        raise ExternalScorerError("handoff_query_limit_exceeded")
    rebuilt = canonical_handoff(
        queries,
        run_manifest_sha256=str(value["lineage"]["run_manifest_sha256"]),
        commit_generation_sha256=str(value["lineage"]["commit_generation_sha256"]),
        source_scope=str(value["scope"]),
    )
    if rebuilt != value:
        raise ExternalScorerError("handoff_not_canonical")


def create_package_manifest(
    package_dir: Path,
    *,
    scorer_name: str,
    scorer_version: str,
    entrypoint_source: str,
) -> dict[str, Any]:
    package_dir.mkdir(parents=True, exist_ok=False)
    entrypoint = package_dir / "scorer.py"
    entrypoint.write_text(entrypoint_source, encoding="utf-8")
    input_schema = _synthetic_input_schema()
    output_schema = _synthetic_output_schema()
    manifest = {
        "allowed_io": {
            "environment_files": False,
            "input": "canonical_handoff_read_only",
            "network": False,
            "output": "isolated_temporary_output_only",
            "subprocess": False,
        },
        "determinism": {"comparison": "byte_identical", "repeat_runs": 2},
        "entrypoint": ["python", "scorer.py"],
        "entrypoint_sha256": sha256_file(entrypoint),
        "input_schema_summary": input_schema,
        "input_schema_sha256": sha256_bytes(stable_json_bytes(input_schema)),
        "manifest_version": PACKAGE_VERSION,
        "metric_namespace": SYNTHETIC_NAMESPACE,
        "output_schema_summary": output_schema,
        "output_schema_sha256": sha256_bytes(stable_json_bytes(output_schema)),
        "package_type": "strict_conformance_fixture_not_official_scorer",
        "resource_limits": {
            "maximum_input_bytes": 1048576,
            "maximum_output_bytes": 1048576,
            "timeout_seconds": 2.0,
        },
        "runtime": {"executable": "isolated_python_worker", "network": False},
        "scorer_name": scorer_name,
        "scorer_version": scorer_version,
    }
    (package_dir / "manifest.json").write_bytes(stable_json_bytes(manifest))
    return manifest


def verify_package(package_dir: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = package_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    required = set(protocol["package_contract"]["required_fields"]) | {"manifest_version"}
    if set(manifest) != required or manifest.get("manifest_version") != PACKAGE_VERSION:
        raise ExternalScorerError("package_manifest_schema_invalid")
    if manifest.get("entrypoint") != ["python", "scorer.py"]:
        raise ExternalScorerError("package_entrypoint_invalid")
    if manifest.get("package_type") != protocol["package_contract"]["synthetic_package_type"]:
        raise ExternalScorerError("package_type_invalid")
    if manifest.get("metric_namespace") != SYNTHETIC_NAMESPACE:
        raise ExternalScorerError("package_metric_namespace_invalid")
    if manifest.get("input_schema_summary") != _synthetic_input_schema() or manifest.get(
        "output_schema_summary"
    ) != _synthetic_output_schema():
        raise ExternalScorerError("package_schema_summary_invalid")
    if manifest.get("input_schema_sha256") != sha256_bytes(
        stable_json_bytes(_synthetic_input_schema())
    ) or manifest.get("output_schema_sha256") != sha256_bytes(
        stable_json_bytes(_synthetic_output_schema())
    ):
        raise ExternalScorerError("package_schema_digest_invalid")
    expected_limits = {
        "maximum_input_bytes": int(protocol["resource_limits"]["maximum_input_bytes"]),
        "maximum_output_bytes": int(protocol["resource_limits"]["maximum_output_bytes"]),
        "timeout_seconds": float(protocol["resource_limits"]["timeout_seconds"]),
    }
    if manifest.get("resource_limits") != expected_limits:
        raise ExternalScorerError("package_resource_limits_invalid")
    if manifest.get("runtime") != {
        "executable": "isolated_python_worker",
        "network": False,
    }:
        raise ExternalScorerError("package_runtime_invalid")
    if manifest.get("allowed_io") != {
        "environment_files": False,
        "input": "canonical_handoff_read_only",
        "network": False,
        "output": "isolated_temporary_output_only",
        "subprocess": False,
    } or manifest.get("determinism") != {
        "comparison": "byte_identical",
        "repeat_runs": 2,
    }:
        raise ExternalScorerError("package_io_or_determinism_invalid")
    if not str(manifest.get("scorer_name") or "") or not str(
        manifest.get("scorer_version") or ""
    ):
        raise ExternalScorerError("package_identity_invalid")
    entrypoint = package_dir / "scorer.py"
    if entrypoint.is_symlink() or not entrypoint.is_file() or sha256_file(entrypoint) != manifest.get("entrypoint_sha256"):
        raise ExternalScorerError("package_entrypoint_hash_mismatch")
    members = sorted(path.name for path in package_dir.iterdir())
    if members != ["manifest.json", "scorer.py"]:
        raise ExternalScorerError("package_unregistered_member")
    return manifest


def _synthetic_input_schema() -> dict[str, Any]:
    return {
        "required": ["query_identity", "query_order", "results"],
        "version": HANDOFF_VERSION,
    }


def _synthetic_output_schema() -> dict[str, Any]:
    return {
        "query_result_required": ["query_identity", "values"],
        "version": OUTPUT_VERSION,
    }


def _worker_environment(home: Path, temp: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "TMPDIR": str(temp),
    }


def run_scorer(
    package_dir: Path,
    handoff_path: Path,
    protocol: Mapping[str, Any],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    run_ordinal: int = 0,
) -> dict[str, Any]:
    manifest = verify_package(package_dir, protocol)
    handoff = _read_json(handoff_path)
    validate_handoff(handoff, protocol["resource_limits"])
    before = sha256_file(handoff_path)
    if handoff_path.stat().st_size > int(protocol["resource_limits"]["maximum_input_bytes"]):
        raise ExternalScorerError("handoff_input_too_large")
    worker = repository_root / "scripts" / "external_scorer_worker.py"
    with tempfile.TemporaryDirectory(prefix="external-scorer-handoff-") as temp_name:
        temp = Path(temp_name)
        home = temp / "home"
        output_dir = temp / "output"
        home.mkdir()
        output_dir.mkdir()
        (home / ".env").write_text(SENSITIVE_MARKER, encoding="utf-8")
        output = output_dir / "scorer-output.json"
        request = temp / "request.json"
        request.write_bytes(
            stable_json_bytes(
                {
                    "entrypoint": str((package_dir / "scorer.py").resolve()),
                    "entrypoint_sha256": manifest["entrypoint_sha256"],
                    "handoff": str(handoff_path.resolve()),
                    "handoff_sha256": before,
                    "output": str(output.resolve()),
                    "output_limit": int(protocol["resource_limits"]["maximum_output_bytes"]),
                    "run_ordinal": int(run_ordinal),
                    "schema_version": SCHEMA_VERSION,
                }
            )
        )
        response = temp / "worker-response.json"
        command = [
            sys.executable,
            str(worker),
            "--request",
            str(request),
            "--response",
            str(response),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=temp,
                env=_worker_environment(home, temp),
                capture_output=True,
                check=False,
                timeout=float(protocol["resource_limits"]["timeout_seconds"]),
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalScorerError("scorer_timeout") from exc
        if SENSITIVE_MARKER in completed.stderr.decode("utf-8", errors="replace"):
            raise ExternalScorerError("sensitive_stderr_echo")
        if sha256_file(handoff_path) != before:
            raise ExternalScorerError("handoff_input_mutated")
        if not response.is_file():
            raise ExternalScorerError("scorer_nonzero_exit")
        worker_report = _read_json(response)
        if completed.returncode != 0:
            raise ExternalScorerError(str(worker_report.get("reason") or "scorer_nonzero_exit"))
        if worker_report.get("status") != "completed":
            raise ExternalScorerError(str(worker_report.get("reason") or "worker_violation"))
        if not output.is_file() or output.stat().st_size > int(protocol["resource_limits"]["maximum_output_bytes"]):
            raise ExternalScorerError("scorer_output_missing_or_too_large")
        raw_output = output.read_bytes()
        parsed = _read_json(output)
        validate_output(parsed, handoff, manifest)
    return {
        "output": parsed,
        "output_bytes": raw_output,
        "output_sha256": sha256_bytes(raw_output),
        "worker_audit": worker_report["audit"],
    }


def validate_output(
    output: Mapping[str, Any], handoff: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    if set(output) != {
        "metric_namespace",
        "query_results",
        "schema_version",
        "scorer_name",
        "scorer_version",
    }:
        raise ExternalScorerError("scorer_output_schema_invalid")
    if output.get("schema_version") != OUTPUT_VERSION:
        raise ExternalScorerError("scorer_output_version_invalid")
    if output.get("scorer_name") != manifest["scorer_name"] or output.get("scorer_version") != manifest["scorer_version"]:
        raise ExternalScorerError("scorer_output_package_binding_mismatch")
    if output.get("metric_namespace") != manifest["metric_namespace"]:
        raise ExternalScorerError("scorer_output_metric_namespace_invalid")
    rows = output.get("query_results")
    if not isinstance(rows, list):
        raise ExternalScorerError("scorer_query_results_invalid")
    expected = [str(item["query_identity"]) for item in handoff["queries"]]
    observed: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"query_identity", "values"}:
            raise ExternalScorerError("scorer_query_result_schema_invalid")
        identity = str(row["query_identity"])
        if identity in observed:
            raise ExternalScorerError("scorer_duplicate_query")
        observed.append(identity)
        values = row["values"]
        if not isinstance(values, dict) or set(values) != {"synthetic_handoff.result_count"}:
            raise ExternalScorerError("scorer_unknown_metric")
        value = values["synthetic_handoff.result_count"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ExternalScorerError("scorer_non_finite_or_invalid_value")
    if observed != expected:
        raise ExternalScorerError("scorer_query_coverage_or_order_mismatch")


def execute_worker_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one preloaded scorer source under field-level I/O hooks."""

    entrypoint = Path(str(request["entrypoint"])).resolve()
    handoff = Path(str(request["handoff"])).resolve()
    output = Path(str(request["output"])).resolve()
    if sha256_file(entrypoint) != request.get("entrypoint_sha256") or sha256_file(handoff) != request.get("handoff_sha256"):
        raise ExternalScorerError("worker_input_hash_mismatch")
    source = entrypoint.read_text(encoding="utf-8")
    payload = _read_json(handoff)
    audit = {
        "blocked_file_operations": 0,
        "blocked_network_operations": 0,
        "blocked_subprocess_operations": 0,
        "input_mutation_count": 0,
    }
    original_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open

    def checked_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        path = Path(file).resolve() if isinstance(file, (str, os.PathLike)) else None
        writing = any(character in mode for character in "wax+")
        if path == handoff and not writing:
            return original_open(file, mode, *args, **kwargs)
        if path == output and writing:
            return original_open(file, mode, *args, **kwargs)
        audit["blocked_file_operations"] += 1
        if path == handoff and writing:
            audit["input_mutation_count"] += 1
        raise PermissionError("scorer_file_access_blocked")

    def checked_os_open(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        path = Path(file).resolve() if isinstance(file, (str, os.PathLike)) else None
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if path == handoff and not writing:
            return original_os_open(file, flags, *args, **kwargs)
        if path == output and writing:
            return original_os_open(file, flags, *args, **kwargs)
        audit["blocked_file_operations"] += 1
        if path == handoff and writing:
            audit["input_mutation_count"] += 1
        raise PermissionError("scorer_file_access_blocked")

    def blocked_network(*_args: Any, **_kwargs: Any) -> Any:
        audit["blocked_network_operations"] += 1
        raise PermissionError("scorer_network_blocked")

    def blocked_subprocess(*_args: Any, **_kwargs: Any) -> Any:
        audit["blocked_subprocess_operations"] += 1
        raise PermissionError("scorer_subprocess_blocked")

    namespace: dict[str, Any] = {
        "HANDOFF_INPUT_PATH": str(handoff),
        "SCORER_RUN_ORDINAL": int(request.get("run_ordinal", 0)),
        "SCORER_OUTPUT_PATH": str(output),
        "__name__": "__external_scorer__",
    }
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(builtins, "open", checked_open))
            stack.enter_context(patch.object(io, "open", checked_open))
            stack.enter_context(patch.object(os, "open", checked_os_open))
            stack.enter_context(patch.object(socket, "socket", blocked_network))
            stack.enter_context(patch.object(socket, "create_connection", blocked_network))
            stack.enter_context(patch.object(socket, "getaddrinfo", blocked_network))
            stack.enter_context(patch.object(subprocess, "run", blocked_subprocess))
            stack.enter_context(patch.object(subprocess, "Popen", blocked_subprocess))
            for name in (
                "fork",
                "posix_spawn",
                "posix_spawnp",
                "spawnl",
                "spawnle",
                "spawnlp",
                "spawnlpe",
                "spawnv",
                "spawnve",
                "spawnvp",
                "spawnvpe",
                "system",
            ):
                if hasattr(os, name):
                    stack.enter_context(patch.object(os, name, blocked_subprocess))
            exec(compile(source, "<registered-scorer>", "exec"), namespace, namespace)
            score = namespace.get("score")
            if not callable(score):
                raise ExternalScorerError("scorer_entry_function_missing")
            result = score(payload)
            output.write_bytes(stable_json_bytes(result))
    except Exception as exc:  # noqa: BLE001 - worker maps all scorer failures to codes
        reason = str(exc)
        if reason not in {
            "scorer_file_access_blocked",
            "scorer_network_blocked",
            "scorer_subprocess_blocked",
            "scorer_entry_function_missing",
            "synthetic_crash",
        }:
            reason = "scorer_execution_failed"
        return {"audit": audit, "reason": reason, "status": "violation"}
    return {"audit": audit, "reason": None, "status": "completed"}


def synthetic_handoff() -> dict[str, Any]:
    queries = []
    for query_order, result_count in enumerate((2, 1, 0)):
        query_identity = hashlib.sha256(f"synthetic-query-{query_order}".encode()).hexdigest()
        results = [
            {
                "authority_digest": hashlib.sha256(f"authority-{query_order}-{rank}".encode()).hexdigest(),
                "rank": rank,
                "result_identity": hashlib.sha256(f"result-{query_order}-{rank}".encode()).hexdigest(),
            }
            for rank in range(1, result_count + 1)
        ]
        queries.append({"query_identity": query_identity, "query_order": query_order, "results": results})
    return canonical_handoff(
        queries,
        run_manifest_sha256=hashlib.sha256(b"synthetic-run-manifest").hexdigest(),
        commit_generation_sha256=hashlib.sha256(b"synthetic-generation").hexdigest(),
    )


def synthetic_scorer_source(scenario: str) -> str:
    prelude = (
        "def _base(payload):\n"
        "    return {'schema_version':'synthetic_scorer_output_v1','scorer_name':'synthetic-strict-scorer','scorer_version':'1','metric_namespace':'synthetic_handoff','query_results':[{'query_identity':q['query_identity'],'values':{'synthetic_handoff.result_count':len(q['results'])}} for q in payload['queries']]}\n"
    )
    bodies = {
        "valid": "def score(payload):\n    return _base(payload)\n",
        "input_tamper": "def score(payload):\n    open(HANDOFF_INPUT_PATH,'w').write('tamper')\n    return _base(payload)\n",
        "network_attempt": "def score(payload):\n    import socket\n    socket.socket()\n    return _base(payload)\n",
        "dotenv_read": "def score(payload):\n    open('.env').read()\n    return _base(payload)\n",
        "outside_write": "def score(payload):\n    open('/tmp/unregistered-scorer-output','w').write('x')\n    return _base(payload)\n",
        "subprocess_attempt": "def score(payload):\n    import subprocess\n    subprocess.run(['false'])\n    return _base(payload)\n",
        "missing_query": "def score(payload):\n    out=_base(payload); out['query_results']=out['query_results'][:-1]; return out\n",
        "duplicate_query": "def score(payload):\n    out=_base(payload); out['query_results'].append(out['query_results'][0]); return out\n",
        "illegal_schema": "def score(payload):\n    out=_base(payload); out['extra']='forbidden'; return out\n",
        "nondeterministic_output": "def score(payload):\n    out=_base(payload); out['query_results'][0]['values']['synthetic_handoff.result_count']=SCORER_RUN_ORDINAL; return out\n",
        "timeout": "def score(payload):\n    while True: pass\n",
        "crash": "def score(payload):\n    raise RuntimeError('synthetic_crash')\n",
        "partial_write": "def score(payload):\n    open(SCORER_OUTPUT_PATH,'w').write('{')\n    raise RuntimeError('synthetic_crash')\n",
        "extra_metric": "def score(payload):\n    out=_base(payload); out['query_results'][0]['values']['forged.metric']=1; return out\n",
        "sensitive_stderr": "def score(payload):\n    import sys\n    print('SYNTHETIC_SECRET_SENTINEL', file=sys.stderr)\n    return _base(payload)\n",
    }
    if scenario not in bodies:
        raise ExternalScorerError("unknown_synthetic_scenario")
    return prelude + bodies[scenario]


def run_synthetic_matrix(protocol: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="external-scorer-matrix-") as temp_name:
        root = Path(temp_name)
        handoff = synthetic_handoff()
        handoff_path = root / "handoff.json"
        handoff_path.write_bytes(stable_json_bytes(handoff))
        for scenario in protocol["synthetic_matrix"]:
            package = root / f"package-{scenario}"
            create_package_manifest(
                package,
                scorer_name="synthetic-strict-scorer",
                scorer_version="1",
                entrypoint_source=synthetic_scorer_source(str(scenario)),
            )
            expected = "passed" if scenario == "valid" else "rejected"
            observed = "passed"
            reason = None
            output_sha256 = None
            try:
                first = run_scorer(package, handoff_path, protocol, repository_root=repository_root, run_ordinal=1)
                second = run_scorer(package, handoff_path, protocol, repository_root=repository_root, run_ordinal=2)
                output_sha256 = first["output_sha256"]
                if first["output_bytes"] != second["output_bytes"]:
                    raise ExternalScorerError("scorer_output_nondeterministic")
            except ExternalScorerError as exc:
                observed = "rejected"
                reason = str(exc)
            rows.append(
                {
                    "expected": expected,
                    "observed": observed,
                    "output_sha256": output_sha256,
                    "reason": reason,
                    "scenario": scenario,
                }
            )
    if any(row["expected"] != row["observed"] for row in rows):
        raise ExternalScorerError("synthetic_matrix_expectation_mismatch")
    return {
        "analysis": CONTRACT_VERSION,
        "execution": {
            "gold_or_qrels_loaded": False,
            "llm_request_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "snapshot_write_count": 0,
        },
        "exit_code": EXIT_VERIFIED,
        "formal_validation_complete": False,
        "official_score_generated": False,
        "scenario_count": len(rows),
        "scenarios": rows,
        "schema_version": SCHEMA_VERSION,
        "status": "handoff_chain_verified",
    }


def audit_real_readiness(protocol: Mapping[str, Any], *, repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    resume_spec = protocol["frozen_readiness_inputs"]["full1000_resume"]
    resume = _read_json(_repo_file(root, resume_spec["path"]))
    record_manifest = _read_json(
        _repo_file(root, protocol["frozen_readiness_inputs"]["record160_delivery_manifest"]["path"])
    )
    missing = [
        "official_scorer_package",
        "official_input_schema",
        "official_output_schema",
        "official_metric_namespace",
    ]
    if int(resume.get("record_terminal_case_count", 0)) < 1000:
        missing.append("complete_full1000_authoritative_input")
    if "run_manifest_sha256" not in record_manifest:
        missing.append("record160_run_manifest_binding")
    if "commit_generation_sha256" not in record_manifest:
        missing.append("record160_commit_generation_binding")
    return {
        "analysis": CONTRACT_VERSION,
        "blocked_reasons": sorted(missing),
        "execution": protocol["execution"],
        "exit_code": EXIT_BLOCKED,
        "formal_validation_complete": False,
        "official_package": protocol["official_package"],
        "official_score_generated": False,
        "record160_scope": "internal_rehearsal_only_not_complete_input",
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_missing_official_scorer_or_complete_input",
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stable_json_bytes(value))
