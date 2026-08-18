from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scholar_agent.evaluation.run_provenance import (
    EXIT_INTEGRITY_FAILURE,
    EXIT_LEGACY_METADATA_INCOMPLETE,
    GitProvenance,
    RunProvenanceError,
    audit_legacy_profiles,
    build_run_manifest,
    classify_worktree_paths,
    sha256_file,
    stable_hash,
    validate_run_manifest,
    write_json,
)
from scholar_agent.evaluation.resource_accounting import (
    _fixture_run_ledger,
    build_run_ledger,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _git(*, submodule_dirty: bool = False) -> GitProvenance:
    dirty_paths = ["third_party/paper-qa"] if submodule_dirty else []
    payload = {
        "commit": "a" * 40,
        "dirty_paths": dirty_paths,
        "allowed_dirty_paths": dirty_paths,
        "unexpected_dirty_paths": [],
    }
    return GitProvenance(
        **payload,
        dirty=bool(dirty_paths),
        worktree_state_sha256=stable_hash(payload),
    )


def _fixture(
    root: Path,
    *,
    run_name: str = "run",
    result_rows: int = 2,
    status: str = "completed",
    completed_count: int | None = None,
) -> tuple[dict[str, object], Path]:
    queries = [
        {"query_id": "q-α", "query": "Unicode query α"},
        {"query_id": "q-2", "query": "second query"},
    ]
    _write_jsonl(root / "data/queries.jsonl", queries)
    (root / "data/source.txt").write_text("dataset-v1\n", encoding="utf-8")
    _write_json(root / "prompts/manifest.json", {"planner": "planner-v1"})
    run_dir = root / "runs" / run_name
    config = {
        "dataset": {"name": "fixture", "version": "v1"},
        "prompt_versions": {"planner": "planner-v1"},
        "sources": ["arxiv", "openalex"],
        "budgets": {"max_queries": 2, "top_k": 20},
        "evaluator": {"name": "internal_retrieval", "version": "v2"},
    }
    _write_json(run_dir / "config.json", config)
    _write_jsonl(
        run_dir / "results.jsonl",
        [{"query_id": f"q-{index}"} for index in range(result_rows)],
    )
    bindings = [
        ("/dataset/name", "/dataset/name"),
        ("/dataset/version", "/dataset/version"),
        ("/prompt_versions", "/prompt/versions"),
        ("/sources", "/configuration/sources"),
        ("/budgets", "/configuration/budgets"),
        ("/evaluator/name", "/evaluator/name"),
        ("/evaluator/version", "/evaluator/version"),
    ]
    spec: dict[str, object] = {
        "run_id": run_name,
        "dataset": {
            "name": "fixture",
            "version": "v1",
            "input_paths": ["data/source.txt"],
        },
        "queries": {
            "input_path": "data/queries.jsonl",
            "id_field": "query_id",
            "text_field": "query",
        },
        "prompt": {
            "manifest_path": "prompts/manifest.json",
            "versions": {"planner": "planner-v1"},
            "used": True,
        },
        "configuration": {
            "sources": ["arxiv", "openalex"],
            "budgets": {"max_queries": 2, "top_k": 20},
            "values": {"mode": "replay", "experimental_flags": False},
        },
        "evaluator": {"name": "internal_retrieval", "version": "v2"},
        "determinism": {
            "random_seed": 7,
            "parameters": {"sort": "stable", "temperature": 0},
        },
        "progress": {
            "status": status,
            "expected_count": 2,
            "completed_count": result_rows
            if completed_count is None
            else completed_count,
            "record_output_path": f"runs/{run_name}/results.jsonl",
        },
        "lineage": {"checkpoint_id": f"{run_name}-cp", "resume_index": 0},
        "output_directory": f"runs/{run_name}",
        "outputs": [
            {
                "path": f"runs/{run_name}/results.jsonl",
                "role": "query_results",
                "format": "jsonl",
            },
            {
                "path": f"runs/{run_name}/config.json",
                "role": "run_config",
                "format": "json",
            },
        ],
        "metadata_bindings": [
            {
                "artifact_path": f"runs/{run_name}/config.json",
                "artifact_json_pointer": artifact,
                "manifest_json_pointer": manifest,
            }
            for artifact, manifest in bindings
        ],
    }
    return spec, run_dir


def _materialize(root: Path, spec: dict[str, object], name: str) -> Path:
    manifest = build_run_manifest(spec, repository_root=root, git_provenance=_git())
    path = root / "manifests" / f"{name}.json"
    write_json(path, manifest.model_dump(mode="json"))
    return path


def _kinds(report: dict[str, object]) -> set[str]:
    return {item["kind"] for item in report["violations"]}  # type: ignore[index]


def test_run_manifest_passes_and_is_byte_deterministic(tmp_path: Path) -> None:
    spec, _ = _fixture(tmp_path)
    reversed_spec = deepcopy(spec)
    reversed_spec["outputs"] = list(reversed(reversed_spec["outputs"]))  # type: ignore[index]
    reversed_spec["metadata_bindings"] = list(  # type: ignore[index]
        reversed(reversed_spec["metadata_bindings"])  # type: ignore[index]
    )
    first = build_run_manifest(spec, repository_root=tmp_path, git_provenance=_git())
    second = build_run_manifest(
        reversed_spec, repository_root=tmp_path, git_provenance=_git()
    )
    assert first == second
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(first_path, first.model_dump(mode="json"))
    write_json(second_path, second.model_dump(mode="json"))
    assert first_path.read_bytes() == second_path.read_bytes()
    assert validate_run_manifest(first_path, repository_root=tmp_path)["status"] == "passed"


def test_run_manifest_registers_and_validates_authoritative_resource_ledger(
    tmp_path: Path,
) -> None:
    spec, run_dir = _fixture(tmp_path, run_name="offline-resource-fixture")
    fixture = _fixture_run_ledger(shard_resume=False)
    queries = fixture.queries[:2]
    identities = [item.query_identity for item in queries]
    ledger = build_run_ledger(
        queries,
        run_identity=fixture.run_identity,
        manifest_identity=fixture.manifest_identity,
        expected_query_identities=identities,
        selected_attempts={
            item.query_identity: item.attempt_identity for item in queries
        },
    )
    ledger_path = run_dir / "resource_ledger.json"
    _write_json(ledger_path, ledger.model_dump(mode="json"))
    spec["outputs"].append(  # type: ignore[union-attr]
        {
            "path": "runs/offline-resource-fixture/resource_ledger.json",
            "role": "resource_ledger_v1",
            "format": "json",
        }
    )
    spec["resource_ledger"] = {
        "output_path": "runs/offline-resource-fixture/resource_ledger.json",
        "manifest_identity": ledger.manifest_identity,
    }

    manifest_path = _materialize(tmp_path, spec, "resource-ledger")
    report = validate_run_manifest(manifest_path, repository_root=tmp_path)

    assert report["status"] == "passed"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["resource_ledger"]["authority"] == "committed_generation_only"
    assert manifest["resource_ledger"]["output"]["sha256"] == sha256_file(
        ledger_path
    )


def test_tampered_missing_and_unregistered_files_are_reported(tmp_path: Path) -> None:
    spec, run_dir = _fixture(tmp_path)
    manifest_path = _materialize(tmp_path, spec, "run")
    (run_dir / "results.jsonl").write_text("tampered\n", encoding="utf-8")
    (run_dir / "config.json").unlink()
    (run_dir / "extra.txt").write_text("unregistered", encoding="utf-8")
    kinds = _kinds(validate_run_manifest(manifest_path, repository_root=tmp_path))
    assert {"file_tampered", "file_missing", "unregistered_output_file"} <= kinds


def test_query_reordering_and_metadata_drift_are_reported(tmp_path: Path) -> None:
    spec, run_dir = _fixture(tmp_path)
    manifest_path = _materialize(tmp_path, spec, "run")
    rows = [json.loads(line) for line in (tmp_path / "data/queries.jsonl").read_text().splitlines()]
    _write_jsonl(tmp_path / "data/queries.jsonl", list(reversed(rows)))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config["budgets"]["top_k"] = 10
    _write_json(run_dir / "config.json", config)
    kinds = _kinds(validate_run_manifest(manifest_path, repository_root=tmp_path))
    assert "query_order_sha256_mismatch" in kinds
    assert "metadata_binding_mismatch" in kinds


def test_completed_claim_with_partial_records_fails(tmp_path: Path) -> None:
    spec, _ = _fixture(tmp_path, result_rows=1, status="partial")
    manifest_path = _materialize(tmp_path, spec, "partial")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["progress"]["status"] = "completed"
    write_json(manifest_path, payload)
    report = validate_run_manifest(manifest_path, repository_root=tmp_path)
    assert report["exit_code"] == EXIT_INTEGRITY_FAILURE
    assert "completed_run_record_count_insufficient" in _kinds(report)


def test_partial_checkpoint_and_legal_resume_lineage(tmp_path: Path) -> None:
    parent_spec, _ = _fixture(
        tmp_path, run_name="parent", result_rows=1, status="partial"
    )
    parent_path = _materialize(tmp_path, parent_spec, "parent")
    child_spec, _ = _fixture(tmp_path, run_name="child")
    child_spec["lineage"] = {
        "checkpoint_id": "child-cp",
        "resume_index": 1,
        "parent": {
            "manifest_path": "manifests/parent.json",
            "manifest_sha256": sha256_file(parent_path),
            "run_id": "parent",
            "checkpoint_id": "parent-cp",
        },
    }
    child_path = _materialize(tmp_path, child_spec, "child")
    assert validate_run_manifest(child_path, repository_root=tmp_path)["status"] == "passed"


def test_broken_and_cyclic_lineage_are_reported(tmp_path: Path) -> None:
    spec, _ = _fixture(tmp_path)
    manifest_path = _materialize(tmp_path, spec, "run")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["lineage"] = {
        "checkpoint_id": "run-cp",
        "resume_index": 1,
        "parent": {
            "manifest_path": "manifests/run.json",
            "manifest_sha256": "0" * 64,
            "run_id": "run",
            "checkpoint_id": "run-cp",
        },
    }
    write_json(manifest_path, payload)
    kinds = _kinds(validate_run_manifest(manifest_path, repository_root=tmp_path))
    assert "lineage_parent_hash_mismatch" in kinds
    assert "lineage_cycle" in kinds

    payload["lineage"]["parent"]["manifest_path"] = "manifests/missing.json"
    write_json(manifest_path, payload)
    assert "lineage_parent_missing" in _kinds(
        validate_run_manifest(manifest_path, repository_root=tmp_path)
    )


def test_schema_mismatch_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_json(path, {"manifest_kind": "run_manifest_v1", "schema_version": "2"})
    assert "schema_version_incompatible" in _kinds(
        validate_run_manifest(path, repository_root=tmp_path)
    )


def test_legacy_profile_never_guesses_missing_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "legacy/results.jsonl"
    _write_jsonl(artifact, [{"query_id": "q-1"}])
    profile = tmp_path / "legacy_profiles.json"
    _write_json(
        profile,
        {
            "schema_version": "1",
            "profiles": [
                {
                    "profile_id": "legacy",
                    "expected_query_count": 2,
                    "results_path": "legacy/results.jsonl",
                    "missing_run_manifest_v1_fields": [
                        "queries.order_sha256",
                        "evaluator.version",
                    ],
                    "frozen_files": [
                        {
                            "path": "legacy/results.jsonl",
                            "sha256": sha256_file(artifact),
                        }
                    ],
                }
            ],
        },
    )
    first = audit_legacy_profiles(profile, repository_root=tmp_path)
    second = audit_legacy_profiles(profile, repository_root=tmp_path)
    assert first == second
    assert first["status"] == "legacy_metadata_incomplete"
    assert first["exit_code"] == EXIT_LEGACY_METADATA_INCOMPLETE
    assert first["profiles"][0]["record_is_complete"] is False


def test_existing_submodule_state_is_allowed_and_audited() -> None:
    dirty, allowed, unexpected = classify_worktree_paths(
        [" m third_party/paper-qa", "?? local.tmp"], ["third_party/paper-qa"]
    )
    assert dirty == ["local.tmp", "third_party/paper-qa"]
    assert allowed == ["third_party/paper-qa"]
    assert unexpected == ["local.tmp"]


def test_required_metadata_bindings_cannot_be_omitted(tmp_path: Path) -> None:
    spec, _ = _fixture(tmp_path)
    spec["metadata_bindings"] = []
    with pytest.raises(ValueError, match="required metadata bindings"):
        build_run_manifest(spec, repository_root=tmp_path, git_provenance=_git())


@pytest.mark.run_provenance_regression
def test_run_provenance_regression_gate_is_offline(tmp_path: Path) -> None:
    spec, _ = _fixture(tmp_path)
    manifest_path = _materialize(tmp_path, spec, "gate")
    first = validate_run_manifest(manifest_path, repository_root=tmp_path)
    second = validate_run_manifest(manifest_path, repository_root=tmp_path)
    assert first == second
    assert first["status"] == "passed"
    assert first["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "gold_fields_accessed": False,
    }
