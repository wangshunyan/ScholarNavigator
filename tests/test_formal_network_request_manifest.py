from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_agent.evaluation import formal_network_request_manifest as module


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/formal_network_request_manifest_v1_protocol.json"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return module.load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.fixture(scope="module")
def built(
    protocol: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return module.build_request_manifest(ROOT, protocol)


def test_full1000_request_population_closes_against_frozen_plan(
    built: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    intents, manifest = built

    assert manifest["query_count"] == 1000
    assert manifest["subquery_count"] == 2410
    assert manifest["logical_source_request_count"] == 9640
    assert manifest["http_attempt_upper"] == 19280
    assert manifest["source_counts"] == {
        "arxiv": 2410,
        "openalex": 2410,
        "pubmed": 2410,
        "semantic_scholar": 2410,
    }
    assert len(intents) == 9640
    assert len({row["intent_id"] for row in intents}) == 9640
    assert len({row["query_identity"] for row in intents}) == 1000
    assert sum(manifest["shard_counts"].values()) == 9640  # type: ignore[union-attr]


def test_every_query_subquery_source_slot_is_unique_and_on_one_shard(
    built: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    intents, _manifest = built
    slots = [
        (row["query_identity"], row["subquery_index"], row["source"])
        for row in intents
    ]
    assert len(slots) == len(set(slots))
    for row in intents:
        assert row["shard_index"] == row["query_order"] % 20


def test_parameter_order_is_canonical() -> None:
    left = module._parameter_evidence({"z": "3", "a": "1", "m": "2"})
    right = module._parameter_evidence({"m": "2", "z": "3", "a": "1"})

    assert left == right


def test_auth_context_is_explicitly_bound_without_credentials(
    built: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    intents, _manifest = built
    scopes = {
        row["source"]: row["request_spec"]["auth_scope_alias"]  # type: ignore[index]
        for row in intents
    }

    assert scopes == {
        "arxiv": "public_anonymous",
        "openalex": "openalex_polite_pool_optional",
        "pubmed": "ncbi_api_key_optional",
        "semantic_scholar": "semantic_scholar_api_key_optional",
    }
    assert all(
        row["request_spec"]["auth_affects_response_semantics"] is False  # type: ignore[index]
        for row in intents
    )


def test_pubmed_response_dependent_child_is_not_materialized(
    built: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    intents, _manifest = built
    pubmed = next(row for row in intents if row["source"] == "pubmed")
    spec = pubmed["request_spec"]
    children = spec["response_dependent_children"]  # type: ignore[index]

    assert spec["pagination_strategy"] == (  # type: ignore[index]
        "fixed_esearch_then_response_dependent_fetch"
    )
    assert children == [
        {
            "accept": "application/xml",
            "cursor_or_ids": "not_materialized_before_response",
            "endpoint_alias": "pubmed_efetch",
            "max_retries": 0,
            "method": "GET",
            "parameter_rule": "response_order_pmids_joined_by_comma",
            "parent_field": "esearchresult.idlist",
            "response_media_type": "application/xml",
        }
    ]
    assert pubmed["http_attempt_upper"] == 2


def test_cache_collision_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    protocol: dict[str, object],
) -> None:
    monkeypatch.setattr(
        module,
        "retrieval_cache_identity",
        lambda _source, _query, _limit: ("same", "same", 20),
    )

    with pytest.raises(
        module.NetworkRequestManifestError, match="semantic_cache_collision"
    ):
        module.build_request_manifest(ROOT, protocol)


def test_full1000_configuration_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    protocol: dict[str, object],
) -> None:
    original = module.load_json

    def changed(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "full1000_execution_plan_v1.json":
            value = copy.deepcopy(value)
            value["execution_contract"]["sources"] = list(  # type: ignore[index]
                reversed(value["execution_contract"]["sources"])  # type: ignore[index]
            )
        return value

    monkeypatch.setattr(module, "load_json", changed)
    with pytest.raises(
        module.NetworkRequestManifestError, match="execution_source_order_drift"
    ):
        module.build_request_manifest(ROOT, protocol)


def test_snapshot_binding_conflict_is_rejected(
    built: tuple[list[dict[str, object]], dict[str, object]],
    protocol: dict[str, object],
) -> None:
    intents, _manifest = built
    changed = copy.deepcopy(intents)
    historical, _evidence = module._historical_snapshot_keys(ROOT, protocol)
    frozen_protocol = module.load_json(
        ROOT / "benchmark/source_reliability_diagnostics_v1_protocol.json"
    )
    store = module.SnapshotStore(
        ROOT / frozen_protocol["frozen_input"]["snapshot_dir"]
    )
    foreign_key = next(
        key
        for key in sorted(historical)
        if store.read_retrieval(key).source != changed[0]["source"]
    )
    changed[0]["snapshot_key"] = foreign_key

    with pytest.raises(
        module.NetworkRequestManifestError,
        match="historical_snapshot_binding_conflict",
    ):
        module.audit_snapshots(ROOT, protocol, changed)


def test_historical_snapshot_gap_is_closed_without_completion_claim(
    built: tuple[list[dict[str, object]], dict[str, object]],
    protocol: dict[str, object],
) -> None:
    intents, _manifest = built
    report = module.audit_snapshots(ROOT, protocol, intents)

    assert report["historical_key_count"] == 1093
    assert report["exact_match_count"] == 816
    assert report["historical_orphan_count"] == 277
    assert report["planned_uncovered_count"] == 8784
    assert report["conflict_count"] == 0
    assert report["authority"] == (
        "historical_reference_only_not_checkpoint_or_completion"
    )


def test_sensitive_or_raw_request_fields_are_rejected() -> None:
    with pytest.raises(
        module.NetworkRequestManifestError, match="forbidden_output_field"
    ):
        module._assert_safe_output({"query": "do not emit"})
    with pytest.raises(
        module.NetworkRequestManifestError, match="sensitive_output_value"
    ):
        module._assert_safe_output({"value": "Bearer sentinel"})


def test_manifest_emits_hashes_not_query_or_credentials(
    built: tuple[list[dict[str, object]], dict[str, object]],
) -> None:
    intents, manifest = built
    encoded = module.canonical_jsonl(intents) + module.canonical_json(manifest)

    assert all("query" not in row for row in intents)
    assert all(
        set(row["request_spec"]["parameter_evidence"]["value_sha256"].values())  # type: ignore[index,union-attr]
        and all(
            module.HEX64_RE.fullmatch(value)
            for value in row["request_spec"]["parameter_evidence"][  # type: ignore[index,union-attr]
                "value_sha256"
            ].values()
        )
        for row in intents
    )
    assert b'"headers":' not in encoded
    assert b'"credential":' not in encoded
    assert b'"request_url":' not in encoded
    assert b'"adapted_query_sha256":' in encoded


def test_bundle_is_byte_deterministic_and_rebuild_verifiable(
    tmp_path: Path,
    built: tuple[list[dict[str, object]], dict[str, object]],
    protocol: dict[str, object],
) -> None:
    intents, manifest = built
    audit = module.audit_snapshots(ROOT, protocol, intents)
    addendum = module.build_launch_addendum(protocol, manifest, audit)
    first = tmp_path / "first"
    second = tmp_path / "second"
    module.write_bundle(first, intents, manifest, audit, addendum)
    module.write_bundle(second, intents, manifest, audit, addendum)

    first_files = {
        path.name: path.read_bytes() for path in sorted(first.iterdir())
    }
    second_files = {
        path.name: path.read_bytes() for path in sorted(second.iterdir())
    }
    assert first_files == second_files
    assert module.verify_bundle(
        first, protocol, repository_root=ROOT
    )["status"] == "request_manifest_ready_network_blocked"


def test_resealed_duplicate_intent_bundle_is_rejected(
    tmp_path: Path,
    built: tuple[list[dict[str, object]], dict[str, object]],
    protocol: dict[str, object],
) -> None:
    intents, manifest = built
    audit = module.audit_snapshots(ROOT, protocol, intents)
    addendum = module.build_launch_addendum(protocol, manifest, audit)
    output = tmp_path / "bundle"
    module.write_bundle(output, intents, manifest, audit, addendum)
    intent_path = output / "intents.jsonl"
    content = intent_path.read_bytes()
    first_line = content.splitlines(keepends=True)[0]
    changed = content + first_line
    intent_path.write_bytes(changed)
    bundle = module.load_json(output / "bundle.json")
    bundle["files"]["intents.jsonl"] = {
        "size": len(changed),
        "sha256": hashlib.sha256(changed).hexdigest(),
    }
    bundle["bundle_sha256"] = module.stable_hash(bundle["files"])
    module.write_json(output / "bundle.json", bundle)

    with pytest.raises(
        module.NetworkRequestManifestError,
        match="request_manifest_rebuild_drift",
    ):
        module.verify_bundle(output, protocol, repository_root=ROOT)


def test_missing_request_metadata_returns_exit_three(tmp_path: Path) -> None:
    protocol = module.load_json(PROTOCOL_PATH)
    protocol["inputs"]["query_input"] = "benchmark/missing-query-input.jsonl"
    protocol["input_hashes"]["benchmark/missing-query-input.jsonl"] = "0" * 64
    path = tmp_path / "protocol.json"
    module.write_json(path, protocol)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_formal_network_request_manifest.py",
            "--protocol",
            str(path),
            "audit-readiness",
        ],
        cwd=ROOT,
        env={"PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 3
    assert completed.stderr == ""
    assert report["status"] == "not_ready_missing_request_metadata"


def test_cli_readiness_is_deterministic_and_network_blocked() -> None:
    command = [
        sys.executable,
        "scripts/check_formal_network_request_manifest.py",
        "audit-readiness",
    ]
    first = subprocess.run(
        command,
        cwd=ROOT,
        env={"PYTHONPATH": "src"},
        capture_output=True,
        check=False,
    )
    second = subprocess.run(
        command,
        cwd=ROOT,
        env={"PYTHONPATH": "src"},
        capture_output=True,
        check=False,
    )

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == b""
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["status"] == "request_manifest_ready_network_blocked"
    assert report["execution"] == module.EXECUTION_ZERO


def test_release_gates_register_request_manifest_without_clearing_blockers() -> None:
    readiness = module.load_json(
        ROOT / "benchmark/validation_readiness_bundle_v1_contract.json"
    )
    public = module.load_json(
        ROOT / "benchmark/public_contract_compatibility_v1_protocol.json"
    )
    addenda = module.load_json(
        ROOT / "benchmark/validation_evidence_freshness_v1_addenda.json"
    )

    claim = next(
        row
        for row in readiness["claims"]
        if row["claim_id"]
        == "architecture_formal_network_request_manifest_ready"
    )
    assert claim["status"] == "verified"
    assert len(readiness["blockers"]) == 3
    assert readiness["release"]["status"] == "ready_with_declared_blockers"
    assert "formal_network_request_manifest" in public["artifact_contracts"]
    assert "formal_network_request_manifest" in public["cli_contracts"]
    assert addenda["claim_component_bindings"][claim["claim_id"]] == [
        "formal_network_request_manifest"
    ]
