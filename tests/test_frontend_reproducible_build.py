from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.frontend_reproducible_build import (
    PROTOCOL,
    diagnose_archive_pair,
    verify_evidence,
    verify_runtime_archive,
)
from scholar_agent.evaluation.release_candidate_reproducibility import (
    _tar_bytes,
    build_frontend,
    canonical_json,
    materialize_source,
    sha256_bytes,
    stable_digest,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/frontend_reproducible_build_v1_protocol.json"
CONTRACT_PATH = ROOT / "benchmark/frontend_reproducible_build_v1_release_contract.json"


@pytest.fixture(scope="module")
def protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _runtime_archive(path: Path, contract: dict[str, object], *, build_id: bytes | None = None) -> None:
    expected = f"spar-release-{contract['source_commit'][:20]}".encode()
    files = {
        "frontend/BUILD_ID": build_id or expected,
        "frontend/build-manifest.json": canonical_json({"rootMainFiles": ["static/chunks/app.js"]}),
        "frontend/routes-manifest.json": canonical_json({"staticRoutes": [{"page": "/"}]}),
        "frontend/server/app/index.html": (
            b'<script src="/_next/static/chunks/app.js"></script>'
            b'<script>self.__next_f.push([1,"ok"])</script>'
        ),
        "frontend/server/app/index.rsc": b"rsc",
        "frontend/static/chunks/app.js": b"runtime",
    }
    path.write_bytes(_tar_bytes(files, contract))


def test_protocol_freezes_source_build_id_and_canonical_staging(
    protocol: dict[str, object], contract: dict[str, object]
) -> None:
    assert protocol["protocol"] == PROTOCOL
    assert contract["source_commit"] == "b0667d658ed8b8df5bc7f2ffde7625ba0aed0d19"
    assert contract["frontend_canonical_staging"]["strategy"] == "posix_tmp_fixed_source_digest_v1"
    assert contract["frontend_reproducible_build_protocol_sha256"] == stable_digest(protocol)


def test_runtime_contract_accepts_routes_assets_hydration_and_api_type(
    tmp_path: Path, contract: dict[str, object]
) -> None:
    archive = tmp_path / "frontend.tar.gz"
    _runtime_archive(archive, contract)
    report = verify_runtime_archive(archive, contract)
    assert report["passed"] is True
    assert report["hydration_payload_present"] is True
    assert report["api_type_contract_sha256"]


def test_random_build_id_and_missing_runtime_asset_are_rejected(
    tmp_path: Path, contract: dict[str, object]
) -> None:
    archive = tmp_path / "frontend.tar.gz"
    _runtime_archive(archive, contract, build_id=b"random-id")
    report = verify_runtime_archive(archive, contract)
    assert report["passed"] is False
    assert {item["invariant"] for item in report["violations"]} == {"stable_build_id"}

    files = {
        "frontend/BUILD_ID": f"spar-release-{contract['source_commit'][:20]}".encode(),
        "frontend/build-manifest.json": canonical_json({"rootMainFiles": ["static/chunks/missing.js"]}),
        "frontend/routes-manifest.json": canonical_json({"staticRoutes": [{"page": "/"}]}),
        "frontend/server/app/index.html": b"self.__next_f.push([])",
        "frontend/server/app/index.rsc": b"rsc",
    }
    archive.write_bytes(_tar_bytes(files, contract))
    report = verify_runtime_archive(archive, contract)
    assert any(item["invariant"] == "manifest_static_reference" for item in report["violations"])


def test_byte_diagnosis_detects_tamper_absolute_path_and_member_drift(
    tmp_path: Path, contract: dict[str, object]
) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first.write_bytes(_tar_bytes({"frontend/BUILD_ID": b"same", "frontend/static/chunks/a.js": b"ok"}, contract))
    second.write_bytes(
        _tar_bytes(
            {"frontend/BUILD_ID": b"same", "frontend/static/chunks/b.js": b"/home/sentinel/source"},
            contract,
        )
    )
    report = diagnose_archive_pair(first, second)
    assert report["archive_sha256_equal"] is False
    assert report["build_id_equal"] is True
    assert report["member_sets_equal"] is False
    assert report["forbidden_literal_counts"]["/home/"] == 1


def test_evidence_rejects_protocol_drift_and_is_byte_deterministic(
    protocol: dict[str, object], contract: dict[str, object]
) -> None:
    members = [{"path": "frontend/BUILD_ID", "size": 1, "sha256": "a" * 64}]
    runtime = {
        "passed": True,
        "member_count": 1,
        "member_tree_sha256": stable_digest(members),
        "members": members,
    }
    evidence = {
        "protocol": PROTOCOL,
        "protocol_sha256": stable_digest(protocol),
        "release_contract_sha256": stable_digest(contract),
        "source_commit": contract["source_commit"],
        "status": "qualified",
        "root_cause": {"canonical_staging_control_passed": True},
        "fixed_pair": {"archive_sha256_equal": True, "differing_member_count": 0},
        "frontend_archive": {
            "member_count": 1,
            "member_tree_sha256": stable_digest(members),
            "members": members,
        },
        "runtime_fidelity": [copy.deepcopy(runtime), copy.deepcopy(runtime)],
        "release_candidate": {
            "frontend_qualified": True,
            "status": "build_or_supply_chain_violation",
        },
    }
    first = verify_evidence(evidence, protocol, contract)
    second = verify_evidence(copy.deepcopy(evidence), protocol, contract)
    assert first["exit_code"] == 0
    assert canonical_json(first) == canonical_json(second)
    evidence["protocol_sha256"] = "0" * 64
    assert verify_evidence(evidence, protocol, contract)["exit_code"] == 2

    evidence["protocol_sha256"] = stable_digest(protocol)
    evidence["frontend_archive"]["members"][0]["size"] = 2
    assert verify_evidence(evidence, protocol, contract)["exit_code"] == 2


def test_real_cross_parent_empty_cache_build_is_byte_identical(
    tmp_path: Path, contract: dict[str, object]
) -> None:
    outputs = []
    for name in ("first", "different-parent/second"):
        profile = tmp_path / name
        source = profile / "source"
        materialize_source(ROOT, contract, source)
        output = profile / "frontend-static.tar.gz"
        build_frontend(source, output, contract, ROOT, profile)
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
    assert sha256_bytes(outputs[0]) == sha256_bytes(outputs[1])
