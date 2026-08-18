from __future__ import annotations

import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.check_reproduction_capsule as capsule_cli  # noqa: E402
from scholar_agent.evaluation.reproduction_capsule import (  # noqa: E402
    CAPSULE_MANIFEST_NAME,
    CONTRACT_VERSION,
    EXIT_INTEGRITY_OR_REPLAY_MISMATCH,
    EXIT_NOT_ELIGIBLE,
    MAX_MEMBERS,
    CapsuleIntegrityError,
    CapsuleNotEligible,
    audit_frozen_baseline_eligibility,
    export_capsule,
    load_gate_protocol,
    materialize_local_replay_run,
    replay_capsule,
    verify_capsule,
)
from scholar_agent.evaluation.snapshot_resume import stable_hash  # noqa: E402


PROTOCOL = ROOT / "benchmark" / "reproduction_capsule_v1_protocol.json"
EXECUTION_PROTOCOL = ROOT / "benchmark" / "execution_determinism_v1_protocol.json"


def _source(tmp_path: Path, name: str = "source") -> Path:
    return materialize_local_replay_run(
        tmp_path / name,
        host_repository_root=ROOT,
        execution_protocol_path=EXECUTION_PROTOCOL,
    )


def _capsule(tmp_path: Path) -> tuple[Path, Path]:
    source = _source(tmp_path)
    archive = tmp_path / "fixture.tar"
    export_capsule(source, archive, host_repository_root=ROOT)
    return source, archive


def _members(path: Path) -> dict[str, tuple[tarfile.TarInfo, bytes]]:
    result: dict[str, tuple[tarfile.TarInfo, bytes]] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive.getmembers():
            handle = archive.extractfile(member)
            result[member.name] = (member, handle.read() if handle else b"")
    return result


def _write_tar(
    path: Path,
    rows: list[tuple[str, bytes, str]],
) -> None:
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, content, kind in rows:
            info = tarfile.TarInfo(name)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.mode = 0o644
            if kind == "file":
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = "payload/run_manifest.json"
                archive.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = "payload/run_manifest.json"
                archive.addfile(info)
            else:  # pragma: no cover - helper misuse
                raise AssertionError(kind)


def _rewrite(
    source: Path,
    target: Path,
    transform: Callable[[list[tuple[str, bytes, str]]], list[tuple[str, bytes, str]]],
) -> None:
    rows = [
        (name, content, "file")
        for name, (_info, content) in sorted(_members(source).items())
    ]
    _write_tar(target, transform(rows))


@pytest.mark.reproduction_capsule_regression
def test_deterministic_export_verify_and_cross_directory_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    first = tmp_path / "one.tar"
    second = tmp_path / "two.tar"
    export_one = export_capsule(source, first, host_repository_root=ROOT)
    export_two = export_capsule(source, second, host_repository_root=ROOT)
    assert first.read_bytes() == second.read_bytes()
    assert export_one == export_two

    verify_one = verify_capsule(first)
    verify_two = verify_capsule(second)
    assert verify_one == verify_two
    foreign_cwd = tmp_path / "different-parent" / "cwd"
    foreign_cwd.mkdir(parents=True)
    monkeypatch.chdir(foreign_cwd)
    replay = replay_capsule(first, host_repository_root=ROOT)
    assert replay["status"] == "passed"
    assert replay["replay"]["query_count"] == 3
    assert replay["replay"]["network_request_count"] == 0
    assert replay["replay"]["llm_request_count"] == 0
    assert replay["replay"]["snapshot_write_count"] == 0
    assert replay["replay"]["archived_code_execution_count"] == 0


def test_file_tamper_missing_and_extra_members_are_rejected(tmp_path: Path) -> None:
    _source_path, archive = _capsule(tmp_path)
    tampered = tmp_path / "tampered.tar"

    def change(rows: list[tuple[str, bytes, str]]) -> list[tuple[str, bytes, str]]:
        result = list(rows)
        index = next(i for i, row in enumerate(result) if "retrieval_outputs" in row[0])
        name, content, kind = result[index]
        result[index] = (name, content + b"\n", kind)
        return result

    _rewrite(archive, tampered, change)
    with pytest.raises(CapsuleIntegrityError, match="hash_or_size"):
        verify_capsule(tampered)

    missing = tmp_path / "missing.tar"
    _rewrite(
        archive,
        missing,
        lambda rows: [row for row in rows if "retrieval_outputs" not in row[0]],
    )
    with pytest.raises(CapsuleIntegrityError, match="inventory"):
        verify_capsule(missing)

    extra = tmp_path / "extra.tar"
    _rewrite(archive, extra, lambda rows: [*rows, ("payload/extra.json", b"{}", "file")])
    with pytest.raises(CapsuleIntegrityError, match="inventory"):
        verify_capsule(extra)


@pytest.mark.parametrize(
    ("name", "kind", "reason"),
    [
        ("../escape", "file", "unsafe_archive_member_path"),
        ("/absolute", "file", "unsafe_archive_member_path"),
        ("payload/link", "symlink", "links_or_special"),
        ("payload/hard", "hardlink", "links_or_special"),
    ],
)
def test_unsafe_paths_and_links_are_rejected(
    tmp_path: Path, name: str, kind: str, reason: str
) -> None:
    archive = tmp_path / "unsafe.tar"
    _write_tar(archive, [(name, b"x", kind)])
    with pytest.raises(CapsuleIntegrityError, match=reason):
        verify_capsule(archive)


def test_duplicate_and_normalized_path_collision_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.tar"
    _write_tar(
        duplicate,
        [
            (CAPSULE_MANIFEST_NAME, b"{}", "file"),
            (CAPSULE_MANIFEST_NAME, b"{}", "file"),
        ],
    )
    with pytest.raises(CapsuleIntegrityError, match="duplicate_archive_member"):
        verify_capsule(duplicate)

    collision = tmp_path / "collision.tar"
    _write_tar(
        collision,
        [
            (CAPSULE_MANIFEST_NAME, b"{}", "file"),
            ("payload/Case.json", b"{}", "file"),
            ("payload/case.json", b"{}", "file"),
        ],
    )
    with pytest.raises(CapsuleIntegrityError, match="normalized_archive_path_collision"):
        verify_capsule(collision)


def test_resource_member_count_limit_is_enforced(tmp_path: Path) -> None:
    archive = tmp_path / "oversized-member-count.tar"
    rows = [
        (f"payload/item-{index:04d}.json", b"{}", "file")
        for index in range(MAX_MEMBERS + 1)
    ]
    _write_tar(archive, rows)
    with pytest.raises(CapsuleIntegrityError, match="file_count_limit"):
        verify_capsule(archive)


def test_source_query_configuration_and_external_dependency_drift_are_ineligible(
    tmp_path: Path,
) -> None:
    query_source = _source(tmp_path, "query-source")
    query_path = query_source / "inputs/queries.jsonl"
    rows = query_path.read_text(encoding="utf-8").splitlines()
    query_path.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")
    with pytest.raises(CapsuleNotEligible, match="run_manifest_v1_invalid"):
        export_capsule(query_source, tmp_path / "query.tar", host_repository_root=ROOT)

    config_source = _source(tmp_path, "config-source")
    config = config_source / "artifacts/config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["configuration"]["budgets"]["max_search_rounds"] += 1
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CapsuleNotEligible, match="run_manifest_v1_invalid"):
        export_capsule(config_source, tmp_path / "config.tar", host_repository_root=ROOT)

    external_source = _source(tmp_path, "external-source")
    fixture = external_source / "inputs/retrieval_outputs.json"
    fixture.unlink()
    fixture.symlink_to(ROOT / "datasets/eval_fixtures/sample/retrieval_outputs.json")
    with pytest.raises((CapsuleNotEligible, CapsuleIntegrityError)):
        export_capsule(external_source, tmp_path / "external.tar", host_repository_root=ROOT)


def test_generation_lineage_damage_is_not_exported(tmp_path: Path) -> None:
    source = _source(tmp_path)
    generations = sorted((source / "artifacts/.run_commits/generations").glob("generation-*"))
    latest = generations[-1]
    manifest = latest / "generation_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["parent_generation"] = payload["generation"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CapsuleNotEligible, match="does_not_match|unavailable"):
        export_capsule(source, tmp_path / "broken.tar", host_repository_root=ROOT)


def test_semantic_replay_drift_produces_stable_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source_path, archive = _capsule(tmp_path)
    code = capsule_cli.main(
        [
            "--repository-root",
            str(ROOT),
            "replay",
            "--capsule",
            str(archive),
            "--fault",
            "semantic_result_change",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_INTEGRITY_OR_REPLAY_MISMATCH
    assert payload["status"] == "integrity_or_replay_mismatch"
    assert payload["violation"]["invariant"] == "canonical_query_results"
    assert "deduplicated_count" in payload["violation"]["first_difference_path"]


def test_cli_success_json_is_byte_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _source_path, archive = _capsule(tmp_path)
    arguments = [
        "--repository-root",
        str(ROOT),
        "verify",
        "--capsule",
        str(archive),
    ]
    assert capsule_cli.main(arguments) == 0
    first = capsys.readouterr().out
    assert capsule_cli.main(arguments) == 0
    second = capsys.readouterr().out
    assert first == second
    assert json.loads(first)["status"] == "passed"


def test_frozen_record160_and_full1000_are_structurally_not_eligible() -> None:
    protocol = load_gate_protocol(PROTOCOL, repository_root=ROOT)
    report = audit_frozen_baseline_eligibility(protocol, repository_root=ROOT)
    assert report["exit_code"] == EXIT_NOT_ELIGIBLE
    assert report["profile_count"] == 2
    assert {row["profile_id"] for row in report["profiles"]} == {
        "autoscholar_record160_analysis_input",
        "autoscholar_full1000_frozen_baseline",
    }
    assert all(row["status"] == "not_eligible" for row in report["profiles"])
    assert all(row["files_modified"] == 0 for row in report["profiles"])


def test_manifest_rejects_self_hash_and_inventory_drift(tmp_path: Path) -> None:
    _source_path, archive = _capsule(tmp_path)
    malformed = tmp_path / "manifest-drift.tar"

    def mutate(rows: list[tuple[str, bytes, str]]) -> list[tuple[str, bytes, str]]:
        result = []
        for name, content, kind in rows:
            if name == CAPSULE_MANIFEST_NAME:
                payload: dict[str, Any] = json.loads(content)
                payload["query_set"]["count"] += 1
                without_hash = dict(payload)
                without_hash.pop("capsule_summary_sha256")
                payload["capsule_summary_sha256"] = stable_hash(without_hash)
                content = json.dumps(payload, sort_keys=True).encode("utf-8")
            result.append((name, content, kind))
        return result

    _rewrite(archive, malformed, mutate)
    with pytest.raises(CapsuleIntegrityError, match="capsule_manifest_invalid"):
        verify_capsule(malformed)


def test_protocol_and_reports_do_not_claim_quality_scores(tmp_path: Path) -> None:
    _source_path, archive = _capsule(tmp_path)
    manifest = json.loads(_members(archive)[CAPSULE_MANIFEST_NAME][1])
    serialized = json.dumps(manifest, sort_keys=True).lower()
    assert manifest["protocol"] == CONTRACT_VERSION
    assert manifest["score_scope"] == (
        "portable_replay_only_not_quality_or_official_score"
    )
    for forbidden in ('"precision"', '"recall"', '"f1"', '"official_score"'):
        assert forbidden not in serialized
