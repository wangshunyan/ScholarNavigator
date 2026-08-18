from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.check_runtime_hermeticity as hermeticity_cli  # noqa: E402
from scholar_agent.evaluation.execution_determinism import (  # noqa: E402
    load_protocol as load_execution_protocol,
    replay_canonical_fixture,
)
from scholar_agent.evaluation.runtime_hermeticity import (  # noqa: E402
    EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION,
    EXIT_NOT_ELIGIBLE,
    RuntimeHermeticityError,
    audit_frozen_baseline_eligibility,
    load_protocol,
    run_runtime_hermeticity_gate,
    stable_json_bytes,
)


PROTOCOL_PATH = ROOT / "benchmark" / "runtime_hermeticity_v1_protocol.json"
EXECUTION_PROTOCOL = ROOT / "benchmark" / "execution_determinism_v1_protocol.json"


def _protocol() -> dict:
    return load_protocol(PROTOCOL_PATH, repository_root=ROOT)


@pytest.mark.runtime_hermeticity_regression
def test_repository_runtime_hermeticity_profiles_pass_and_are_byte_stable() -> None:
    first = run_runtime_hermeticity_gate(_protocol(), repository_root=ROOT)
    second = run_runtime_hermeticity_gate(_protocol(), repository_root=ROOT)

    assert first["status"] == "passed"
    assert first["exit_code"] == 0
    assert first["profile_count"] == 7
    assert first["supported_profile_count"] == 7
    assert first["violation_count"] == 0
    assert stable_json_bytes(first) == stable_json_bytes(second)
    assert {row["kind"] for row in first["profiles"]} == {
        "minimal_environment",
        "hash_seed",
        "working_directory_home_tmpdir",
        "timezone",
        "locale",
        "thread_environment",
        "polluted_environment",
    }
    assert {row["semantic_sha256"] for row in first["profiles"]} == {
        first["profiles"][0]["semantic_sha256"]
    }
    assert all(row["lineage_sha256"] for row in first["profiles"])
    serialized = stable_json_bytes(first).decode("utf-8").casefold()
    assert "hermeticity_sentinel" not in serialized
    assert "/users/" not in serialized


@pytest.mark.parametrize(
    ("fault", "operation"),
    [
        ("dotenv_read", "file_read"),
        ("forbidden_file_read", "file_read"),
        ("network_attempt", "network_attempt"),
        ("forbidden_file_write", "file_write"),
        ("cache_residue", "file_residue"),
        ("subprocess_attempt", "subprocess_attempt"),
        ("sensitive_environment_read", "environment_read"),
        ("sensitive_sentinel_echo", "sensitive_output"),
    ],
)
def test_forbidden_io_and_sensitive_echo_are_attributed(
    fault: str, operation: str
) -> None:
    report = run_runtime_hermeticity_gate(
        _protocol(),
        repository_root=ROOT,
        fault=fault,
        profile_ids=["minimal"],
    )
    assert report["exit_code"] == EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION
    assert report["status"] == "hermeticity_or_semantic_violation"
    assert operation in {item["operation_type"] for item in report["violations"]}
    serialized = stable_json_bytes(report).decode("utf-8").casefold()
    assert "hermeticity_sentinel" not in serialized
    assert "/users/" not in serialized


@pytest.mark.parametrize(
    ("fault", "profiles"),
    [
        ("hash_seed_semantic_drift", ["minimal", "hash-seed-98765"]),
        ("timezone_semantic_drift", ["minimal", "timezone-asia-shanghai"]),
        ("cwd_semantic_drift", ["minimal", "cwd-home-tmp-alternate"]),
        ("home_semantic_drift", ["minimal", "cwd-home-tmp-alternate"]),
    ],
)
def test_environment_dependent_semantics_are_rejected(
    fault: str, profiles: list[str]
) -> None:
    report = run_runtime_hermeticity_gate(
        _protocol(),
        repository_root=ROOT,
        fault=fault,
        profile_ids=profiles,
    )
    assert report["exit_code"] == EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION
    assert any(
        item["invariant"] == "normalized_semantics_equal_across_profiles"
        for item in report["violations"]
    )


def test_unsupported_locale_is_structured_not_eligible() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["environment_profiles"][0]["locale"] = "not_A_real_locale.UTF-8"
    report = run_runtime_hermeticity_gate(
        protocol,
        repository_root=ROOT,
        profile_ids=["minimal"],
    )
    assert report["exit_code"] == EXIT_NOT_ELIGIBLE
    assert report["status"] == "profile_not_supported"
    assert report["profiles"][0]["reason"] == "locale_profile_not_supported"


def test_protocol_rejects_broad_or_drifted_allowed_inputs(tmp_path: Path) -> None:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    value["allowed_inputs"][0]["path"] = "benchmark"
    temporary = tmp_path / "invalid-protocol.json"
    temporary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeHermeticityError):
        load_protocol(temporary, repository_root=ROOT)


def test_replay_lineage_collection_is_opt_in_and_semantically_observational(
    tmp_path: Path,
) -> None:
    protocol = load_execution_protocol(EXECUTION_PROTOCOL, repository_root=ROOT)
    without = replay_canonical_fixture(
        protocol, repository_root=ROOT, snapshot_root=tmp_path / "without"
    )
    with_lineage = replay_canonical_fixture(
        protocol,
        repository_root=ROOT,
        snapshot_root=tmp_path / "with",
        collect_result_lineage=True,
    )
    assert "result_lineage" not in without
    assert with_lineage["result_lineage"]["query_count"] == 3
    assert without["records_sha256"] == with_lineage["records_sha256"]
    assert without["records"] == with_lineage["records"]


def test_frozen_baselines_are_read_only_not_eligible() -> None:
    report = audit_frozen_baseline_eligibility(_protocol(), repository_root=ROOT)
    assert report["exit_code"] == EXIT_NOT_ELIGIBLE
    assert report["eligible_count"] == 0
    assert report["profile_count"] >= 2
    assert all(row["status"] == "not_eligible" for row in report["profiles"])


def test_cli_exit_codes_for_pass_violation_and_frozen_audit(capsys) -> None:
    passed = hermeticity_cli.main(["check"])
    assert passed == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"

    violated = hermeticity_cli.main(["check", "--fault", "network_attempt"])
    assert violated == EXIT_HERMETICITY_OR_SEMANTIC_VIOLATION
    violation_report = json.loads(capsys.readouterr().out)
    assert violation_report["status"] == "hermeticity_or_semantic_violation"

    frozen = hermeticity_cli.main(["audit-frozen"])
    assert frozen == EXIT_NOT_ELIGIBLE
    assert json.loads(capsys.readouterr().out)["status"] == "not_eligible"
