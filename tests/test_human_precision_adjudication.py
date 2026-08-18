from __future__ import annotations

import copy
import hashlib
import json
import runpy
import shutil
from pathlib import Path
from typing import Any

import pytest

from scholar_agent.evaluation.human_precision_adjudication import (
    LabelIntegrityViolation,
    PackageNotEligible,
    invalid_report,
    load_protocol,
    run_human_precision_gate,
    write_json,
)
from scholar_agent.evaluation.precision_annotation import LABELS, PUBLIC_FIELDS
from scholar_agent.evaluation.snapshot_resume import stable_hash


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REAL_PROTOCOL = (
    REPOSITORY_ROOT / "benchmark" / "human_precision_adjudication_v1_protocol.json"
)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_digest(root: Path) -> tuple[str, int]:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return stable_hash(entries), len(entries)


def build_synthetic_protocol(root: Path) -> tuple[Path, dict[str, Any]]:
    package_root = root / "blind_package"
    item_ids = ["SPAR-LX-TEST-0001", "SPAR-LX-TEST-0002"]
    manifest = {
        "package": "synthetic_blind_package_v1",
        "version": "1",
        "annotation": {"labels": list(LABELS)},
        "forbidden_public_fields": [
            "gold",
            "qrels",
            "case_id",
            "strategy",
            "rank",
            "source",
            "score",
        ],
    }
    rubric = {
        "schema_version": "1",
        "labels": list(LABELS),
        "definitions": {
            "relevant": "directly answers or materially supports the query",
            "partially_relevant": "addresses a meaningful part of the query but is incomplete or indirect",
            "not_relevant": "does not materially address the query",
            "insufficient_information": "title and abstract are insufficient for a reliable decision",
        },
        "independence": "annotators work independently before adjudication",
        "adjudication": "required only where the two labels differ",
        "public_sample_fields": list(PUBLIC_FIELDS),
    }
    public_rows = [
        {
            "sample_id": item_id,
            "query": "blind query",
            "title": f"Blind paper {index}",
            "abstract": "Only public evidence.",
            "year": 2025,
        }
        for index, item_id in enumerate(item_ids, start=1)
    ]
    annotation = [
        {
            "sample_id": item_id,
            "annotator_id": "annotator_template",
            "label": None,
            "notes": "",
        }
        for item_id in item_ids
    ]
    adjudication = [
        {
            "sample_id": item_id,
            "adjudicator_id": "",
            "final_label": None,
            "rationale": "",
        }
        for item_id in item_ids
    ]
    mapping = {
        "schema_version": "1",
        "package": manifest["package"],
        "case_count": 1,
        "top_k": 20,
        "population": {},
        "samples": [
            {
                "sample_id": item_ids[0],
                "occurrences": [
                    {
                        "case_id": "private-query-a",
                        "case_order": 1,
                        "direction": "experiment_admitted",
                        "overlaps_prior_auto_dev_val": False,
                        "rank": 1,
                        "successful_source_count": 4,
                    }
                ],
            },
            {
                "sample_id": item_ids[1],
                "occurrences": [
                    {
                        "case_id": "private-query-a",
                        "case_order": 1,
                        "direction": "baseline_removed",
                        "overlaps_prior_auto_dev_val": False,
                        "rank": 2,
                        "successful_source_count": 4,
                    }
                ],
            },
        ],
        "prior_package_overlaps": [],
    }
    _json(package_root / "manifest.json", manifest)
    _json(package_root / "public" / "annotation_schema.json", rubric)
    _jsonl(package_root / "public" / "blind_samples.jsonl", public_rows)
    _jsonl(package_root / "public" / "annotator_1.jsonl", annotation)
    _jsonl(package_root / "public" / "annotator_2.jsonl", annotation)
    _jsonl(package_root / "public" / "adjudication.jsonl", adjudication)
    _json(package_root / "private" / "mapping.json", mapping)
    digest, file_count = _package_digest(package_root)
    package_binding = {
        "package_id": manifest["package"],
        "package_version": manifest["version"],
        "root": package_root.relative_to(root).as_posix(),
        "package_sha256": digest,
        "file_count": file_count,
        "expected_item_count": len(item_ids),
        "item_set_sha256": stable_hash(sorted(item_ids)),
        "required_prior_item_count": 0,
        "manifest_path": "manifest.json",
        "rubric_path": "public/annotation_schema.json",
        "mapping_path": "private/mapping.json",
        "public_samples_path": "public/blind_samples.jsonl",
        "annotator_1_template_path": "public/annotator_1.jsonl",
        "annotator_2_template_path": "public/annotator_2.jsonl",
        "adjudication_template_path": "public/adjudication.jsonl",
    }
    protocol = {
        "schema_version": "1",
        "contract": "human_precision_adjudication_v1",
        "digest_algorithms": {
            "opaque_item_identity": "sha256_utf8_v1",
            "package_tree": "sorted_relative_path_size_sha256_v1",
        },
        "score_scope": "internal_non_official_human_precision",
        "annotation": {
            "labels": list(LABELS),
            "independent_annotator_count": 2,
            "adjudication_required_on_disagreement": True,
            "annotation_optional_fields": ["notes"],
            "adjudication_optional_fields": ["rationale"],
            "annotator_identity_format": "opaque_anon_identifier_v1",
            "confidence_supported": False,
        },
        "exclusions": {"allowed_reasons": [], "expected_count": 0},
        "evaluator": {
            "identity_version": "deduplicated_gold_identity_v2",
            "statistics_version": "full_swap_precision_annotation_v1",
            "official_scorer": False,
        },
        "package": package_binding,
        "prior_package": None,
    }
    protocol_path = root / "protocol.json"
    _json(protocol_path, protocol)
    return protocol_path, protocol


def write_submission(
    path: Path,
    protocol: dict[str, Any],
    *,
    round_id: str,
    labels: list[tuple[str, str]],
    identity: str,
) -> Path:
    binding = protocol["package"]
    package = {
        "package_id": binding["package_id"],
        "package_version": binding["package_version"],
        "package_sha256": binding["package_sha256"],
    }
    if round_id == "adjudication":
        payload = {
            "schema_version": "1",
            "contract": "human_precision_adjudication_v1",
            "package": package,
            "round_id": round_id,
            "adjudicator_id": identity,
            "decisions": [
                {"item_id": item_id, "final_label": label, "rationale": ""}
                for item_id, label in labels
            ],
        }
    else:
        payload = {
            "schema_version": "1",
            "contract": "human_precision_adjudication_v1",
            "package": package,
            "round_id": round_id,
            "annotator_id": identity,
            "labels": [
                {"item_id": item_id, "label": label, "notes": ""}
                for item_id, label in labels
            ],
        }
    _json(path, payload)
    return path


def attach_prior_package(root: Path, protocol: dict[str, Any]) -> str:
    current_root = root / protocol["package"]["root"]
    prior_root = root / "prior_package"
    shutil.copytree(current_root, prior_root)
    prior_id = "SPAR-LX-PRIOR-0001"
    prior_manifest = json.loads((prior_root / "manifest.json").read_text())
    prior_manifest["package"] = "synthetic_prior_blind_package_v1"
    _json(prior_root / "manifest.json", prior_manifest)
    prior_mapping = json.loads(
        (prior_root / "private" / "mapping.json").read_text()
    )
    prior_mapping["package"] = prior_manifest["package"]
    prior_mapping["samples"] = [{"sample_id": prior_id}]
    _json(prior_root / "private" / "mapping.json", prior_mapping)
    prior_public = {
        "sample_id": prior_id,
        "query": "prior blind query",
        "title": "Prior blind paper",
        "abstract": "Prior public evidence.",
        "year": 2024,
    }
    _jsonl(prior_root / "public" / "blind_samples.jsonl", [prior_public])
    prior_annotation = {
        "sample_id": prior_id,
        "annotator_id": "annotator_template",
        "label": None,
        "notes": "",
    }
    _jsonl(prior_root / "public" / "annotator_1.jsonl", [prior_annotation])
    _jsonl(prior_root / "public" / "annotator_2.jsonl", [prior_annotation])
    _jsonl(
        prior_root / "public" / "adjudication.jsonl",
        [
            {
                "sample_id": prior_id,
                "adjudicator_id": "",
                "final_label": None,
                "rationale": "",
            }
        ],
    )
    prior_digest, prior_file_count = _package_digest(prior_root)
    protocol["prior_package"] = {
        "package_id": prior_manifest["package"],
        "package_version": prior_manifest["version"],
        "root": prior_root.relative_to(root).as_posix(),
        "package_sha256": prior_digest,
        "file_count": prior_file_count,
        "expected_item_count": 1,
        "item_set_sha256": stable_hash([prior_id]),
        "manifest_path": "manifest.json",
        "rubric_path": "public/annotation_schema.json",
        "mapping_path": "private/mapping.json",
        "public_samples_path": "public/blind_samples.jsonl",
        "annotator_1_template_path": "public/annotator_1.jsonl",
        "annotator_2_template_path": "public/annotator_2.jsonl",
        "adjudication_template_path": "public/adjudication.jsonl",
    }
    current_mapping_path = current_root / "private" / "mapping.json"
    current_mapping = json.loads(current_mapping_path.read_text())
    current_mapping["prior_package_overlaps"] = [
        {
            "prior_sample_id": prior_id,
            "occurrences": [
                {
                    "case_id": "private-query-a",
                    "case_order": 1,
                    "direction": "baseline_removed",
                    "overlaps_prior_auto_dev_val": True,
                    "rank": 3,
                    "successful_source_count": 4,
                }
            ],
        }
    ]
    _json(current_mapping_path, current_mapping)
    current_digest, current_file_count = _package_digest(current_root)
    protocol["package"]["package_sha256"] = current_digest
    protocol["package"]["file_count"] = current_file_count
    protocol["package"]["required_prior_item_count"] = 1
    protocol["package"]["required_prior_item_set_sha256"] = stable_hash(
        [prior_id]
    )
    return prior_id


def _run_complete(tmp_path: Path) -> dict[str, Any]:
    protocol_path, raw_protocol = build_synthetic_protocol(tmp_path)
    ids = ["SPAR-LX-TEST-0001", "SPAR-LX-TEST-0002"]
    labels = [(ids[0], "relevant"), (ids[1], "not_relevant")]
    first = write_submission(
        tmp_path / "first.json",
        raw_protocol,
        round_id="independent_1",
        labels=labels,
        identity="anon-a",
    )
    second = write_submission(
        tmp_path / "second.json",
        raw_protocol,
        round_id="independent_2",
        labels=labels,
        identity="anon-b",
    )
    protocol = load_protocol(protocol_path, repository_root=tmp_path)
    return run_human_precision_gate(
        protocol,
        repository_root=tmp_path,
        annotator_one_path=first,
        annotator_two_path=second,
        snapshot_root=tmp_path / "snapshots",
    )


def test_complete_agreement_validates_and_report_is_deterministic(tmp_path: Path) -> None:
    first = _run_complete(tmp_path)
    second = _run_complete(tmp_path)

    assert first == second
    assert first["state"] == "validated"
    assert first["exit_code"] == 0
    assert first["agreement"]["disagreement_count"] == 0
    assert first["statistics"]["scope"] == "internal_non_official_change_only_precision"
    assert first["statistics"]["metrics"]["precision_at_20"] == {
        "baseline": None,
        "experiment": None,
        "reason": "unchanged_top20_items_are_not_in_the_change_only_package",
    }
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    write_json(left, first)
    write_json(right, second)
    assert left.read_bytes() == right.read_bytes()


def test_disagreement_requires_and_accepts_independent_adjudication(tmp_path: Path) -> None:
    protocol_path, raw = build_synthetic_protocol(tmp_path)
    ids = ["SPAR-LX-TEST-0001", "SPAR-LX-TEST-0002"]
    first = write_submission(
        tmp_path / "first.json",
        raw,
        round_id="independent_1",
        labels=[(ids[0], "relevant"), (ids[1], "not_relevant")],
        identity="anon-a",
    )
    second = write_submission(
        tmp_path / "second.json",
        raw,
        round_id="independent_2",
        labels=[(ids[0], "partially_relevant"), (ids[1], "not_relevant")],
        identity="anon-b",
    )
    protocol = load_protocol(protocol_path, repository_root=tmp_path)
    pending = run_human_precision_gate(
        protocol,
        repository_root=tmp_path,
        annotator_one_path=first,
        annotator_two_path=second,
        snapshot_root=tmp_path / "snapshots",
    )
    assert pending["state"] == "adjudication_required"
    assert pending["exit_code"] == 3
    assert pending["agreement"]["disagreement_count"] == 1
    adjudication = write_submission(
        tmp_path / "adjudication.json",
        raw,
        round_id="adjudication",
        labels=[(ids[0], "relevant")],
        identity="anon-c",
    )
    complete = run_human_precision_gate(
        protocol,
        repository_root=tmp_path,
        annotator_one_path=first,
        annotator_two_path=second,
        adjudication_path=adjudication,
        snapshot_root=tmp_path / "snapshots",
    )
    assert complete["state"] == "validated"
    assert complete["adjudication_trace"]["resolved_disagreement_count"] == 1
    assert {row["resolution"] for row in complete["adjudication_trace"]["decision_records"]} == {
        "annotator_agreement",
        "adjudicated",
    }


def test_partial_coverage_waits_without_generating_statistics(tmp_path: Path) -> None:
    protocol_path, raw = build_synthetic_protocol(tmp_path)
    first = write_submission(
        tmp_path / "first.json",
        raw,
        round_id="independent_1",
        labels=[("SPAR-LX-TEST-0001", "relevant")],
        identity="anon-a",
    )
    protocol = load_protocol(protocol_path, repository_root=tmp_path)
    report = run_human_precision_gate(
        protocol,
        repository_root=tmp_path,
        annotator_one_path=first,
        snapshot_root=tmp_path / "snapshots",
    )
    assert report["state"] == "awaiting_labels"
    assert report["statistics"] is None
    assert report["coverage"]["received_counts"]["independent_1"] == 1
    assert len(
        report["coverage"]["missing_item_identity_sha256"]["independent_1"]
    ) == 1
    assert len(
        report["coverage"]["missing_item_identity_sha256"]["independent_2"]
    ) == 2


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("unknown_item", "unknown_item_identity"),
        ("duplicate_row", "duplicate_item_submission"),
        ("illegal_label", "invalid_annotation_label"),
        ("leakage_field", "independent_submission_schema_invalid"),
        ("non_anonymous_annotator", "independent_submission_schema_invalid"),
        ("cross_package", "submission_package_mismatch"),
    ],
)
def test_invalid_submissions_are_rejected_without_echoing_identity(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    protocol_path, raw = build_synthetic_protocol(tmp_path)
    first_path = write_submission(
        tmp_path / "first.json",
        raw,
        round_id="independent_1",
        labels=[
            ("SPAR-LX-TEST-0001", "relevant"),
            ("SPAR-LX-TEST-0002", "not_relevant"),
        ],
        identity="anon-a",
    )
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    if mutation == "unknown_item":
        payload["labels"][0]["item_id"] = "SECRET-UNKNOWN-ITEM"
    elif mutation == "duplicate_row":
        payload["labels"].append(copy.deepcopy(payload["labels"][0]))
    elif mutation == "illegal_label":
        payload["labels"][0]["label"] = "gold"
    elif mutation == "leakage_field":
        payload["labels"][0]["case_id"] = "hidden-case"
    elif mutation == "non_anonymous_annotator":
        payload["annotator_id"] = "person@example.invalid"
    elif mutation == "cross_package":
        payload["package"]["package_id"] = "different-blind-package"
    _json(first_path, payload)
    protocol = load_protocol(protocol_path, repository_root=tmp_path)
    with pytest.raises(LabelIntegrityViolation) as caught:
        run_human_precision_gate(
            protocol,
            repository_root=tmp_path,
            annotator_one_path=first_path,
            snapshot_root=tmp_path / "snapshots",
        )
    report = invalid_report(caught.value)
    assert report["state"] == "invalid"
    assert report["exit_code"] == 2
    assert report["violations"][0]["code"] == expected_code
    assert "SECRET-UNKNOWN-ITEM" not in json.dumps(report, sort_keys=True)


def test_same_annotator_and_forged_adjudication_are_rejected(tmp_path: Path) -> None:
    protocol_path, raw = build_synthetic_protocol(tmp_path)
    ids = ["SPAR-LX-TEST-0001", "SPAR-LX-TEST-0002"]
    labels = [(ids[0], "relevant"), (ids[1], "not_relevant")]
    first = write_submission(
        tmp_path / "first.json",
        raw,
        round_id="independent_1",
        labels=labels,
        identity="anon-same",
    )
    second = write_submission(
        tmp_path / "second.json",
        raw,
        round_id="independent_2",
        labels=labels,
        identity="anon-same",
    )
    protocol = load_protocol(protocol_path, repository_root=tmp_path)
    with pytest.raises(LabelIntegrityViolation, match="annotator_identity_reused"):
        run_human_precision_gate(
            protocol,
            repository_root=tmp_path,
            annotator_one_path=first,
            annotator_two_path=second,
            snapshot_root=tmp_path / "snapshots",
        )

    second = write_submission(
        second,
        raw,
        round_id="independent_2",
        labels=labels,
        identity="anon-other",
    )
    forged = write_submission(
        tmp_path / "adjudication.json",
        raw,
        round_id="adjudication",
        labels=[(ids[0], "relevant")],
        identity="anon-adjudicator",
    )
    with pytest.raises(
        LabelIntegrityViolation, match="adjudication_without_disagreement"
    ):
        run_human_precision_gate(
            protocol,
            repository_root=tmp_path,
            annotator_one_path=first,
            annotator_two_path=second,
            adjudication_path=forged,
            snapshot_root=tmp_path / "snapshots",
        )


def test_package_hash_and_incomplete_rubric_are_not_eligible(tmp_path: Path) -> None:
    protocol_path, raw = build_synthetic_protocol(tmp_path)
    package_root = tmp_path / raw["package"]["root"]
    public_path = package_root / "public" / "blind_samples.jsonl"
    public_path.write_text(public_path.read_text(encoding="utf-8") + "\n")
    protocol = load_protocol(protocol_path, repository_root=tmp_path)
    with pytest.raises(PackageNotEligible, match="package_digest_mismatch"):
        run_human_precision_gate(
            protocol,
            repository_root=tmp_path,
            snapshot_root=tmp_path / "snapshots",
        )

    protocol_path, raw = build_synthetic_protocol(tmp_path / "rubric")
    package_root = (tmp_path / "rubric") / raw["package"]["root"]
    rubric_path = package_root / "public" / "annotation_schema.json"
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    rubric["definitions"]["relevant"] = ""
    _json(rubric_path, rubric)
    digest, file_count = _package_digest(package_root)
    raw["package"]["package_sha256"] = digest
    raw["package"]["file_count"] = file_count
    _json(protocol_path, raw)
    protocol = load_protocol(protocol_path, repository_root=tmp_path / "rubric")
    with pytest.raises(PackageNotEligible, match="rubric_definitions_incomplete"):
        run_human_precision_gate(
            protocol,
            repository_root=tmp_path / "rubric",
            snapshot_root=tmp_path / "rubric" / "snapshots",
        )


def test_cli_usage_error_has_stable_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "scripts" / "check_human_precision_labels.py")
    )
    assert namespace["main"](["--unknown-option"]) == 4
    report = json.loads(capsys.readouterr().out)
    assert report["exit_code"] == 4
    assert report["reason"] == "usage_error"


def test_prior_package_resolution_keeps_original_judgments_traceable(
    tmp_path: Path,
) -> None:
    protocol_path, raw = build_synthetic_protocol(tmp_path)
    prior_id = attach_prior_package(tmp_path, raw)
    _json(protocol_path, raw)
    labels = [
        ("SPAR-LX-TEST-0001", "relevant"),
        ("SPAR-LX-TEST-0002", "not_relevant"),
    ]
    first = write_submission(
        tmp_path / "first.json",
        raw,
        round_id="independent_1",
        labels=labels,
        identity="anon-a",
    )
    second = write_submission(
        tmp_path / "second.json",
        raw,
        round_id="independent_2",
        labels=labels,
        identity="anon-b",
    )
    protocol = load_protocol(protocol_path, repository_root=tmp_path)
    pending = run_human_precision_gate(
        protocol,
        repository_root=tmp_path,
        annotator_one_path=first,
        annotator_two_path=second,
        snapshot_root=tmp_path / "snapshots",
    )
    assert pending["state"] == "awaiting_labels"
    assert pending["reason"] == "resolved_prior_package_labels_required"

    prior_binding = raw["prior_package"]
    prior = {
        "schema_version": "1",
        "contract": "human_precision_adjudication_v1",
        "package": {
            "package_id": prior_binding["package_id"],
            "package_version": prior_binding["package_version"],
            "package_sha256": prior_binding["package_sha256"],
        },
        "round_id": "resolved_prior_package",
        "annotator_1_id": "anon-prior-a",
        "annotator_2_id": "anon-prior-b",
        "adjudicator_id": "anon-prior-c",
        "labels": [
            {
                "item_id": prior_id,
                "annotator_1_label": "relevant",
                "annotator_2_label": "partially_relevant",
                "final_label": "relevant",
                "resolution": "adjudicated",
            }
        ],
    }
    prior_path = tmp_path / "prior.json"
    _json(prior_path, prior)
    complete = run_human_precision_gate(
        protocol,
        repository_root=tmp_path,
        annotator_one_path=first,
        annotator_two_path=second,
        prior_resolved_path=prior_path,
        snapshot_root=tmp_path / "snapshots",
    )
    assert complete["state"] == "validated"
    assert complete["statistics"]["prior_resolved_item_count"] == 1
    assert complete["input_artifacts"]["prior_resolved"]["sha256"] == _sha256(
        prior_path
    )

    prior["annotator_2_id"] = "anon-prior-a"
    _json(prior_path, prior)
    with pytest.raises(
        LabelIntegrityViolation, match="prior_annotator_identity_reused"
    ):
        run_human_precision_gate(
            protocol,
            repository_root=tmp_path,
            annotator_one_path=first,
            annotator_two_path=second,
            prior_resolved_path=prior_path,
            snapshot_root=tmp_path / "snapshots",
        )

@pytest.mark.human_precision_adjudication_regression
def test_real_frozen_package_is_ready_but_awaiting_human_labels() -> None:
    protocol = load_protocol(REAL_PROTOCOL, repository_root=REPOSITORY_ROOT)
    first = run_human_precision_gate(
        protocol,
        repository_root=REPOSITORY_ROOT,
    )
    second = run_human_precision_gate(
        protocol,
        repository_root=REPOSITORY_ROOT,
    )
    assert first == second
    assert first["state"] == "awaiting_labels"
    assert first["exit_code"] == 3
    assert first["statistics"] is None
    assert first["package"]["expected_item_count"] == 439
    assert first["package"]["required_prior_item_count"] == 32
    assert first["execution"] == {
        "network_request_count": 0,
        "llm_request_count": 0,
        "snapshot_write_count": 0,
        "official_scorer_call_count": 0,
    }
