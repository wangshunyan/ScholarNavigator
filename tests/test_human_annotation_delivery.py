from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scholar_agent.evaluation.human_annotation_delivery import (
    DeliveryViolation,
    _make_submission,
    ingest,
    load_delivery_protocol,
    prepare_delivery,
    readiness,
    submission_hash,
    synthetic_dry_run,
    verify_delivery,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "benchmark/human_annotation_delivery_v1_protocol.json"


@pytest.fixture(scope="module")
def delivery(tmp_path_factory):
    protocol = load_delivery_protocol(PROTOCOL_PATH, ROOT)
    package = tmp_path_factory.mktemp("delivery") / "package"
    prepare_delivery(protocol, repository_root=ROOT, output=package)
    return protocol, package


def _locked(package: Path, side: str, path: Path) -> Path:
    _make_submission(package, side, ["relevant", "partially_relevant"], path)
    return path


def test_packages_cover_all_items_and_isolate_aliases(delivery):
    protocol, package = delivery
    report = verify_delivery(protocol, package)
    assert report["item_count_per_annotator"] == 471
    a = json.loads((package / "annotator-A/items.json").read_text())
    b = json.loads((package / "annotator-B/items.json").read_text())
    assert {x["alias"] for x in a}.isdisjoint({x["alias"] for x in b})
    assert [x["alias"] for x in a] != [x["alias"] for x in b]
    assert all(set(x) == {"alias", "query", "title", "abstract", "year"} for x in a + b)
    public = json.dumps(a + b).lower()
    assert all(key not in public for key in ('"sample_id"', '"case_id"', '"gold"', '"arm"', '"rank"'))


def test_generation_is_byte_deterministic(tmp_path):
    protocol = load_delivery_protocol(PROTOCOL_PATH, ROOT)
    one, two = tmp_path / "one", tmp_path / "two"
    prepare_delivery(protocol, repository_root=ROOT, output=one)
    prepare_delivery(protocol, repository_root=ROOT, output=two)
    files_one = [(p.relative_to(one), p.read_bytes()) for p in sorted(x for x in one.rglob("*") if x.is_file())]
    files_two = [(p.relative_to(two), p.read_bytes()) for p in sorted(x for x in two.rglob("*") if x.is_file())]
    assert files_one == files_two


def test_ingest_recovers_current_and_prior(delivery, tmp_path):
    protocol, package = delivery
    a, b = _locked(package, "A", tmp_path / "a.json"), _locked(package, "B", tmp_path / "b.json")
    report = ingest(protocol, package_root=package, annotator_a=a, annotator_b=b)
    assert report["recovered_counts"] == {"A": {"current": 439, "prior": 32}, "B": {"current": 439, "prior": 32}}
    assert report["statistics"] is None


@pytest.mark.parametrize("mutation,code", [
    ("missing", "missing_or_cross_package_alias"),
    ("duplicate", "duplicate_alias"),
    ("illegal", "illegal_label"),
    ("unlocked", "submission_not_locked"),
    ("tampered", "submission_lock_hash_mismatch"),
])
def test_ingest_rejects_bad_or_modified_submissions(delivery, tmp_path, mutation, code):
    protocol, package = delivery
    a, b = _locked(package, "A", tmp_path / "a.json"), _locked(package, "B", tmp_path / "b.json")
    data = json.loads(a.read_text())
    if mutation == "missing": data["labels"].pop()
    elif mutation == "duplicate": data["labels"][-1] = copy.deepcopy(data["labels"][0])
    elif mutation == "illegal": data["labels"][0]["label"] = "maybe"
    elif mutation == "unlocked": data["locked"] = False
    else: data["labels"][0]["notes"] = "changed after lock"
    if mutation in {"missing", "duplicate", "illegal", "unlocked"}: data["labels_sha256"] = submission_hash(data)
    a.write_text(json.dumps(data))
    with pytest.raises(DeliveryViolation, match=code): ingest(protocol, package_root=package, annotator_a=a, annotator_b=b)


def test_cross_package_alias_and_formula_notes_are_rejected(delivery, tmp_path):
    protocol, package = delivery
    a, b = _locked(package, "A", tmp_path / "a.json"), _locked(package, "B", tmp_path / "b.json")
    data = json.loads(a.read_text()); other = json.loads((package / "annotator-B/items.json").read_text())
    data["labels"][0]["alias"] = other[0]["alias"]; data["labels_sha256"] = submission_hash(data); a.write_text(json.dumps(data))
    with pytest.raises(DeliveryViolation, match="missing_or_cross_package_alias"): ingest(protocol, package_root=package, annotator_a=a, annotator_b=b)
    _locked(package, "A", a); data = json.loads(a.read_text()); data["labels"][0]["notes"] = "=1+1"; data["labels_sha256"] = submission_hash(data); a.write_text(json.dumps(data))
    with pytest.raises(DeliveryViolation, match="unsafe_or_oversize_notes"): ingest(protocol, package_root=package, annotator_a=a, annotator_b=b)


def test_static_ui_uses_text_content_and_lock(delivery):
    _, package = delivery
    js = (package / "annotator-A/app.js").read_text()
    assert "textContent" in js and "state.locked" in js and "localStorage" in js
    assert "innerHTML" not in js and "eval(" not in js


def test_synthetic_round_trip_and_real_readiness(delivery):
    protocol, package = delivery
    dry = synthetic_dry_run(protocol, repository_root=ROOT)
    assert dry["synthetic_gate_state"] == "validated"
    assert dry["statistics"] is None and not dry["synthetic_artifacts_persisted"]
    real = readiness(protocol, repository_root=ROOT, package_root=package)
    assert real["exit_code"] == 3
    assert real["state"] == "blocked_awaiting_real_annotators"
    assert real["statistics"] is None


def test_synthetic_round_trip_exposes_read_only_rehearsal_hooks(delivery):
    protocol, _package = delivery
    observed = []
    temporary_roots = []

    def locked(base: Path, annotator_a: Path, annotator_b: Path) -> None:
        temporary_roots.append(base)
        observed.append(("locked", annotator_a.is_file(), annotator_b.is_file()))

    def adjudicated(base: Path, gate: dict[str, object]) -> None:
        assert base == temporary_roots[0]
        observed.append(
            ("adjudicated", gate["state"], gate["statistics"] is not None)
        )

    dry = synthetic_dry_run(
        protocol,
        repository_root=ROOT,
        locked_submission_callback=locked,
        adjudication_callback=adjudicated,
    )
    assert dry["synthetic_gate_state"] == "validated"
    assert observed == [
        ("locked", True, True),
        ("adjudicated", "validated", True),
    ]
    assert not temporary_roots[0].exists()
