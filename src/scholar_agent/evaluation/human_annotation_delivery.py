"""Deterministic offline delivery and recovery for blind human annotation."""

from __future__ import annotations

import hashlib
import json
import random
import re
import tempfile
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scholar_agent.evaluation.human_precision_adjudication import (
    AdjudicationSubmission,
    AdjudicationRow,
    IndependentSubmission,
    LabelRow,
    PackageReference,
    PriorLabelRow,
    PriorResolvedSubmission,
    load_protocol as load_adjudication_protocol,
    run_human_precision_gate,
    validate_package,
)
from scholar_agent.evaluation.precision_annotation import LABELS
from scholar_agent.evaluation.snapshot_resume import sha256_file, stable_hash


CONTRACT = "human_annotation_delivery_v1"
SCHEMA_VERSION = "1"
EXIT_READY = 0
EXIT_VIOLATION = 2
EXIT_AWAITING = 3
EXIT_USAGE = 4
PUBLIC_FIELDS = ("alias", "query", "title", "abstract", "year")
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "sample_id", "item_id", "opaque_id", "arm", "strategy", "rank",
        "source", "score", "gold", "qrels", "case_id", "target_paper",
        "package_role", "prior", "current", "paper_identity",
    }
)


class DeliveryError(RuntimeError):
    pass


class DeliveryViolation(DeliveryError):
    def __init__(self, code: str, path: str = "$") -> None:
        super().__init__(code)
        self.code = code
        self.path = path


class DeliverySubmissionRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str = Field(pattern=r"^item-[0-9a-f]{24}$")
    label: str
    notes: str | None = None


class DeliverySubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    contract: str
    package_id: str
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotator_id: str
    side: str
    locked: bool
    labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels: list[DeliverySubmissionRow]


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def load_delivery_protocol(path: Path, repository_root: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION or data.get("contract") != CONTRACT:
        raise DeliveryError("protocol_version_incompatible")
    if data.get("expected_item_count") != 471 or tuple(data.get("annotation", {}).get("labels", ())) != LABELS:
        raise DeliveryError("protocol_population_or_labels_drift")
    bound = repository_root / data["adjudication_protocol"]["path"]
    if sha256_file(bound) != data["adjudication_protocol"]["sha256"]:
        raise DeliveryError("adjudication_protocol_hash_drift")
    return data


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _population(protocol: Mapping[str, Any], root: Path) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    adj_path = root / protocol["adjudication_protocol"]["path"]
    adj = load_adjudication_protocol(adj_path, repository_root=root)
    context = validate_package(adj, repository_root=root)
    current_root = root / adj["package"]["root"]
    prior_root = root / adj["prior_package"]["root"]
    current_rows = {str(r["sample_id"]): r for r in _jsonl(current_root / adj["package"]["public_samples_path"])}
    prior_rows = {str(r["sample_id"]): r for r in _jsonl(prior_root / adj["prior_package"]["public_samples_path"])}
    required_prior = set(context.required_prior_item_ids)
    items: list[dict[str, Any]] = []
    for item_id in context.item_ids:
        items.append({"stable_key": f"current:{item_id}", "role": "current", "item_id": item_id, "public": current_rows[item_id]})
    for item_id in sorted(required_prior):
        if item_id not in prior_rows:
            raise DeliveryError("prior_public_item_missing")
        items.append({"stable_key": f"prior:{item_id}", "role": "prior", "item_id": item_id, "public": prior_rows[item_id]})
    if len(items) != protocol["expected_item_count"] or len({x["stable_key"] for x in items}) != len(items):
        raise DeliveryError("delivery_population_not_closed")
    return items, context, adj


def _alias(seed: str, stable_key: str, length: int) -> str:
    return "item-" + hashlib.sha256(f"{seed}\0{stable_key}".encode()).hexdigest()[:length]


def _asset_hashes(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name != "bundle.json")
    ]


def prepare_delivery(protocol: Mapping[str, Any], *, repository_root: Path, output: Path, replace_existing: bool = False) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()) and not replace_existing:
        raise DeliveryViolation("output_directory_not_empty")
    output.mkdir(parents=True, exist_ok=True)
    items, _, _ = _population(protocol, repository_root)
    rubric = json.loads((repository_root / "benchmark/lexical_normalization_record160_precision_annotation/public/annotation_schema.json").read_text())
    mapping: dict[str, Any] = {"schema_version": "1", "contract": CONTRACT, "items": []}
    side_summaries = []
    for spec in protocol["annotators"]:
        side = spec["side"]
        aliases = {x["stable_key"]: _alias(protocol["alias"]["seeds"][side], x["stable_key"], protocol["alias"]["length"]) for x in items}
        ordered = list(items)
        random.Random(spec["order_seed"]).shuffle(ordered)
        public = []
        for item in ordered:
            row = item["public"]
            public.append({"alias": aliases[item["stable_key"]], "query": row.get("query"), "title": row.get("title"), "abstract": row.get("abstract"), "year": row.get("year")})
            mapping["items"].append({"side": side, "alias": aliases[item["stable_key"]], "role": item["role"], "item_id": item["item_id"], "content_sha256": stable_hash({k: row.get(k) for k in ("query", "title", "abstract", "year")})})
        side_root = output / f"annotator-{side}"
        package = {"schema_version": "1", "contract": CONTRACT, "package_id": f"record160-471-{side}", "annotator_id": spec["annotator_id"], "side": side, "item_count": len(public), "rubric_sha256": protocol["rubric"]["current_sha256"], "items_sha256": stable_hash(public), "labels": list(LABELS)}
        package["package_sha256"] = stable_hash(package)
        write_json(side_root / "package.json", package)
        write_json(side_root / "rubric.json", rubric)
        write_json(side_root / "items.json", public)
        (side_root / "package-data.js").write_text("window.SPAR_PACKAGE=" + json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + ";\n", encoding="utf-8")
        (side_root / "items-data.js").write_text("window.SPAR_ITEMS=" + json.dumps(public, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + ";\n", encoding="utf-8")
        (side_root / "index.html").write_text(_HTML, encoding="utf-8")
        (side_root / "app.js").write_text(_JS, encoding="utf-8")
        side_summaries.append({"side": side, "package_id": package["package_id"], "package_sha256": package["package_sha256"], "items_sha256": package["items_sha256"]})
    if len({x["alias"] for x in mapping["items"]}) != len(mapping["items"]):
        raise DeliveryViolation("alias_collision")
    mapping["mapping_sha256"] = stable_hash(mapping["items"])
    write_json(output / "operator" / "mapping.json", mapping)
    assets = _asset_hashes(output)
    bundle = {"schema_version": "1", "contract": CONTRACT, "item_count_per_annotator": len(items), "annotators": side_summaries, "asset_count": len(assets), "assets": assets, "execution": protocol["execution"], "statistics": None}
    bundle["bundle_sha256"] = stable_hash(bundle)
    write_json(output / "bundle.json", bundle)
    verify_delivery(protocol, output)
    return {"schema_version": "1", "contract": CONTRACT, "state": "delivery_ready", "exit_code": 0, "item_count_per_annotator": 471, "bundle_sha256": bundle["bundle_sha256"], "statistics": None, "execution": protocol["execution"]}


def _walk_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            found.add(k.lower())
            found |= _walk_keys(v)
    elif isinstance(value, list):
        for v in value: found |= _walk_keys(v)
    return found


def verify_delivery(protocol: Mapping[str, Any], package_root: Path) -> dict[str, Any]:
    bundle = json.loads((package_root / "bundle.json").read_text())
    assets = _asset_hashes(package_root)
    if assets != bundle.get("assets") or len(assets) != bundle.get("asset_count"):
        raise DeliveryViolation("asset_manifest_mismatch")
    check = dict(bundle); digest = check.pop("bundle_sha256", None)
    if digest != stable_hash(check): raise DeliveryViolation("bundle_hash_mismatch")
    aliases: dict[str, set[str]] = {}
    for spec in protocol["annotators"]:
        side = spec["side"]; base = package_root / f"annotator-{side}"
        package = json.loads((base / "package.json").read_text()); rows = json.loads((base / "items.json").read_text())
        if len(rows) != 471 or any(set(r) != set(PUBLIC_FIELDS) for r in rows): raise DeliveryViolation("public_population_or_schema_invalid")
        if _walk_keys(rows) & FORBIDDEN_PUBLIC_KEYS: raise DeliveryViolation("public_field_leak")
        if stable_hash(rows) != package.get("items_sha256"): raise DeliveryViolation("public_items_hash_mismatch")
        aliases[side] = {r["alias"] for r in rows}
        if len(aliases[side]) != 471: raise DeliveryViolation("public_alias_duplicate")
        text = (base / "app.js").read_text() + (base / "index.html").read_text()
        if re.search(r"\b(innerHTML|eval\s*\(|document\.write\s*\()", text): raise DeliveryViolation("unsafe_static_interface")
    if aliases["A"] & aliases["B"]: raise DeliveryViolation("cross_annotator_alias_linkable")
    mapping = json.loads((package_root / "operator/mapping.json").read_text())
    if len(mapping.get("items", [])) != 942 or stable_hash(mapping["items"]) != mapping.get("mapping_sha256"): raise DeliveryViolation("operator_mapping_invalid")
    return {"schema_version": "1", "contract": CONTRACT, "state": "delivery_ready", "exit_code": 0, "item_count_per_annotator": 471, "bundle_sha256": bundle["bundle_sha256"], "statistics": None, "execution": protocol["execution"]}


def submission_hash(payload: Mapping[str, Any]) -> str:
    data = dict(payload); data.pop("labels_sha256", None)
    return hashlib.sha256(canonical_bytes(data)).hexdigest()


def _safe_notes(value: str | None, maximum: int) -> None:
    if value is None: return
    if len(value) > maximum or re.match(r"^[\s\t]*[=+\-@]", value): raise DeliveryViolation("unsafe_or_oversize_notes")


def load_submission(path: Path, *, package_root: Path, side: str, protocol: Mapping[str, Any]) -> DeliverySubmission:
    try: sub = DeliverySubmission.model_validate_json(path.read_text())
    except (OSError, ValidationError) as exc: raise DeliveryViolation("submission_schema_invalid") from exc
    package = json.loads((package_root / f"annotator-{side}/package.json").read_text())
    if sub.contract != CONTRACT or sub.schema_version != "1" or sub.side != side or sub.annotator_id != package["annotator_id"] or sub.package_id != package["package_id"] or sub.package_sha256 != package["package_sha256"]: raise DeliveryViolation("submission_package_or_annotator_mismatch")
    if not sub.locked: raise DeliveryViolation("submission_not_locked")
    if sub.labels_sha256 != submission_hash(sub.model_dump(mode="json")): raise DeliveryViolation("submission_lock_hash_mismatch")
    aliases = [x.alias for x in sub.labels]
    expected = {x["alias"] for x in json.loads((package_root / f"annotator-{side}/items.json").read_text())}
    if len(aliases) != len(set(aliases)): raise DeliveryViolation("duplicate_alias")
    if set(aliases) != expected: raise DeliveryViolation("missing_or_cross_package_alias")
    for row in sub.labels:
        if row.label not in LABELS: raise DeliveryViolation("illegal_label")
        _safe_notes(row.notes, protocol["annotation"]["notes_max_length"])
    return sub


def ingest(protocol: Mapping[str, Any], *, package_root: Path, annotator_a: Path, annotator_b: Path, output: Path | None = None) -> dict[str, Any]:
    verify_delivery(protocol, package_root)
    submissions = {s: load_submission(p, package_root=package_root, side=s, protocol=protocol) for s, p in (("A", annotator_a), ("B", annotator_b))}
    mapping = json.loads((package_root / "operator/mapping.json").read_text())
    lookup = {(x["side"], x["alias"]): x for x in mapping["items"]}
    recovered: dict[str, dict[str, list[dict[str, Any]]]] = {"A": {"current": [], "prior": []}, "B": {"current": [], "prior": []}}
    for side, sub in submissions.items():
        for row in sub.labels:
            target = lookup[(side, row.alias)]
            recovered[side][target["role"]].append({"item_id": target["item_id"], "label": row.label, "notes": row.notes})
        for role in ("current", "prior"): recovered[side][role].sort(key=lambda x: x["item_id"])
    if output:
        output.mkdir(parents=True, exist_ok=True)
        for side in ("A", "B"):
            for role in ("current", "prior"): write_json(output / f"{side.lower()}_{role}.json", {"schema_version": "1", "contract": CONTRACT, "annotator_id": submissions[side].annotator_id, "role": role, "labels": recovered[side][role]})
    return {"schema_version": "1", "contract": CONTRACT, "state": "delivery_ready", "exit_code": 0, "recovered_counts": {s: {r: len(v) for r, v in roles.items()} for s, roles in recovered.items()}, "statistics": None, "execution": protocol["execution"], "recovered": recovered}


def _make_submission(package_root: Path, side: str, labels: list[str], path: Path) -> None:
    package = json.loads((package_root / f"annotator-{side}/package.json").read_text())
    rows = json.loads((package_root / f"annotator-{side}/items.json").read_text())
    payload = {"schema_version": "1", "contract": CONTRACT, "package_id": package["package_id"], "package_sha256": package["package_sha256"], "annotator_id": package["annotator_id"], "side": side, "locked": True, "labels": [{"alias": row["alias"], "label": labels[i % len(labels)], "notes": ""} for i, row in enumerate(rows)]}
    payload["labels_sha256"] = submission_hash(payload); write_json(path, payload)


def synthetic_dry_run(
    protocol: Mapping[str, Any],
    *,
    repository_root: Path,
    locked_submission_callback: Callable[[Path, Path, Path], None] | None = None,
    adjudication_callback: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="synthetic_rehearsal_only-human-") as td:
        base = Path(td); package = base / "package"; prepare_delivery(protocol, repository_root=repository_root, output=package)
        a = base / "a.json"; b = base / "b.json"
        _make_submission(package, "A", [LABELS[0], LABELS[1], LABELS[2]], a)
        _make_submission(package, "B", [LABELS[0], LABELS[1], LABELS[3]], b)
        if locked_submission_callback is not None:
            locked_submission_callback(base, a, b)
        recovered = ingest(protocol, package_root=package, annotator_a=a, annotator_b=b)
        adj_protocol = load_adjudication_protocol(repository_root / protocol["adjudication_protocol"]["path"], repository_root=repository_root)
        context = validate_package(adj_protocol, repository_root=repository_root)
        current_ref = context.reference; prior_ref = context.prior_reference
        paths = {k: base / f"{k}.json" for k in ("one", "two", "adjudication", "prior")}
        current = {}
        for side, round_id, name in (("A", "independent_1", "one"), ("B", "independent_2", "two")):
            obj = IndependentSubmission(contract="human_precision_adjudication_v1", package=current_ref, round_id=round_id, annotator_id=f"anon-synthetic-{side}", labels=[LabelRow(**x) for x in recovered["recovered"][side]["current"]])
            current[side] = {x.item_id: x.label for x in obj.labels}; write_json(paths[name], obj.model_dump(mode="json"))
        disagreements = [i for i in context.item_ids if current["A"][i] != current["B"][i]]
        adjudication = AdjudicationSubmission(contract="human_precision_adjudication_v1", package=current_ref, round_id="adjudication", adjudicator_id="anon-synthetic-adjudicator", decisions=[AdjudicationRow(item_id=i, final_label=current["A"][i], rationale="synthetic fixture") for i in disagreements])
        write_json(paths["adjudication"], adjudication.model_dump(mode="json"))
        prior_rows = []
        prior_maps = {s: {x["item_id"]: x["label"] for x in recovered["recovered"][s]["prior"]} for s in ("A", "B")}
        for item_id in context.required_prior_item_ids:
            x, y = prior_maps["A"][item_id], prior_maps["B"][item_id]
            prior_rows.append(PriorLabelRow(item_id=item_id, annotator_1_label=x, annotator_2_label=y, final_label=x, resolution="annotator_agreement" if x == y else "adjudicated"))
        prior = PriorResolvedSubmission(contract="human_precision_adjudication_v1", package=prior_ref, round_id="resolved_prior_package", annotator_1_id="anon-synthetic-A", annotator_2_id="anon-synthetic-B", adjudicator_id="anon-synthetic-adjudicator", labels=prior_rows)
        write_json(paths["prior"], prior.model_dump(mode="json"))
        gate = run_human_precision_gate(adj_protocol, repository_root=repository_root, annotator_one_path=paths["one"], annotator_two_path=paths["two"], adjudication_path=paths["adjudication"], prior_resolved_path=paths["prior"])
        if gate["state"] != "validated": raise DeliveryViolation("synthetic_adjudication_gate_not_validated")
        if adjudication_callback is not None:
            adjudication_callback(base, gate)
    return {"schema_version": "1", "contract": CONTRACT, "state": "delivery_ready", "exit_code": 0, "synthetic_item_count": 471, "synthetic_gate_state": "validated", "synthetic_artifacts_persisted": False, "statistics": None, "execution": protocol["execution"]}


def readiness(protocol: Mapping[str, Any], *, repository_root: Path, package_root: Path | None = None) -> dict[str, Any]:
    _population(protocol, repository_root)
    if package_root: verify_delivery(protocol, package_root)
    return {"schema_version": "1", "contract": CONTRACT, "state": "blocked_awaiting_real_annotators", "exit_code": 3, "item_count": 471, "real_annotator_submission_count": 0, "formal_validation_complete": False, "statistics": None, "execution": protocol["execution"]}


_HTML = """<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>SPAR blind annotation</title></head><body><main><h1>盲化论文相关性标注</h1><p id=\"progress\"></p><article><h2 id=\"title\"></h2><p id=\"query\"></p><p id=\"abstract\"></p><p id=\"year\"></p></article><fieldset id=\"labels\"></fieldset><textarea id=\"notes\" maxlength=\"1000\"></textarea><button id=\"prev\">上一项</button><button id=\"next\">下一项</button><button id=\"save\">保存进度</button><button id=\"lock\">锁定并导出</button></main><script src=\"package-data.js\"></script><script src=\"items-data.js\"></script><script src=\"app.js\"></script></body></html>"""

_JS = r"""'use strict';
let pkg, items, state={index:0,labels:{},locked:false};
const el=id=>document.getElementById(id);
const safeNote=s=>!/^\s*[=+\-@]/.test(s);
const canonical=v=>Array.isArray(v)?'['+v.map(canonical).join(',')+']':(v&&typeof v==='object'?'{'+Object.keys(v).sort().map(k=>JSON.stringify(k)+':'+canonical(v[k])).join(',')+'}':JSON.stringify(v));
async function digest(text){const b=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text));return [...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,'0')).join('');}
function key(){return 'spar-human-annotation:'+pkg.package_sha256;}
function render(){const x=items[state.index];el('title').textContent=x.title||'';el('query').textContent=x.query||'';el('abstract').textContent=x.abstract||'';el('year').textContent=x.year==null?'':String(x.year);el('progress').textContent=`${state.index+1}/${items.length}`;el('notes').value=(state.labels[x.alias]||{}).notes||'';el('notes').disabled=state.locked;for(const r of document.querySelectorAll('input[name=label]')){r.checked=(state.labels[x.alias]||{}).label===r.value;r.disabled=state.locked;}}
function capture(){if(state.locked)return;const x=items[state.index],checked=document.querySelector('input[name=label]:checked'),notes=el('notes').value;if(!safeNote(notes))throw new Error('unsafe_notes');state.labels[x.alias]={label:checked?checked.value:null,notes};}
function save(){capture();localStorage.setItem(key(),JSON.stringify(state));}
async function lock(){capture();if(items.some(x=>!state.labels[x.alias]||!pkg.labels.includes(state.labels[x.alias].label)))throw new Error('incomplete');state.locked=true;const payload={schema_version:'1',contract:'human_annotation_delivery_v1',package_id:pkg.package_id,package_sha256:pkg.package_sha256,annotator_id:pkg.annotator_id,side:pkg.side,locked:true,labels:items.map(x=>({alias:x.alias,label:state.labels[x.alias].label,notes:state.labels[x.alias].notes||''}))};payload.labels_sha256=await digest(canonical(payload)+'\n');localStorage.setItem(key(),JSON.stringify(state));const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.download='locked-submission.json';a.href=URL.createObjectURL(blob);a.click();URL.revokeObjectURL(a.href);render();}
pkg=window.SPAR_PACKAGE;items=window.SPAR_ITEMS;if(!pkg||!items)throw new Error('package_data_missing');const old=localStorage.getItem(key());if(old)state=JSON.parse(old);for(const label of pkg.labels){const input=document.createElement('input');input.type='radio';input.name='label';input.value=label;const span=document.createElement('span');span.textContent=label;el('labels').append(input,span,document.createElement('br'));}el('prev').onclick=()=>{save();state.index=Math.max(0,state.index-1);render();};el('next').onclick=()=>{save();state.index=Math.min(items.length-1,state.index+1);render();};el('save').onclick=save;el('lock').onclick=lock;render();
"""
