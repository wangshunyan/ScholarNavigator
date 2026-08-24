from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_query_manifest_is_gold_blind_and_matches_curated_set() -> None:
    path = ROOT / "docs" / "contest" / "demo-queries.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert [row["case_id"] for row in rows] == [
        "demo_01",
        "demo_02",
        "demo_03",
        "demo_04",
        "demo_05",
    ]
    assert all(isinstance(row.get("query"), str) and row["query"].strip() for row in rows)
    assert all(row.get("top_k") == 5 for row in rows)
    assert all(row.get("run_profile") == "balanced" for row in rows)
    serialized = path.read_text(encoding="utf-8").lower()
    assert "gold" not in serialized
    assert "qrels" not in serialized

