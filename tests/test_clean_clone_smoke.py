from pathlib import Path

from scripts.check_clean_clone_smoke import run_smoke


ROOT = Path(__file__).resolve().parents[1]


def test_clean_clone_smoke_exports_structured_offline_result() -> None:
    report = run_smoke(ROOT)
    assert report["status"] == "ready"
    assert report["network_request_count"] == 0
    assert report["dotenv_read"] is False
    assert report["dependency_inputs"] == {
        "requirements_txt": True,
        "frontend_package_json": True,
        "frontend_lockfile": True,
    }
    export = report["local_bm25"]
    assert export["schema_version"] == "offline-search-result-v1"
    assert export["result_count"] == 5
    assert [row["rank"] for row in export["results"]] == [1, 2, 3, 4, 5]
    assert all("gold" not in row for row in export["results"])
