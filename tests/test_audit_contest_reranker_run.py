from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_contest_reranker_run import main


def test_full_reranker_audit_accepts_completed_1000_rows(tmp_path: Path) -> None:
    run = tmp_path / "contest_full_dense_reranker_v4"
    generation = run / ".run_commits" / "generations" / "generation-00001002"
    generation.mkdir(parents=True)
    (generation / "RUN_COMPLETED").write_text("{}", encoding="utf-8")
    (run / "results.jsonl").write_text(
        "\n".join(
            json.dumps({
                "case_id": f"q-{index}",
                "diagnostics": {
                    "local_model_batch_count": 1,
                    "local_model_fallback_count": 0,
                    "local_model_inference_success_count": 1,
                    "local_model_candidate_count": 120,
                    "local_model_prompt_version": "qwen3-reranker-v1",
                    "local_model_device": "cuda:1",
                    "local_model_max_length": 2048,
                    "local_model_fingerprint": "fingerprint",
                    "local_model_latency_seconds": 0.5,
                    "local_model_batch_size": 8,
                    "local_model_candidate_limit": 120,
                    "local_model_peak_vram_bytes": 1,
                },
            })
            for index in range(1000)
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"
    assert main(["--run", str(run), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
