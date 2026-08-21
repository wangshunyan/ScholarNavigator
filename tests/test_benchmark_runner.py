from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_benchmark
from scholar_agent.connectors.local_bm25 import (
    LocalBM25Config,
    LocalBM25FieldConfig,
)
from scholar_agent.agents.judgement_config import (
    CALIBRATED_RULES_V1_CONFIG,
    judgement_config_hash,
)
from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.core.diagnostics_schemas import ConnectorDiagnostics
from scholar_agent.core.paper_schemas import Paper, PaperIdentifiers
from scholar_agent.core.pipeline_diagnostics import (
    CandidateProvenance,
    DiagnosticCandidate,
    StageCandidateSnapshot,
)
from scholar_agent.core.search_schemas import (
    EvidenceItem,
    JudgementResult,
    QueryAnalysis,
    QueryConstraint,
    RankedPaper,
    RerankScoreBreakdown,
    SearchPlan,
    SearchSubquery,
)
from scholar_agent.evaluation.resource_accounting import (
    ResourceLedgerV1,
    opaque_resource_identity,
    validate_resource_ledger,
)
from scholar_agent.services.search_service import SearchServiceOutput


def test_limit_and_offset_follow_source_order(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=4)
    service = FakeService()
    result = run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="subset", offset=1, limit=2),
        service=service,
    )

    assert [row["case_id"] for row in result.result_rows] == ["case-1", "case-2"]
    assert result.config["selection_order"] == "source_order"
    assert result.config["case_ids"] == ["case-1", "case-2"]


def test_query_adapter_policy_is_recorded_and_passed_to_search_service(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=1)
    service = FakeService()

    result = run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="adapter-policy").model_copy(
            update={"query_adapter_policy": "safe_original"}
        ),
        service=service,
    )

    assert result.config["query_adapter_policy"] == "safe_original"
    assert service.kwargs[0]["query_adapter_policy"] == "safe_original"


def test_benchmark_default_and_cli_support_adaptive(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=1)

    assert _options(
        tmp_path, dataset, run_id="adaptive-default"
    ).query_adapter_policy == "adaptive"
    parsed = run_benchmark._parser().parse_args(  # noqa: SLF001
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "adaptive-cli",
            "--query-adapter-policy",
            "adaptive",
        ]
    )
    assert parsed.query_adapter_policy == "adaptive"


def test_local_bm25_cli_is_default_off_and_records_public_index_config(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=1)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "1", "title": "Local paper", "text": "abstract"})
        + "\n",
        encoding="utf-8",
    )
    parsed = run_benchmark._parser().parse_args(  # noqa: SLF001
        ["--dataset", "auto_scholar_query", "--run-id", "default-sources"]
    )
    assert "local_bm25" not in parsed.sources

    options = _options(tmp_path, dataset, run_id="local-config").model_copy(
        update={
            "sources": ["local_bm25"],
            "local_bm25_config": LocalBM25Config(
                corpus_path=corpus,
                cache_dir=tmp_path / "cache",
                fields=LocalBM25FieldConfig(
                    abstract="text",
                    document_id_identity="s2orc_corpus_id",
                ),
            ),
        }
    )
    result = run_benchmark.run_benchmark(options, service=FakeService())

    assert result.config["sources"] == ["local_bm25"]
    assert result.config["local_bm25"]["document_count"] == 1
    assert result.config["local_bm25"]["parameters"] == {
        "k1": 1.5,
        "b": 0.75,
        "epsilon": 0.25,
    }
    assert result.config["local_bm25"]["evaluator_data_access"] is False


def test_resume_ignores_local_bm25_operational_cache_state(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=1)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "1", "title": "Local paper", "text": "abstract"})
        + "\n",
        encoding="utf-8",
    )
    options = _options(tmp_path, dataset, run_id="local-resume").model_copy(
        update={
            "sources": ["local_bm25"],
            "local_bm25_config": LocalBM25Config(
                corpus_path=corpus,
                cache_dir=tmp_path / "cache",
                fields=LocalBM25FieldConfig(
                    abstract="text",
                    document_id_identity="s2orc_corpus_id",
                ),
            ),
        }
    )

    first = run_benchmark.run_benchmark(options, service=FakeService())
    resumed = run_benchmark.run_benchmark(
        options.model_copy(update={"resume": True}),
        service=FakeService(),
    )

    assert first.config["local_bm25"]["index_cache_hit"] is False
    assert resumed.config["local_bm25"]["index_cache_hit"] is False
    assert first.config["resume_signature"] == resumed.config["resume_signature"]
    assert [row["case_id"] for row in resumed.result_rows] == ["case-0"]


def test_ranking_policy_is_default_off_recorded_and_passed_to_service(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=1)
    assert _options(tmp_path, dataset, run_id="ranking-default").ranking_policy == (
        "current_rules"
    )
    service = FakeService()
    options = _options(tmp_path, dataset, run_id="rrf").model_copy(
        update={"ranking_policy": "rrf_fusion"}
    )

    result = run_benchmark.run_benchmark(options, service=service)
    parsed = run_benchmark._parser().parse_args(  # noqa: SLF001
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "rrf-cli",
            "--ranking-policy",
            "rrf_fusion",
        ]
    )

    assert result.config["ranking_policy"] == "rrf_fusion"
    assert service.kwargs[0]["ranking_policy"] == "rrf_fusion"
    assert parsed.ranking_policy == "rrf_fusion"
    assert run_benchmark._ablation_group_name(options) == "baseline"  # noqa: SLF001


def test_query_evolution_policy_is_recorded_and_passed_to_service(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=1)
    service = FakeService()
    options = _options(tmp_path, dataset, run_id="coverage-gap").model_copy(
        update={
            "enable_query_evolution": True,
            "query_evolution_policy": "coverage_gap",
        }
    )

    result = run_benchmark.run_benchmark(options, service=service)
    parsed = run_benchmark._parser().parse_args(  # noqa: SLF001
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "policy-cli",
            "--enable-query-evolution",
            "--query-evolution-policy",
            "seed_expansion",
        ]
    )

    assert result.config["query_evolution_policy"] == "coverage_gap"
    assert service.kwargs[0]["query_evolution_policy"] == "coverage_gap"
    assert parsed.query_evolution_policy == "seed_expansion"
    assert run_benchmark._ablation_group_name(options) == (  # noqa: SLF001
        "query_evolution_coverage_gap"
    )


def test_query_planning_policy_is_recorded_passed_and_namespaced(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=1)
    service = FakeService()
    options = _options(tmp_path, dataset, run_id="facet-planner").model_copy(
        update={"query_planning_policy": "facet_balanced"}
    )

    result = run_benchmark.run_benchmark(options, service=service)
    parsed = run_benchmark._parser().parse_args(  # noqa: SLF001
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "facet-cli",
            "--query-planning-policy",
            "facet_balanced",
        ]
    )

    assert result.config["query_planning_policy"] == "facet_balanced"
    assert result.config["query_planner_version"] == "1.9.0"
    assert service.kwargs[0]["query_planning_policy"] == "facet_balanced"
    assert parsed.query_planning_policy == "facet_balanced"
    assert run_benchmark._ablation_group_name(options) == "facet_balanced"  # noqa: SLF001
    assert run_benchmark._ablation_group_name(  # noqa: SLF001
        options.model_copy(
            update={
                "enable_query_evolution": True,
                "query_evolution_policy": "coverage_gap",
                "enable_refchain": True,
            }
        )
    ) == "facet_balanced_query_evolution_coverage_gap_plus_refchain"


def test_judgement_policy_config_and_hash_are_recorded_and_passed(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=1)
    config_path = tmp_path / "judgement.json"
    config_path.write_text(
        CALIBRATED_RULES_V1_CONFIG.model_dump_json(indent=2),
        encoding="utf-8",
    )
    service = FakeService()
    options = _options(tmp_path, dataset, run_id="judgement-policy").model_copy(
        update={
            "judgement_policy": "calibrated_rules_v1",
            "judgement_config_path": config_path,
        }
    )

    result = run_benchmark.run_benchmark(options, service=service)

    assert result.config["judgement_policy"] == "calibrated_rules_v1"
    assert result.config["judgement_config"] == (
        CALIBRATED_RULES_V1_CONFIG.model_dump(mode="json")
    )
    assert result.config["judgement_config_hash"] == judgement_config_hash(
        CALIBRATED_RULES_V1_CONFIG
    )
    assert service.kwargs[0]["judgement_policy"] == "calibrated_rules_v1"
    assert service.kwargs[0]["judgement_config"] == CALIBRATED_RULES_V1_CONFIG


def test_resume_rejects_changed_judgement_config(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=1)
    config_path = tmp_path / "judgement.json"
    config_path.write_text(
        CALIBRATED_RULES_V1_CONFIG.model_dump_json(indent=2),
        encoding="utf-8",
    )
    options = _options(tmp_path, dataset, run_id="judgement-resume").model_copy(
        update={
            "judgement_policy": "calibrated_rules_v1",
            "judgement_config_path": config_path,
        }
    )
    run_benchmark.run_benchmark(options, service=FakeService())
    changed = CALIBRATED_RULES_V1_CONFIG.model_copy(
        update={"title_topic_weight": 0.11}
    )
    config_path.write_text(changed.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="resume config is incompatible"):
        run_benchmark.run_benchmark(
            options.model_copy(update={"resume": True}),
            service=FakeService(),
        )


def test_llm_semantic_cli_is_supported_but_live_run_requires_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = run_benchmark._parser().parse_args(  # noqa: SLF001
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "llm-cli",
            "--query-planning-policy",
            "llm_semantic",
            "--llm-mode",
            "record",
            "--llm-snapshot-dir",
            str(tmp_path / "llm"),
        ]
    )
    assert parsed.query_planning_policy == "llm_semantic"
    assert parsed.llm_mode == "record"

    monkeypatch.setenv("SCHOLAR_AGENT_LLM_PROVIDER", "disabled")
    dataset = _dataset(tmp_path, count=1)
    options = _options(tmp_path, dataset, run_id="llm-disabled").model_copy(
        update={"query_planning_policy": "llm_semantic"}
    )
    with pytest.raises(ValueError, match="requires an available LLM provider"):
        run_benchmark.run_benchmark(options)


def test_llm_constrained_rewrite_cli_is_supported_and_namespaced(
    tmp_path: Path,
) -> None:
    parsed = run_benchmark._parser().parse_args(  # noqa: SLF001
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "rewrite-cli",
            "--query-planning-policy",
            "llm_constrained_rewrite",
            "--llm-mode",
            "replay",
            "--llm-snapshot-dir",
            str(tmp_path / "llm"),
        ]
    )
    options = _options(tmp_path, _dataset(tmp_path, count=1), run_id="rewrite")
    options = options.model_copy(
        update={"query_planning_policy": "llm_constrained_rewrite"}
    )

    assert parsed.query_planning_policy == "llm_constrained_rewrite"
    assert run_benchmark._ablation_group_name(options) == (  # noqa: SLF001
        "llm_constrained_rewrite"
    )


def test_runner_writes_required_outputs_and_uses_shared_metrics(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=2)
    result = run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="outputs"),
        service=FakeService(),
    )
    run_dir = result.run_dir

    assert {path.name for path in run_dir.iterdir()} == {
        ".run_commits",
        "config.json",
        "dataset_report.json",
        "results.jsonl",
        "metrics.json",
        "failures.jsonl",
        "summary.md",
    }
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["success_only_metrics"]["f1_at_k"]["5"] == pytest.approx(1 / 3)
    assert metrics["end_to_end_metrics"]["recall_at_k"]["20"] == 1.0
    assert metrics["success_only_metrics"]["mrr"] == 1.0
    assert metrics["benchmark_statistics"]["average_api_calls"] == 1.0
    assert metrics["benchmark_statistics"]["average_candidate_count"] == 1.0
    assert metrics["benchmark_statistics"]["average_final_result_count"] == 1.0
    assert "小规模 smoke" in (run_dir / "summary.md").read_text(encoding="utf-8")


def test_resource_ledger_uses_contract_compliant_query_identities(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=2)
    result = run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="resource-ledger").model_copy(
            update={"enable_resource_ledger": True}
        ),
        service=FakeService(),
    )

    ledger = ResourceLedgerV1.model_validate_json(
        (result.run_dir / "resource_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger.expected_query_identities == [
        opaque_resource_identity("query", "case-0"),
        opaque_resource_identity("query", "case-1"),
    ]
    assert all(len(identity) == 64 for identity in ledger.expected_query_identities)
    assert validate_resource_ledger(
        ledger, authoritative_records=result.result_rows
    )["status"] == "passed"


def test_runner_uses_shared_result_selection_policy(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=1)
    highly_only = run_benchmark.run_benchmark(
        _options(
            tmp_path,
            dataset,
            run_id="high-only",
            result_policy="highly_only",
        ),
        service=FakeService(category="partially_relevant"),
    )
    with_partial = run_benchmark.run_benchmark(
        _options(
            tmp_path,
            dataset,
            run_id="with-partial",
            result_policy="highly_and_partial",
        ),
        service=FakeService(category="partially_relevant"),
    )

    assert highly_only.metrics["end_to_end_metrics"]["recall_at_k"]["5"] == 0.0
    assert with_partial.metrics["end_to_end_metrics"]["recall_at_k"]["5"] == 1.0


def test_failed_case_is_zero_in_end_to_end_and_written_to_failures(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=2)
    result = run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="failure"),
        service=FakeService(fail_queries={"query-1"}),
    )

    assert result.metrics["success_only_metrics"]["recall_at_k"]["5"] == 1.0
    assert result.metrics["end_to_end_metrics"]["recall_at_k"]["5"] == 0.5
    assert result.metrics["case_statistics"]["failed_case_rate"] == 0.5
    failures = _read_jsonl(result.run_dir / "failures.jsonl")
    assert failures[0]["case_id"] == "case-1"
    assert failures[0]["error_type"] == "RuntimeError"


def test_resume_skips_success_and_retries_failed_without_duplicates(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=2)
    first_service = FakeService(fail_queries={"query-1"})
    options = _options(tmp_path, dataset, run_id="resume")
    run_benchmark.run_benchmark(options, service=first_service)

    retry_service = FakeService()
    resumed = run_benchmark.run_benchmark(
        options.model_copy(update={"resume": True}),
        service=retry_service,
    )

    assert retry_service.calls == ["query-1"]
    assert [row["case_id"] for row in resumed.result_rows] == ["case-0", "case-1"]
    assert all(row["status"] == "succeeded" for row in resumed.result_rows)
    assert len(_read_jsonl(resumed.run_dir / "results.jsonl")) == 2
    assert _read_jsonl(resumed.run_dir / "failures.jsonl") == []


def test_resume_uses_committed_generation_instead_of_torn_compatibility_view(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=2)
    options = _options(tmp_path, dataset, run_id="committed-resume")
    first = run_benchmark.run_benchmark(options, service=FakeService())
    (first.run_dir / "results.jsonl").write_text('{"torn":', encoding="utf-8")

    service = FakeService()
    resumed = run_benchmark.run_benchmark(
        options.model_copy(update={"resume": True}), service=service
    )

    assert service.calls == []
    assert [row["case_id"] for row in resumed.result_rows] == ["case-0", "case-1"]
    assert len(_read_jsonl(first.run_dir / "results.jsonl")) == 2


def test_resume_rejects_incompatible_config(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=1)
    options = _options(tmp_path, dataset, run_id="incompatible")
    run_benchmark.run_benchmark(options, service=FakeService())

    with pytest.raises(ValueError, match="resume config is incompatible"):
        run_benchmark.run_benchmark(
            options.model_copy(update={"resume": True, "top_k": 10}),
            service=FakeService(),
        )


def test_config_records_llm_prompt_budget_and_code_metadata(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, count=1)
    result = run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="metadata"),
        service=FakeService(),
    )
    config = result.config

    assert config["llm"]["llm_enabled"] is False
    assert config["llm"]["requested"] is False
    assert {item["name"] for item in config["prompts"]} == {
            "llm_constrained_rewrite",
            "llm_feedback_evolution",
            "llm_relevance_adjudicator_v1_1",
        "llm_relevance_judge_v1_1",
        "judge_backend_qualification_v1",
        "llm_query_planning",
        "query_understanding",
        "relevance_judgement",
    }
    assert all(len(item["hash"]) == 64 for item in config["prompts"])
    assert config["budgets"]["max_search_rounds"] == 2
    assert len(config["dataset_sha256"]) == 64
    assert len(config["runtime_code_hash"]) == 64
    assert "commit" in config["code"]
    assert "dirty" in config["code"]
    assert isinstance(config["execution"]["process_id"], int)
    assert config["execution"]["process_id"] > 0
    assert isinstance(config["execution"]["launch_command"], list)
    assert config["execution"]["log_path"] is None


def test_outputs_redact_api_keys(tmp_path: Path, monkeypatch) -> None:
    secret = "do-not-write-this-secret"
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_API_KEY", secret)
    dataset = _dataset(tmp_path, count=1)
    result = run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="redacted"),
        service=FakeService(error_message=f"api_key={secret}"),
    )

    combined = "".join(
        path.read_text(encoding="utf-8")
        for path in result.run_dir.iterdir()
        if path.is_file()
    )
    assert secret not in combined
    assert "[REDACTED]" in combined


def test_search_service_receives_query_and_runtime_options_but_not_gold(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=1)
    service = FakeService()
    run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="no-gold"),
        service=service,
    )

    assert service.calls == ["query-0"]
    assert service.kwargs[0]["sources_override"] == ["openalex"]
    assert "gold" not in service.kwargs[0]
    assert "gold_papers" not in service.kwargs[0]


def test_diagnostics_outputs_and_resume_do_not_duplicate_gold_rows(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, count=2)
    options = _options(tmp_path, dataset, run_id="diagnostics").model_copy(
        update={"diagnostics": True}
    )
    first = run_benchmark.run_benchmark(options, service=FakeService())

    assert first.stage_metrics is not None
    assert {
        "stage_metrics.json",
        "error_analysis.json",
        "gold_diagnostics.jsonl",
    }.issubset({path.name for path in first.run_dir.iterdir()})
    assert len(_read_jsonl(first.run_dir / "gold_diagnostics.jsonl")) == 2

    resumed = run_benchmark.run_benchmark(
        options.model_copy(update={"resume": True}),
        service=FakeService(),
    )
    assert len(_read_jsonl(resumed.run_dir / "gold_diagnostics.jsonl")) == 2
    assert "阶段诊断" in (resumed.run_dir / "summary.md").read_text(encoding="utf-8")


def test_diagnostics_do_not_store_abstract_prompt_or_api_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "diagnostic-secret-value"
    monkeypatch.setenv("SCHOLAR_AGENT_LLM_API_KEY", secret)
    dataset = _dataset(tmp_path, count=1)
    result = run_benchmark.run_benchmark(
        _options(tmp_path, dataset, run_id="safe-diagnostics").model_copy(
            update={"diagnostics": True}
        ),
        service=FakeService(),
    )

    row = _read_jsonl(result.run_dir / "results.jsonl")[0]
    diagnostic_text = json.dumps(row["stage_diagnostics"], ensure_ascii=False)
    assert secret not in diagnostic_text
    assert "offline benchmark fixture" not in diagnostic_text
    assert "abstract" not in diagnostic_text.casefold()
    assert "prompt" not in diagnostic_text.casefold()


def test_benchmark_gold_does_not_appear_in_production_search_strategy() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "scholar_agent"
    strategy_text = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in ("agents", "services", "connectors", "retrieval")
        for path in (root / directory).rglob("*.py")
    )

    assert "Gold Fixture Paper 0" not in strategy_text
    assert "2401.00000" not in strategy_text
    assert "AutoScholarQuery" not in strategy_text


class FakeService:
    def __init__(
        self,
        *,
        category: str = "highly_relevant",
        fail_queries: set[str] | None = None,
        error_message: str | None = None,
    ) -> None:
        self.category = category
        self.fail_queries = fail_queries or set()
        self.error_message = error_message
        self.calls: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def run_search(self, query: str, **kwargs: Any) -> SearchServiceOutput:
        self.calls.append(query)
        self.kwargs.append(kwargs)
        if query in self.fail_queries or self.error_message:
            raise RuntimeError(self.error_message or "offline fixture failure")
        index = int(query.rsplit("-", 1)[-1])
        return _output(query, f"2401.{index:05d}", category=self.category)


def _output(
    query: str,
    arxiv_id: str,
    *,
    category: str,
) -> SearchServiceOutput:
    paper = Paper(
        title=f"Gold Fixture Paper {int(arxiv_id[-1])}",
        authors=["Fixture"],
        year=2024,
        abstract="offline benchmark fixture",
        identifiers=PaperIdentifiers(arxiv_id=arxiv_id),
        sources=["openalex"],
    )
    analysis = QueryAnalysis(
        original_query=query,
        language="en",
        intent="survey",
        domain="machine_learning",
        constraints=QueryConstraint(),
    )
    plan = SearchPlan(
        query_analysis=analysis,
        subqueries=[
            SearchSubquery(
                query=query,
                source_hints=["openalex"],
                purpose="original_query",
            )
        ],
        selected_sources=["openalex"],
        top_k=20,
    )
    evidence = [EvidenceItem(source="title", text=paper.title, confidence=1.0)]
    judgement = JudgementResult(
        paper=paper,
        score=0.9,
        category=category,
        reasoning="offline fixture",
        evidence=evidence,
    )
    ranked = RankedPaper(
        rank=1,
        paper=paper,
        final_score=0.9,
        category=category,
        score_breakdown=RerankScoreBreakdown(
            relevance_score=0.9,
            authority_score=0.5,
            timeliness_score=0.5,
            metadata_score=1.0,
            final_score=0.9,
            relevance_weight=0.65,
            authority_weight=0.1,
            timeliness_weight=0.15,
            metadata_weight=0.1,
        ),
        ranking_reason="offline fixture",
        evidence=evidence,
    )
    stats = SourceStats(
        source="openalex",
        returned_count=1,
        diagnostics=ConnectorDiagnostics(request_count=1),
    )
    retrieval = RetrievalOutput(
        query=query,
        requested_sources=["openalex"],
        raw_count=1,
        deduplicated_count=1,
        papers=[paper],
        source_stats=[stats],
    )
    return SearchServiceOutput(
        search_plan=plan,
        retrieval_outputs=[retrieval],
        raw_count=1,
        deduplicated_count=1,
        judgements=[judgement],
        ranked_papers=[ranked],
        all_ranked_papers=[ranked],
        source_stats=[stats],
        search_diagnostics=stats.diagnostics,
        stage_snapshots=_stage_snapshots(ranked, judgement),
    )


def _stage_snapshots(
    ranked: RankedPaper,
    judgement: JudgementResult,
) -> list[StageCandidateSnapshot]:
    provenance = [
        CandidateProvenance(
            origin_kind="initial_query",
            origin_stage="initial_retrieval",
            origin_subquery="fixture query",
            source="openalex",
        )
    ]
    base = DiagnosticCandidate(
        identifiers=ranked.paper.identifiers,
        title=ranked.paper.title,
        year=ranked.paper.year,
        sources=["openalex"],
        provenance=provenance,
    )
    judged = base.model_copy(
        update={
            "judgement_score": judgement.score,
            "category": judgement.category,
        }
    )
    ranked_candidate = judged.model_copy(
        update={
            "rank": ranked.rank,
            "final_score": ranked.final_score,
        }
    )
    snapshots = [
        StageCandidateSnapshot(stage="initial_retrieval", candidates=[base]),
        StageCandidateSnapshot(stage="initial_deduplicated", candidates=[base]),
        StageCandidateSnapshot(stage="initial_judged", candidates=[judged]),
        StageCandidateSnapshot(stage="initial_reranked", candidates=[ranked_candidate]),
    ]
    snapshots.extend(
        StageCandidateSnapshot(stage=stage, status="skipped", skipped_reason="disabled")
        for stage in (
            "query_evolution_retrieval",
            "post_evolution_deduplicated",
            "post_evolution_judged",
            "post_evolution_reranked",
            "refchain_retrieval",
            "post_refchain_deduplicated",
            "post_refchain_judged",
            "post_refchain_reranked",
        )
    )
    snapshots.append(
        StageCandidateSnapshot(stage="final_ranked", candidates=[ranked_candidate])
    )
    return snapshots


def _dataset(tmp_path: Path, *, count: int) -> Path:
    rows = [
        {
            "qid": f"case-{index}",
            "question": f"query-{index}",
            "answer": [f"Gold Fixture Paper {index}"],
            "answer_arxiv_id": [f"2401.{index:05d}"],
            "source_meta": {"published_time": "20240101"},
        }
        for index in range(count)
    ]
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _options(
    tmp_path: Path,
    dataset: Path,
    *,
    run_id: str,
    offset: int = 0,
    limit: int | None = None,
    result_policy: str = "highly_and_partial",
) -> run_benchmark.BenchmarkRunOptions:
    return run_benchmark.BenchmarkRunOptions(
        dataset="auto_scholar_query",
        dataset_path=dataset,
        offset=offset,
        limit=limit,
        output_root=tmp_path / "runs",
        run_id=run_id,
        sources=["openalex"],
        result_policy=result_policy,
        max_workers=1,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_cli_exposes_neural_reranker_configuration() -> None:
    args = run_benchmark._parser().parse_args(
        [
            "--dataset",
            "auto_scholar_query",
            "--run-id",
            "contest_qual200_reranker_v1",
            "--sources",
            "local_hybrid",
            "--local-hybrid-reranker-model",
            "models/qwen3",
            "--local-hybrid-reranker-candidate-limit",
            "120",
            "--local-hybrid-reranker-batch-size",
            "8",
            "--local-hybrid-reranker-device",
            "cuda:1",
        ]
    )

    assert args.local_hybrid_reranker_model == "models/qwen3"
    assert args.local_hybrid_reranker_candidate_limit == 120
    assert args.local_hybrid_reranker_batch_size == 8
    assert args.local_hybrid_reranker_device == "cuda:1"
