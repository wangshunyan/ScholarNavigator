#!/usr/bin/env python3
"""Evaluate batch JSONL using the shared scholar_agent evaluation policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.core.evaluation_schemas import (  # noqa: E402
    EvalAggregateReport,
    EvalCaseEfficiency,
    EvalCaseStatistics,
    EvalGoldPaper,
    EvalMetricSet,
)
from scholar_agent.evaluation.metrics import (  # noqa: E402
    aggregate_efficiency,
    average_metric_sets,
    evaluable_gold_count,
    evaluate_ranking,
    matched_paper_ids,
    zero_metric_set,
)
from scholar_agent.evaluation.selection import (  # noqa: E402
    DEFAULT_RESULT_POLICY,
    ResultPolicy,
    select_ranked_results,
)


DEFAULT_K_VALUES = [5, 10, 20]
_FAILED_STATUSES = {"failed", "timeout", "cancelled"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate SearchService batch JSONL results against gold qrels."
    )
    parser.add_argument("--batch-results", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--k", action="append", type=int, default=None)
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Legacy alias for --result-policy highly_and_partial.",
    )
    parser.add_argument(
        "--result-policy",
        choices=["highly_only", "highly_and_partial"],
        default=None,
    )
    args = parser.parse_args(argv)

    if args.include_partial and args.result_policy == "highly_only":
        print(
            "--include-partial conflicts with --result-policy highly_only",
            file=sys.stderr,
        )
        return 1
    result_policy: ResultPolicy = (
        "highly_and_partial"
        if args.include_partial
        else args.result_policy or DEFAULT_RESULT_POLICY
    )

    batch_path = Path(args.batch_results)
    gold_path = Path(args.gold)
    for label, path in (("batch results", batch_path), ("gold", gold_path)):
        if not path.exists():
            print(f"{label} file not found: {path}", file=sys.stderr)
            return 1
        if not path.is_file():
            print(f"{label} path is not a file: {path}", file=sys.stderr)
            return 1

    try:
        k_values = _normalize_k_values(args.k or DEFAULT_K_VALUES)
        result = evaluate_batch_results(
            load_jsonl_objects(batch_path, label="batch results"),
            load_gold_rows(gold_path),
            k_values=k_values,
            result_policy=result_policy,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def load_jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid {label} JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"invalid {label} JSONL at line {line_number}: expected object"
            )
        rows.append(payload)
    return rows


def load_gold_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl_objects(path, label="gold")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"invalid gold JSONL at line {index}: missing case_id")
        if case_id in seen:
            raise ValueError(f"invalid gold JSONL at line {index}: duplicate case_id")
        seen.add(case_id)
        raw_papers = row.get("relevant_papers", [])
        if raw_papers is None:
            raw_papers = []
        if not isinstance(raw_papers, list):
            raise ValueError(
                f"invalid gold JSONL at line {index}: relevant_papers must be a list"
            )
        try:
            gold_papers = [
                EvalGoldPaper.model_validate(paper).model_dump(mode="json")
                for paper in raw_papers
            ]
        except Exception as exc:  # noqa: BLE001 - report malformed gold
            raise ValueError(f"invalid gold JSONL at line {index}: {exc}") from exc
        normalized.append({"case_id": case_id, "relevant_papers": gold_papers})
    return normalized


def evaluate_batch_results(
    batch_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    k_values: list[int] | None = None,
    result_policy: ResultPolicy = DEFAULT_RESULT_POLICY,
    include_partial: bool | None = None,
) -> dict[str, Any]:
    if include_partial is not None:
        legacy_policy: ResultPolicy = (
            "highly_and_partial" if include_partial else "highly_only"
        )
        if include_partial and result_policy == "highly_only":
            raise ValueError("include_partial conflicts with highly_only")
        result_policy = legacy_policy
    values = _normalize_k_values(k_values or DEFAULT_K_VALUES)
    batch_by_case = _index_batch_rows(batch_rows)
    all_gold_by_case = {
        str(row["case_id"]): list(row.get("relevant_papers") or [])
        for row in gold_rows
    }
    gold_by_case = {
        case_id: papers
        for case_id, papers in all_gold_by_case.items()
        if evaluable_gold_count(papers) > 0
    }

    missing_gold_cases = sorted(set(batch_by_case) - set(gold_by_case))
    missing_result_cases: list[str] = []
    failed_cases = [
        {
            "case_id": case_id,
            "query": str(row.get("query") or ""),
            "status": str(row.get("status") or "").casefold(),
            "error": str(row.get("error") or ""),
        }
        for case_id, row in batch_by_case.items()
        if str(row.get("status") or "").casefold() in _FAILED_STATUSES
    ]
    per_case: list[dict[str, Any]] = []
    success_metrics: list[EvalMetricSet] = []
    end_to_end_metrics: list[EvalMetricSet] = []
    efficiencies: list[EvalCaseEfficiency] = []

    for case_id, gold in gold_by_case.items():
        row = batch_by_case.get(case_id)
        status = str((row or {}).get("status") or "missing").casefold()
        result = (row or {}).get("result")
        valid_result = status == "succeeded" and _is_valid_result(result)
        if status not in _FAILED_STATUSES and not valid_result:
            missing_result_cases.append(case_id)

        if valid_result:
            ranked = select_ranked_results(result, policy=result_policy)
            metrics = evaluate_ranking(ranked, gold, values)
            success_metrics.append(metrics)
            end_to_end_metrics.append(metrics)
            matched_ids = matched_paper_ids(ranked, gold)
        else:
            ranked = []
            metrics = zero_metric_set(values)
            end_to_end_metrics.append(metrics)
            matched_ids = []

        efficiency = _case_efficiency(row, result if valid_result else None, len(ranked))
        efficiencies.append(efficiency)
        per_case.append(
            {
                "case_id": case_id,
                "query": str((row or {}).get("query") or ""),
                "status": status,
                "ranked_count": len(ranked),
                "gold_count": evaluable_gold_count(gold),
                "matched_ids": matched_ids,
                "metrics": _metric_payload(metrics),
                "efficiency": efficiency.model_dump(mode="json"),
            }
        )

    total_case_count = len(set(batch_by_case).union(gold_by_case))
    failed_count = len(failed_cases)
    success_count = len(success_metrics)
    statistics = EvalCaseStatistics(
        total_case_count=total_case_count,
        gold_case_count=len(gold_by_case),
        evaluated_success_count=success_count,
        failed_case_count=failed_count,
        missing_result_count=len(missing_result_cases),
        missing_gold_count=len(missing_gold_cases),
        success_rate=success_count / total_case_count if total_case_count else 0.0,
        failed_case_rate=failed_count / total_case_count if total_case_count else 0.0,
        missing_result_rate=(
            len(missing_result_cases) / total_case_count if total_case_count else 0.0
        ),
    )
    success_aggregate = (
        average_metric_sets(success_metrics)
        if success_metrics
        else zero_metric_set(values)
    )
    end_to_end_aggregate = (
        average_metric_sets(end_to_end_metrics)
        if end_to_end_metrics
        else zero_metric_set(values)
    )
    efficiency = aggregate_efficiency(efficiencies)
    aggregate_report = EvalAggregateReport(
        success_only_metrics=success_aggregate,
        end_to_end_metrics=end_to_end_aggregate,
        case_statistics=statistics,
        efficiency=efficiency,
    )
    return {
        "config": {
            "k_values": values,
            "result_policy": result_policy,
            "include_partial": result_policy == "highly_and_partial",
            "failed_cases_policy": "zero_in_end_to_end",
        },
        "aggregate": _metric_payload(success_aggregate),
        "success_only_metrics": _metric_payload(success_aggregate),
        "end_to_end_metrics": _metric_payload(end_to_end_aggregate),
        "case_statistics": statistics.model_dump(mode="json"),
        "efficiency": efficiency.model_dump(mode="json"),
        "aggregate_report": aggregate_report.model_dump(mode="json"),
        "per_case": per_case,
        "failed_cases": failed_cases,
        "missing_gold_cases": missing_gold_cases,
        "missing_result_cases": missing_result_cases,
        "case_count": len(batch_rows),
        "evaluated_case_count": success_count,
    }


def extract_ranked_papers(
    result: dict[str, Any],
    *,
    result_policy: ResultPolicy = DEFAULT_RESULT_POLICY,
    include_partial: bool | None = None,
) -> list[dict[str, Any]]:
    if include_partial is not None:
        result_policy = "highly_and_partial" if include_partial else "highly_only"
    return list(select_ranked_results(result, policy=result_policy))


def _index_batch_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"invalid batch results at row {index}: missing case_id")
        if case_id in indexed:
            raise ValueError(f"invalid batch results at row {index}: duplicate case_id")
        indexed[case_id] = row
    return indexed


def _is_valid_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    for key in ("highly_relevant_papers", "partially_relevant_papers"):
        candidates = result.get(key)
        if not isinstance(candidates, list):
            return False
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("paper"),
                dict,
            ):
                return False
    return True


def _case_efficiency(
    row: dict[str, Any] | None,
    result: dict[str, Any] | None,
    returned_result_count: int,
) -> EvalCaseEfficiency:
    if result is None:
        return EvalCaseEfficiency(
            latency_seconds=_as_float((row or {}).get("latency_seconds")),
            warnings=["efficiency_unavailable:result_missing"],
        )
    cost = result.get("cost_report")
    cost = cost if isinstance(cost, dict) else {}
    diagnostics = result.get("retrieval_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    return EvalCaseEfficiency(
        latency_seconds=_as_float(
            (row or {}).get("latency_seconds", cost.get("latency_seconds"))
        ),
        api_call_count=_as_int(cost.get("api_call_count")),
        search_api_call_count=_as_int(cost.get("search_api_call_count")),
        reference_api_call_count=_as_int(cost.get("reference_api_call_count")),
        retry_count=_as_int(cost.get("retry_count")),
        error_count=_as_int(cost.get("error_count")),
        llm_call_count=_as_int(cost.get("llm_call_count")),
        llm_total_tokens=_as_int(cost.get("llm_total_tokens")),
        search_rounds=_as_int(cost.get("search_rounds")),
        raw_count=_as_int(diagnostics.get("raw_count")),
        deduplicated_count=_as_int(diagnostics.get("deduplicated_count")),
        returned_result_count=returned_result_count,
        cache_hit_count=_as_int(cost.get("cache_hit_count")),
        rate_limit_wait_seconds=_as_float(
            cost.get("rate_limit_wait_seconds")
        ),
        source_call_count=(
            _as_int(cost.get("search_api_call_count"))
            + _as_int(cost.get("reference_api_call_count"))
        ),
        source_error_count=_as_int(cost.get("error_count")),
    )


def _metric_payload(metrics: EvalMetricSet) -> dict[str, Any]:
    return metrics.model_dump(mode="json", include={
        "recall_at_k",
        "precision_at_k",
        "f1_at_k",
        "ndcg_at_k",
        "mrr",
    })


def _normalize_k_values(values: list[int]) -> list[int]:
    normalized = sorted({int(value) for value in values if int(value) > 0})
    if not normalized:
        raise ValueError("at least one positive --k value is required")
    return normalized


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
