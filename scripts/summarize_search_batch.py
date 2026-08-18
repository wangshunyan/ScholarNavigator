#!/usr/bin/env python3
"""Summarize run_search_batch.py JSONL output as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize SearchService batch JSONL results as Markdown."
    )
    parser.add_argument("--input", required=True, help="Input batch result JSONL file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top queries, warnings, and papers to display.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"input path is not a file: {input_path}", file=sys.stderr)
        return 1

    try:
        rows = load_batch_rows(input_path)
        summary = summarize_rows(rows, top_n=args.top_n)
        markdown = render_markdown_summary(summary)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


def load_batch_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid JSONL at line {line_number}: expected object")
        rows.append(payload)
    return rows


def summarize_rows(rows: list[dict[str, Any]], top_n: int = 10) -> dict[str, Any]:
    top_n = max(1, int(top_n))
    total_cases = len(rows)
    succeeded_count = sum(1 for row in rows if row.get("status") == "succeeded")
    failed_count = sum(1 for row in rows if row.get("status") == "failed")
    latencies = [_as_float(row.get("latency_seconds")) for row in rows]

    paper_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    source_error_counts: Counter[str] = Counter()
    source_reliability: dict[str, dict[str, Any]] = {}
    case_summaries: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []

    cost_totals = {
        "api_call_count": 0,
        "logical_search_call_count": 0,
        "search_api_call_count": 0,
        "reference_api_call_count": 0,
        "retry_count": 0,
        "error_count": 0,
        "cache_hit_count": 0,
        "rate_limit_wait_seconds": 0.0,
        "llm_call_count": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "estimated_total_tokens": 0,
        "raw_candidate_count": 0,
        "deduplicated_candidate_count": 0,
    }

    for row in rows:
        result = row.get("result")
        result_dict = result if isinstance(result, dict) else None
        if row.get("status") == "succeeded" and result_dict is None:
            warning_counts["succeeded_result_missing"] += 1

        high_count = _paper_list_count(result_dict, "highly_relevant_papers")
        partial_count = _paper_list_count(result_dict, "partially_relevant_papers")
        synthesis_status = _synthesis_status(result_dict)
        expanded_queries = _expanded_queries(result_dict)
        source_preferences = _source_preferences(result_dict)
        raw_count, deduplicated_count = _retrieval_counts(result_dict)
        error = str(row.get("error") or "")

        case_summaries.append(
            {
                "case_id": str(row.get("case_id") or ""),
                "query": str(row.get("query") or ""),
                "status": str(row.get("status") or ""),
                "latency_seconds": _as_float(row.get("latency_seconds")),
                "highly_relevant_count": high_count,
                "partially_relevant_count": partial_count,
                "synthesis_status": synthesis_status,
                "expanded_queries": expanded_queries,
                "source_preferences": source_preferences,
                "raw_count": raw_count,
                "deduplicated_count": deduplicated_count,
                "error": error,
            }
        )

        if row.get("status") == "failed":
            failed_cases.append(
                {
                    "case_id": str(row.get("case_id") or ""),
                    "query": str(row.get("query") or ""),
                    "error": error,
                }
            )

        if result_dict is None:
            continue

        _update_source_reliability(source_reliability, result_dict)

        for key in cost_totals:
            value = _as_float(_cost_report(result_dict).get(key))
            cost_totals[key] += value if key == "rate_limit_wait_seconds" else int(value)

        for paper in _iter_result_papers(result_dict):
            title = _paper_title(paper)
            if title:
                paper_counts[title] += 1

        missing_evidence = result_dict.get("missing_evidence")
        if isinstance(missing_evidence, list):
            for item in missing_evidence:
                message = str(item)
                warning_counts[message] += 1
                if message.startswith("source_error"):
                    source_error_counts[message] += 1

    top_queries = sorted(
        case_summaries,
        key=lambda item: (-item["latency_seconds"], item["case_id"]),
    )[:top_n]

    return {
        "total_cases": total_cases,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "success_rate": succeeded_count / total_cases if total_cases else 0.0,
        "latency": {
            "average": sum(latencies) / len(latencies) if latencies else 0.0,
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "cost_totals": cost_totals,
        "case_summaries": case_summaries,
        "top_queries": top_queries,
        "top_papers": paper_counts.most_common(top_n),
        "warning_counts": warning_counts.most_common(top_n),
        "source_error_counts": source_error_counts.most_common(top_n),
        "source_reliability": _finalize_source_reliability(
            source_reliability,
            top_n=top_n,
        ),
        "failed_cases": failed_cases,
    }


def render_markdown_summary(summary: dict[str, Any]) -> str:
    latency = summary["latency"]
    cost = summary["cost_totals"]
    lines = [
        "# ScholarNavigator Batch Search Summary",
        "",
        "## Overview",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- Succeeded: {summary['succeeded_count']}",
        f"- Failed: {summary['failed_count']}",
        f"- Success rate: {_format_percent(summary['success_rate'])}",
        "- Latency seconds: avg {avg} / min {min_value} / max {max_value}".format(
            avg=_format_float(latency["average"]),
            min_value=_format_float(latency["min"]),
            max_value=_format_float(latency["max"]),
        ),
        "",
        "## Cost / Efficiency",
        "",
        f"- Succeeded cases: {summary['succeeded_count']}",
        f"- Failed cases: {summary['failed_count']}",
        f"- Average latency seconds: {_format_float(latency['average'])}",
        f"- Total API calls: {cost['api_call_count']}",
        f"- Logical search calls: {cost['logical_search_call_count']}",
        f"- Search API calls: {cost['search_api_call_count']}",
        f"- Reference API calls: {cost['reference_api_call_count']}",
        f"- Retries: {cost['retry_count']}",
        f"- Connector errors: {cost['error_count']}",
        f"- Cache hits: {cost['cache_hit_count']}",
        f"- Rate-limit wait seconds: {_format_float(cost['rate_limit_wait_seconds'])}",
        f"- Total LLM calls: {cost['llm_call_count']}",
        f"- Total LLM prompt tokens: {cost['llm_prompt_tokens']}",
        f"- Total LLM completion tokens: {cost['llm_completion_tokens']}",
        f"- Total LLM tokens: {cost['llm_total_tokens']}",
        f"- Estimated input tokens: {cost['estimated_input_tokens']}",
        f"- Estimated output tokens: {cost['estimated_output_tokens']}",
        f"- Estimated total tokens: {cost['estimated_total_tokens']}",
        f"- Raw candidates: {cost['raw_candidate_count']}",
        f"- Deduplicated candidates: {cost['deduplicated_candidate_count']}",
        "",
        "## Case Summary",
        "",
        "| case_id | status | latency_seconds | highly relevant | partially relevant | synthesis status | expanded_queries | source_preferences | raw_count | deduplicated_count | error |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in summary["case_summaries"]:
        lines.append(
            "| {case_id} | {status} | {latency} | {high} | {partial} | {synthesis} | {expanded_queries} | {source_preferences} | {raw_count} | {deduplicated_count} | {error} |".format(
                case_id=_escape_md(item["case_id"]),
                status=_escape_md(item["status"]),
                latency=_format_float(item["latency_seconds"]),
                high=item["highly_relevant_count"],
                partial=item["partially_relevant_count"],
                synthesis=_escape_md(item["synthesis_status"]),
                expanded_queries=_escape_md(item["expanded_queries"]),
                source_preferences=_escape_md(item["source_preferences"]),
                raw_count=_escape_md(item["raw_count"]),
                deduplicated_count=_escape_md(item["deduplicated_count"]),
                error=_escape_md(item["error"] or "-"),
            )
        )

    lines.extend(["", "## Top Queries By Latency", ""])
    lines.extend(_counter_table(["case_id", "status", "latency_seconds", "query"], _top_query_rows(summary)))

    lines.extend(["", "## Top Papers", ""])
    lines.extend(_count_table("title", summary["top_papers"]))

    lines.extend(["", "## Missing Evidence / Warning Counts", ""])
    lines.extend(_count_table("message", summary["warning_counts"]))

    lines.extend(["", "## Source Error Counts", ""])
    lines.extend(_count_table("source_error", summary["source_error_counts"]))

    lines.extend(["", "## Source Reliability", ""])
    lines.extend(_source_reliability_table(summary["source_reliability"]))

    lines.extend(
        [
            "",
            "## Failed Cases",
            "",
            "| case_id | query | error |",
            "| --- | --- | --- |",
        ]
    )
    failed_cases = summary["failed_cases"]
    if failed_cases:
        for item in failed_cases:
            lines.append(
                "| {case_id} | {query} | {error} |".format(
                    case_id=_escape_md(item["case_id"]),
                    query=_escape_md(item["query"]),
                    error=_escape_md(item["error"] or "-"),
                )
            )
    else:
        lines.append("| - | - | - |")

    lines.append("")
    return "\n".join(lines)


def _top_query_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in summary["top_queries"]:
        rows.append(
            [
                str(item["case_id"]),
                str(item["status"]),
                _format_float(item["latency_seconds"]),
                str(item["query"]),
            ]
        )
    return rows


def _counter_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if not rows:
        lines.append("| " + " | ".join("-" for _ in headers) + " |")
        return lines
    for row in rows:
        lines.append("| " + " | ".join(_escape_md(value) for value in row) + " |")
    return lines


def _count_table(label: str, values: list[tuple[str, int]]) -> list[str]:
    lines = [
        f"| {label} | count |",
        "| --- | ---: |",
    ]
    if not values:
        lines.append("| - | 0 |")
        return lines
    for value, count in values:
        lines.append(f"| {_escape_md(value)} | {count} |")
    return lines


def _source_reliability_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| source | call_count | success_count | error_count | cooldown_skip_count | total_returned_count | avg_latency_seconds | top error messages |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not items:
        lines.append("| - | 0 | 0 | 0 | 0 | 0 | 0.000 | - |")
        return lines
    for item in items:
        top_errors = "; ".join(
            f"{message} ({count})" for message, count in item["top_error_messages"]
        )
        lines.append(
            "| {source} | {call_count} | {success_count} | {error_count} | {cooldown_skip_count} | {total_returned_count} | {avg_latency_seconds} | {top_errors} |".format(
                source=_escape_md(item["source"]),
                call_count=item["call_count"],
                success_count=item["success_count"],
                error_count=item["error_count"],
                cooldown_skip_count=item["cooldown_skip_count"],
                total_returned_count=item["total_returned_count"],
                avg_latency_seconds=_format_float(item["avg_latency_seconds"]),
                top_errors=_escape_md(top_errors or "-"),
            )
        )
    return lines


def _update_source_reliability(
    source_reliability: dict[str, dict[str, Any]],
    result: dict[str, Any],
) -> None:
    diagnostics = result.get("retrieval_diagnostics")
    if not isinstance(diagnostics, dict):
        return
    stats_list = diagnostics.get("source_stats")
    if not isinstance(stats_list, list):
        return

    for raw_stats in stats_list:
        if not isinstance(raw_stats, dict):
            continue
        source = str(raw_stats.get("source") or "unknown").strip() or "unknown"
        item = source_reliability.setdefault(
            source,
            {
                "source": source,
                "call_count": 0,
                "success_count": 0,
                "error_count": 0,
                "cooldown_skip_count": 0,
                "total_returned_count": 0,
                "total_latency_seconds": 0.0,
                "error_messages": Counter(),
            },
        )
        returned_count = int(_as_float(raw_stats.get("returned_count")))
        latency_seconds = _as_float(raw_stats.get("latency_seconds"))
        error_message = str(raw_stats.get("error_message") or "").strip()

        item["call_count"] += 1
        item["total_returned_count"] += max(0, returned_count)
        item["total_latency_seconds"] += latency_seconds
        if error_message:
            item["error_count"] += 1
            item["error_messages"][error_message] += 1
            if "source_cooldown_skip" in error_message:
                item["cooldown_skip_count"] += 1
        else:
            item["success_count"] += 1


def _finalize_source_reliability(
    source_reliability: dict[str, dict[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for item in source_reliability.values():
        call_count = item["call_count"]
        finalized.append(
            {
                "source": item["source"],
                "call_count": call_count,
                "success_count": item["success_count"],
                "error_count": item["error_count"],
                "cooldown_skip_count": item["cooldown_skip_count"],
                "total_returned_count": item["total_returned_count"],
                "avg_latency_seconds": (
                    item["total_latency_seconds"] / call_count if call_count else 0.0
                ),
                "top_error_messages": item["error_messages"].most_common(top_n),
            }
        )
    return sorted(finalized, key=lambda value: value["source"])


def _iter_result_papers(result: dict[str, Any]) -> list[Any]:
    papers: list[Any] = []
    for key in ("highly_relevant_papers", "partially_relevant_papers"):
        value = result.get(key)
        if isinstance(value, list):
            papers.extend(value)
    return papers


def _paper_list_count(result: dict[str, Any] | None, key: str) -> int:
    if result is None:
        return 0
    value = result.get(key)
    return len(value) if isinstance(value, list) else 0


def _paper_title(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    paper = item.get("paper")
    if isinstance(paper, dict):
        return str(paper.get("title") or "").strip()
    return str(item.get("title") or "").strip()


def _synthesis_status(result: dict[str, Any] | None) -> str:
    if result is None:
        return "-"
    synthesis = result.get("synthesis")
    if isinstance(synthesis, dict):
        return str(synthesis.get("status") or "-")
    return "-"


def _expanded_queries(result: dict[str, Any] | None) -> str:
    if result is None:
        return "-"
    search_plan = result.get("search_plan")
    if not isinstance(search_plan, dict):
        return "-"
    queries = search_plan.get("expanded_queries")
    if not isinstance(queries, list):
        return "-"
    values = [str(query).strip() for query in queries if str(query).strip()]
    return "; ".join(values) if values else "-"


def _source_preferences(result: dict[str, Any] | None) -> str:
    if result is None:
        return "-"
    search_plan = result.get("search_plan")
    if not isinstance(search_plan, dict):
        return "-"
    sources = search_plan.get("source_preferences")
    if not isinstance(sources, list):
        return "-"
    values = [str(source).strip() for source in sources if str(source).strip()]
    return ",".join(values) if values else "-"


def _retrieval_counts(result: dict[str, Any] | None) -> tuple[str, str]:
    if result is None:
        return "-", "-"
    diagnostics = result.get("retrieval_diagnostics")
    if not isinstance(diagnostics, dict):
        return "-", "-"
    raw_count = diagnostics.get("raw_count")
    deduplicated_count = diagnostics.get("deduplicated_count")
    return _display_count(raw_count), _display_count(deduplicated_count)


def _display_count(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _cost_report(result: dict[str, Any]) -> dict[str, Any]:
    cost = result.get("cost_report")
    return cost if isinstance(cost, dict) else {}


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _format_float(value: Any) -> str:
    return f"{_as_float(value):.3f}"


def _format_percent(value: Any) -> str:
    return f"{_as_float(value) * 100:.1f}%"


def _escape_md(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
