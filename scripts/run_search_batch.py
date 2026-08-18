#!/usr/bin/env python3
"""Run SearchService for a batch of JSONL queries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scholar_agent.services.api_mapper import (  # noqa: E402
    map_search_service_output_to_api_result,
)
from scholar_agent.core.env_loader import load_project_env  # noqa: E402
from scholar_agent.services.search_service import SearchService  # noqa: E402
from scholar_agent.agents.judgement_config import load_judgement_config  # noqa: E402


SUPPORTED_SOURCES = {"openalex", "arxiv", "semantic_scholar", "pubmed"}
QUERY_EVOLUTION_POLICIES = {"off", "seed_expansion", "coverage_gap"}
QUERY_PLANNING_POLICIES = {
    "current_rules",
    "controlled_relaxation",
    "disjunctive_facets",
    "current_plus_disjunctive",
    "facet_union",
    "facet_balanced",
    "llm_semantic",
    "llm_constrained_rewrite",
}
JUDGEMENT_POLICIES = {"current_rules", "calibrated_rules_v1"}


def main(argv: list[str] | None = None) -> int:
    load_project_env(REPO_ROOT)

    parser = argparse.ArgumentParser(
        description="Run SearchService over a JSONL query file and write JSONL results."
    )
    parser.add_argument("--input", required=True, help="Input JSONL query file.")
    parser.add_argument("--output", required=True, help="Output JSONL result file.")
    parser.add_argument("--top-k", type=int, default=20, help="Default top_k.")
    parser.add_argument(
        "--run-profile",
        default="balanced",
        choices=["fast", "balanced", "high_recall", "evaluation"],
        help="Default run profile.",
    )
    parser.add_argument(
        "--current-year",
        type=int,
        default=None,
        help="Default current year for reproducible time parsing.",
    )
    parser.add_argument(
        "--enable-query-evolution",
        action="store_true",
        help="Enable Query Evolution by default for rows that do not override it.",
    )
    parser.add_argument(
        "--query-evolution-policy",
        choices=sorted(QUERY_EVOLUTION_POLICIES),
        default="coverage_gap",
        help="Query Evolution strategy used when the feature is enabled.",
    )
    parser.add_argument(
        "--query-planning-policy",
        choices=sorted(QUERY_PLANNING_POLICIES),
        default="current_rules",
        help="Initial subquery planning strategy.",
    )
    parser.add_argument(
        "--judgement-policy",
        choices=sorted(JUDGEMENT_POLICIES),
        default="current_rules",
        help="Deterministic relevance judgement strategy.",
    )
    parser.add_argument("--judgement-config", default=None)
    parser.add_argument(
        "--enable-refchain",
        action="store_true",
        help="Enable RefChain by default for rows that do not override it.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="SearchService(max_workers=...).",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help=(
            "Comma-separated default retrieval sources, for example "
            "arxiv,semantic_scholar. JSONL source_preferences overrides this."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop and return non-zero after the first per-row failure.",
    )
    parser.add_argument(
        "--sleep-between-cases-seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds between cases. Defaults to 0.",
    )
    parser.add_argument(
        "--dump-ranked-candidates",
        action="store_true",
        help=(
            "Write ranked_candidates.jsonl next to the output JSONL with top10 "
            "internal ranked-paper diagnostics. Defaults to disabled."
        ),
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"input path is not a file: {input_path}", file=sys.stderr)
        return 1

    try:
        cases = _load_cases(input_path)
        default_sources = _parse_sources(args.sources, field_name="--sources")
        sleep_between_cases_seconds = _parse_sleep_between_cases_seconds(
            args.sleep_between_cases_seconds
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    service_kwargs: dict[str, Any] = {"max_workers": args.max_workers}
    if args.judgement_policy != "current_rules" or args.judgement_config:
        service_kwargs.update(
            judgement_policy=args.judgement_policy,
            judgement_config=(
                load_judgement_config(args.judgement_config)
                if args.judgement_config
                else None
            ),
        )
    service = SearchService(**service_kwargs)
    had_failure = False

    ranked_candidates_path = output_path.parent / "ranked_candidates.jsonl"
    ranked_candidates_handle = (
        ranked_candidates_path.open("w", encoding="utf-8")
        if args.dump_ranked_candidates
        else None
    )
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for index, case in enumerate(cases):
                result = _run_case(
                    case,
                    service=service,
                    default_top_k=args.top_k,
                    default_run_profile=args.run_profile,
                    default_current_year=args.current_year,
                    default_enable_query_evolution=args.enable_query_evolution,
                    default_query_evolution_policy=args.query_evolution_policy,
                    default_query_planning_policy=args.query_planning_policy,
                    default_judgement_policy=args.judgement_policy,
                    default_enable_refchain=args.enable_refchain,
                    default_sources=default_sources,
                )
                if result["status"] == "failed":
                    had_failure = True
                debug_payload = result.pop("_ranked_candidates_debug", None)
                handle.write(json.dumps(result, ensure_ascii=False))
                handle.write("\n")
                handle.flush()
                if ranked_candidates_handle is not None:
                    ranked_candidates_handle.write(
                        json.dumps(
                            debug_payload
                            or _empty_ranked_candidates_debug(
                                result["case_id"],
                                result["query"],
                            ),
                            ensure_ascii=False,
                        )
                    )
                    ranked_candidates_handle.write("\n")
                    ranked_candidates_handle.flush()
                if had_failure and args.fail_fast:
                    return 1
                if sleep_between_cases_seconds > 0 and index < len(cases) - 1:
                    time.sleep(sleep_between_cases_seconds)
    finally:
        if ranked_candidates_handle is not None:
            ranked_candidates_handle.close()

    return 0


def _load_cases(input_path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
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
        payload = dict(payload)
        if not str(payload.get("case_id") or "").strip():
            payload["case_id"] = f"row_{len(cases) + 1}"
        cases.append(payload)
    return cases


def _run_case(
    case: dict[str, Any],
    *,
    service: SearchService,
    default_top_k: int,
    default_run_profile: str,
    default_current_year: int | None,
    default_enable_query_evolution: bool,
    default_query_evolution_policy: str,
    default_query_planning_policy: str,
    default_enable_refchain: bool,
    default_sources: list[str] | None,
    default_judgement_policy: str = "current_rules",
) -> dict[str, Any]:
    start = time.perf_counter()
    case_id = str(case["case_id"])
    query = str(case.get("query") or "")
    try:
        if not query.strip():
            raise ValueError("query must not be empty")
        top_k = int(case.get("top_k", default_top_k))
        run_profile = str(case.get("run_profile", default_run_profile))
        current_year = case.get("current_year", default_current_year)
        if current_year is not None:
            current_year = int(current_year)
        enable_query_evolution = bool(
            case.get("enable_query_evolution", default_enable_query_evolution)
        )
        query_evolution_policy = str(
            case.get(
                "query_evolution_policy",
                default_query_evolution_policy,
            )
        )
        if query_evolution_policy not in QUERY_EVOLUTION_POLICIES:
            raise ValueError(
                f"unsupported query_evolution_policy: {query_evolution_policy}"
            )
        query_planning_policy = str(
            case.get(
                "query_planning_policy",
                default_query_planning_policy,
            )
        )
        if query_planning_policy not in QUERY_PLANNING_POLICIES:
            raise ValueError(
                f"unsupported query_planning_policy: {query_planning_policy}"
            )
        judgement_policy = str(
            case.get("judgement_policy", default_judgement_policy)
        )
        if judgement_policy not in JUDGEMENT_POLICIES:
            raise ValueError(f"unsupported judgement_policy: {judgement_policy}")
        enable_refchain = bool(case.get("enable_refchain", default_enable_refchain))
        sources_override = _case_sources(case, default_sources)

        output = service.run_search(
            query,
            top_k=top_k,
            run_profile=run_profile,  # type: ignore[arg-type]
            enable_query_evolution=enable_query_evolution,
            query_evolution_policy=query_evolution_policy,  # type: ignore[arg-type]
            query_planning_policy=query_planning_policy,  # type: ignore[arg-type]
            enable_refchain=enable_refchain,
            enable_synthesis=True,
            current_year=current_year,
            sources_override=sources_override,
            judgement_policy=judgement_policy,  # type: ignore[arg-type]
        )
        api_result = map_search_service_output_to_api_result(
            run_id=f"batch_{case_id}",
            output=output,
            status="succeeded",
            partial=False,
        )
        return {
            "case_id": case_id,
            "query": query,
            "status": "succeeded",
            "result": api_result.model_dump(mode="json"),
            "error": None,
            "latency_seconds": time.perf_counter() - start,
            "_ranked_candidates_debug": _ranked_candidates_debug_payload(
                case_id,
                query,
                output,
            ),
        }
    except Exception as exc:  # noqa: BLE001 - isolate per-row batch failure
        return {
            "case_id": case_id,
            "query": query,
            "status": "failed",
            "result": None,
            "error": str(exc),
            "latency_seconds": time.perf_counter() - start,
        }


def _ranked_candidates_debug_payload(
    case_id: str,
    query: str,
    output: Any,
) -> dict[str, Any]:
    search_plan = getattr(output, "search_plan", None)
    expanded_queries = [
        getattr(subquery, "query", None)
        for subquery in getattr(search_plan, "subqueries", []) or []
    ]
    source_preferences = list(getattr(search_plan, "selected_sources", []) or [])
    return {
        "case_id": case_id,
        "query": query,
        "expanded_queries": expanded_queries,
        "source_preferences": source_preferences,
        "retrieval_queries": _retrieval_queries_by_source(output),
        "raw_count": getattr(output, "raw_count", None),
        "deduplicated_count": getattr(output, "deduplicated_count", None),
        "ranked_candidates": [
            _ranked_candidate_payload(candidate)
            for candidate in _ranked_candidates_for_debug(output)[:10]
        ],
    }


def _empty_ranked_candidates_debug(case_id: str, query: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "query": query,
        "expanded_queries": [],
        "source_preferences": [],
        "retrieval_queries": {},
        "raw_count": None,
        "deduplicated_count": None,
        "ranked_candidates": [],
    }


def _retrieval_queries_by_source(output: Any) -> dict[str, list[str]]:
    by_source: dict[str, list[str]] = {}
    for source_stat in getattr(output, "source_stats", []) or []:
        source = str(getattr(source_stat, "source", "") or "").strip()
        query = str(getattr(source_stat, "query", "") or "").strip()
        if not source or not query:
            continue
        queries = by_source.setdefault(source, [])
        if query not in queries:
            queries.append(query)
    return by_source


def _ranked_candidate_payload(candidate: Any) -> dict[str, Any]:
    paper = getattr(candidate, "paper", None)
    identifiers = getattr(paper, "identifiers", None)
    score_breakdown = getattr(candidate, "score_breakdown", None)
    return {
        "rank": getattr(candidate, "rank", None),
        "title": getattr(paper, "title", None),
        "source": ",".join(getattr(paper, "sources", []) or []) or None,
        "sources": list(getattr(paper, "sources", []) or []),
        "arxiv_id": getattr(identifiers, "arxiv_id", None),
        "semantic_scholar_id": getattr(identifiers, "semantic_scholar_id", None),
        "doi": getattr(identifiers, "doi", None),
        "year": getattr(paper, "year", None),
        "category": getattr(candidate, "category", None),
        "judgement_score": getattr(score_breakdown, "relevance_score", None),
        "final_score": getattr(candidate, "final_score", None),
        "ranking_reason": getattr(candidate, "ranking_reason", None) or "-",
        "score_breakdown": _score_breakdown_payload(score_breakdown),
    }


def _score_breakdown_payload(score_breakdown: Any) -> dict[str, Any]:
    return {
        "judgement": getattr(score_breakdown, "relevance_score", None),
        "authority": getattr(score_breakdown, "authority_score", None),
        "timeliness": getattr(score_breakdown, "timeliness_score", None),
        "metadata": getattr(score_breakdown, "metadata_score", None),
        "category_multiplier": getattr(score_breakdown, "category_multiplier", None),
        "final_score": getattr(score_breakdown, "final_score", None),
    }


def _ranked_candidates_for_debug(output: Any) -> list[Any]:
    all_ranked = list(getattr(output, "all_ranked_papers", []) or [])
    if all_ranked:
        return all_ranked
    return list(getattr(output, "ranked_papers", []) or [])


def _case_sources(
    case: dict[str, Any],
    default_sources: list[str] | None,
) -> list[str] | None:
    if "source_preferences" not in case:
        return default_sources
    return _parse_sources(case.get("source_preferences"), field_name="source_preferences")


def _parse_sources(value: Any, *, field_name: str) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        raw_sources = value.split(",")
    elif isinstance(value, list):
        raw_sources = value
    else:
        raise ValueError(f"{field_name} must be a comma-separated string or list")

    normalized: list[str] = []
    seen: set[str] = set()
    invalid: list[str] = []
    for item in raw_sources:
        source = str(item).strip().lower()
        if not source:
            continue
        if source not in SUPPORTED_SOURCES:
            invalid.append(source)
            continue
        if source not in seen:
            normalized.append(source)
            seen.add(source)

    if invalid:
        allowed = ", ".join(sorted(SUPPORTED_SOURCES))
        raise ValueError(
            f"{field_name} contains unsupported source(s): {', '.join(invalid)}; "
            f"allowed sources: {allowed}"
        )
    return normalized or None


def _parse_sleep_between_cases_seconds(value: float) -> float:
    try:
        sleep_seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("--sleep-between-cases-seconds must be a number") from exc
    if sleep_seconds < 0:
        raise ValueError("--sleep-between-cases-seconds must be >= 0")
    return sleep_seconds


if __name__ == "__main__":
    raise SystemExit(main())
