"""Load local offline evaluation fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scholar_agent.agents.retriever import RetrievalOutput, SourceStats
from scholar_agent.core.evaluation_schemas import EvalQuery
from scholar_agent.core.paper_schemas import Paper
from scholar_agent.evaluation.metrics import canonical_paper_id


class EvaluationFixtures(BaseModel):
    """Loaded local fixtures for an offline evaluation run."""

    eval_queries: list[EvalQuery] = Field(default_factory=list)
    retrieval_outputs: dict[str, RetrievalOutput] = Field(default_factory=dict)
    reference_outputs: dict[str, list[Paper]] = Field(default_factory=dict)


def load_eval_queries(path: str | Path) -> list[EvalQuery]:
    """Load EvalQuery records from JSON or JSONL."""

    fixture_path = Path(path)
    records = _read_json_or_jsonl(fixture_path)
    if isinstance(records, Mapping):
        raw_queries = records.get("queries", records.get("eval_queries", []))
    else:
        raw_queries = records
    return [EvalQuery.model_validate(item) for item in raw_queries]


def load_retrieval_outputs(path: str | Path) -> dict[str, RetrievalOutput]:
    """Load query-keyed RetrievalOutput fixtures from JSON or JSONL."""

    records = _read_json_or_jsonl(Path(path))
    outputs: dict[str, RetrievalOutput] = {}

    if isinstance(records, Mapping) and "outputs" not in records:
        iterable = []
        for query, output in records.items():
            if query in {"metadata", "version"}:
                continue
            raw_output = dict(output)
            raw_output.setdefault("query", query)
            iterable.append(raw_output)
    else:
        iterable = records.get("outputs", []) if isinstance(records, Mapping) else records

    for item in iterable:
        output = RetrievalOutput.model_validate(item)
        outputs[_query_key(output.query)] = output
    return outputs


def load_reference_outputs(path: str | Path) -> dict[str, list[Paper]]:
    """Load canonical-seed-ID keyed reference Paper fixtures."""

    records = _read_json_or_jsonl(Path(path))
    references: dict[str, list[Paper]] = {}

    if isinstance(records, Mapping) and "references" not in records:
        iterable = []
        for seed_id, papers in records.items():
            if seed_id in {"metadata", "version"}:
                continue
            iterable.append({"seed_id": seed_id, "papers": papers})
    else:
        iterable = (
            records.get("references", []) if isinstance(records, Mapping) else records
        )

    for item in iterable:
        seed_id = str(item["seed_id"]).strip().casefold()
        references[seed_id] = [Paper.model_validate(paper) for paper in item["papers"]]
    return references


def load_evaluation_fixtures(fixtures_dir: str | Path) -> EvaluationFixtures:
    """Load the conventional fixture directory layout."""

    root = Path(fixtures_dir)
    return EvaluationFixtures(
        eval_queries=load_eval_queries(root / "search_cases.jsonl"),
        retrieval_outputs=load_retrieval_outputs(root / "retrieval_outputs.json"),
        reference_outputs=load_reference_outputs(root / "reference_outputs.json"),
    )


def build_fixture_retriever(
    retrieval_outputs: Mapping[str, RetrievalOutput],
) -> Callable[[str, int, list[str] | None], RetrievalOutput]:
    """Build a fake retriever from local RetrievalOutput fixtures."""

    normalized_outputs = {
        _query_key(query): output for query, output in retrieval_outputs.items()
    }

    def fake_retriever(
        query: str,
        limit_per_source: int = 20,
        sources: list[str] | None = None,
    ) -> RetrievalOutput:
        del limit_per_source
        key = _query_key(query)
        output = normalized_outputs.get(key)
        if output is None:
            message = f"fixture_missing_retrieval:{query}"
            requested_sources = list(sources or [])
            return RetrievalOutput(
                query=query,
                requested_sources=requested_sources,
                raw_count=0,
                deduplicated_count=0,
                papers=[],
                source_stats=[
                    SourceStats(
                        source="fixture",
                        returned_count=0,
                        latency_seconds=0.0,
                        error_message=message,
                    )
                ],
                warnings=[message],
                latency_seconds=0.0,
            )
        copied = output.model_copy(deep=True)
        copied.query = query
        if sources is not None:
            copied.requested_sources = list(sources)
        return copied

    return fake_retriever


def build_fixture_reference_fetcher(
    reference_outputs: Mapping[str, list[Paper]],
) -> Callable[[Paper, int], list[Paper]]:
    """Build a fake reference fetcher from canonical seed IDs."""

    normalized_outputs = {
        str(seed_id).strip().casefold(): list(papers)
        for seed_id, papers in reference_outputs.items()
    }

    def fake_reference_fetcher(paper: Paper, limit: int = 20) -> list[Paper]:
        seed_id = canonical_paper_id(paper)
        if seed_id is None:
            return []
        papers = normalized_outputs.get(seed_id.casefold(), [])
        return [item.model_copy(deep=True) for item in papers[: max(0, limit)]]

    return fake_reference_fetcher


def _read_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [
                json.loads(line)
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _query_key(query: str) -> str:
    return " ".join(query.casefold().split())
