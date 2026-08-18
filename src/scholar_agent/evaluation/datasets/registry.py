"""Benchmark 数据集注册、加载与只读完整性检查。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery
from scholar_agent.evaluation.datasets.auto_scholar_query import (
    DATASET_NAME as AUTO_SCHOLAR_QUERY,
    load_auto_scholar_query,
    parse_auto_scholar_query_record,
    read_jsonl_records,
)
from scholar_agent.evaluation.datasets.beir_scifact import (
    DATASET_NAME as BEIR_SCI_FACT,
    load_beir_scifact_enriched,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


class BenchmarkDatasetReport(BaseModel):
    dataset: str
    source_path: str
    case_count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    gold_paper_count: int = Field(ge=0)
    cases_without_gold: int = Field(ge=0)
    gold_with_doi: int = Field(ge=0)
    gold_with_arxiv_id: int = Field(ge=0)
    gold_with_openalex_id: int = Field(ge=0)
    gold_with_semantic_scholar_id: int = Field(ge=0)
    gold_with_s2orc_corpus_id: int = Field(ge=0)
    gold_with_pubmed_id: int = Field(ge=0)
    gold_with_title_year_only: int = Field(ge=0)
    invalid_case_count: int = Field(ge=0)
    duplicate_case_id_count: int = Field(ge=0)


@dataclass(frozen=True)
class _DatasetDefinition:
    source_path: Path
    loader: Callable[[str | Path], list[EvalQuery]]


_DATASETS = {
    AUTO_SCHOLAR_QUERY: _DatasetDefinition(
        source_path=REPO_ROOT / "benchmark" / "AutoScholarQuery_test.jsonl",
        loader=load_auto_scholar_query,
    ),
    BEIR_SCI_FACT: _DatasetDefinition(
        source_path=(
            REPO_ROOT
            / "outputs"
            / "benchmark_inputs"
            / "beir_scifact"
            / "extracted"
        ),
        loader=load_beir_scifact_enriched,
    ),
}


def supported_datasets() -> tuple[str, ...]:
    return tuple(_DATASETS)


def dataset_source_path(name: str, path: str | Path | None = None) -> Path:
    definition = _dataset_definition(name)
    return Path(path).expanduser().resolve() if path is not None else definition.source_path


def load_dataset(
    name: str,
    *,
    path: str | Path | None = None,
) -> list[EvalQuery]:
    definition = _dataset_definition(name)
    return definition.loader(dataset_source_path(name, path))


def inspect_dataset(
    name: str,
    *,
    path: str | Path | None = None,
) -> BenchmarkDatasetReport:
    normalized_name = _normalize_name(name)
    source_path = dataset_source_path(normalized_name, path)
    if normalized_name == BEIR_SCI_FACT:
        queries = _dataset_definition(normalized_name).loader(source_path)
        gold = [paper for query in queries for paper in query.gold_papers]
        return BenchmarkDatasetReport(
            dataset=normalized_name,
            source_path=str(source_path),
            case_count=len(queries),
            query_count=len(queries),
            gold_paper_count=len(gold),
            cases_without_gold=sum(not q.gold_papers for q in queries),
            gold_with_doi=sum(bool(item.doi) for item in gold),
            gold_with_arxiv_id=sum(bool(item.arxiv_id) for item in gold),
            gold_with_openalex_id=sum(bool(item.openalex_id) for item in gold),
            gold_with_semantic_scholar_id=sum(
                bool(item.semantic_scholar_id) for item in gold
            ),
            gold_with_s2orc_corpus_id=sum(
                bool(item.s2orc_corpus_id) for item in gold
            ),
            gold_with_pubmed_id=sum(bool(item.pubmed_id) for item in gold),
            gold_with_title_year_only=sum(_is_title_year_only(item) for item in gold),
            invalid_case_count=0,
            duplicate_case_id_count=0,
        )
    if normalized_name != AUTO_SCHOLAR_QUERY:
        raise ValueError(f"inspection is not implemented for dataset: {normalized_name}")

    records = read_jsonl_records(source_path)
    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()
    invalid_count = 0
    duplicate_count = 0
    cases_without_gold = 0
    for line_number, record in records:
        raw_id = str(record.get("qid") or "").strip()
        if raw_id and raw_id in seen_ids:
            duplicate_count += 1
        elif raw_id:
            seen_ids.add(raw_id)
        if not record.get("answer") or not record.get("answer_arxiv_id"):
            cases_without_gold += 1
        try:
            queries.append(
                parse_auto_scholar_query_record(record, line_number=line_number)
            )
        except ValueError:
            invalid_count += 1

    gold = [paper for query in queries for paper in query.gold_papers]
    return BenchmarkDatasetReport(
        dataset=normalized_name,
        source_path=str(source_path),
        case_count=len(records),
        query_count=len(queries),
        gold_paper_count=len(gold),
        cases_without_gold=cases_without_gold,
        gold_with_doi=sum(bool(item.doi) for item in gold),
        gold_with_arxiv_id=sum(bool(item.arxiv_id) for item in gold),
        gold_with_openalex_id=sum(bool(item.openalex_id) for item in gold),
        gold_with_semantic_scholar_id=sum(
            bool(item.semantic_scholar_id) for item in gold
        ),
        gold_with_s2orc_corpus_id=sum(
            bool(item.s2orc_corpus_id) for item in gold
        ),
        gold_with_pubmed_id=sum(bool(item.pubmed_id) for item in gold),
        gold_with_title_year_only=sum(_is_title_year_only(item) for item in gold),
        invalid_case_count=invalid_count,
        duplicate_case_id_count=duplicate_count,
    )


def _dataset_definition(name: str) -> _DatasetDefinition:
    normalized = _normalize_name(name)
    try:
        return _DATASETS[normalized]
    except KeyError as exc:
        choices = ", ".join(supported_datasets())
        raise ValueError(f"unsupported dataset: {normalized}; choose from {choices}") from exc


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("dataset name must not be empty")
    return normalized


def _is_title_year_only(paper: EvalGoldPaper) -> bool:
    has_stable_id = any(
        (
            paper.doi,
            paper.arxiv_id,
            paper.openalex_id,
            paper.semantic_scholar_id,
            paper.s2orc_corpus_id,
            paper.pubmed_id,
        )
    )
    return bool(paper.title and paper.year and not has_stable_id)
