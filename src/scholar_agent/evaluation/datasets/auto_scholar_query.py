"""AutoScholarQuery test 集适配器。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scholar_agent.core.evaluation_schemas import EvalGoldPaper, EvalQuery


DATASET_NAME = "auto_scholar_query"


def load_auto_scholar_query(path: str | Path) -> list[EvalQuery]:
    """按源文件顺序加载 AutoScholarQuery，不修改查询或 gold。"""

    source_path = Path(path)
    records = read_jsonl_records(source_path)
    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()
    for line_number, record in records:
        query = parse_auto_scholar_query_record(record, line_number=line_number)
        if query.query_id in seen_ids:
            raise ValueError(
                f"invalid {DATASET_NAME} at line {line_number}: "
                f"duplicate qid {query.query_id}"
            )
        seen_ids.add(query.query_id)
        queries.append(query)
    return queries


def read_jsonl_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise ValueError(f"dataset file not found: {path}")
    if not path.is_file():
        raise ValueError(f"dataset path is not a file: {path}")

    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid {DATASET_NAME} JSONL at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"invalid {DATASET_NAME} at line {line_number}: expected object"
            )
        records.append((line_number, payload))
    return records


def parse_auto_scholar_query_record(
    record: dict[str, Any],
    *,
    line_number: int,
) -> EvalQuery:
    query_id = _required_string(record, "qid", line_number, preserve=True)
    query_text = _required_string(record, "question", line_number, preserve=True)
    titles = _required_string_list(record, "answer", line_number)
    arxiv_ids = _required_string_list(record, "answer_arxiv_id", line_number)
    if not titles or not arxiv_ids:
        raise ValueError(
            f"invalid {DATASET_NAME} at line {line_number}: gold must not be empty"
        )
    if len(titles) != len(arxiv_ids):
        raise ValueError(
            f"invalid {DATASET_NAME} at line {line_number}: "
            "answer and answer_arxiv_id lengths differ"
        )

    source_meta = record.get("source_meta")
    if source_meta is not None and not isinstance(source_meta, dict):
        raise ValueError(
            f"invalid {DATASET_NAME} at line {line_number}: source_meta must be object"
        )

    gold_papers = [
        EvalGoldPaper(
            title=title,
            arxiv_id=arxiv_id,
            relevance_grade=1.0,
            metadata={"gold_index": index, "label_type": "binary_gold"},
        )
        for index, (title, arxiv_id) in enumerate(zip(titles, arxiv_ids, strict=True))
    ]
    return EvalQuery(
        query_id=query_id,
        query=query_text,
        gold_papers=gold_papers,
        metadata={
            "dataset": DATASET_NAME,
            "source_line": line_number,
            "source_meta": dict(source_meta or {}),
            "split": "test",
        },
    )


def _required_string(
    record: dict[str, Any],
    field: str,
    line_number: int,
    *,
    preserve: bool = False,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"invalid {DATASET_NAME} at line {line_number}: missing {field}"
        )
    return value if preserve else value.strip()


def _required_string_list(
    record: dict[str, Any],
    field: str,
    line_number: int,
) -> list[str]:
    value = record.get(field)
    if not isinstance(value, list):
        raise ValueError(
            f"invalid {DATASET_NAME} at line {line_number}: {field} must be a list"
        )
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"invalid {DATASET_NAME} at line {line_number}: "
                f"{field}[{index}] must be non-empty string"
            )
        normalized.append(item)
    return normalized
