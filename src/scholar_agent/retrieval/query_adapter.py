"""将逻辑子查询确定性适配为各公开检索源可接受的有界查询。"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from scholar_agent.core.search_schemas import CombinationMode, QueryConstraint


QueryAdapterPolicy = Literal["safe_original", "hybrid", "adaptive"]
DEFAULT_QUERY_ADAPTER_POLICY: QueryAdapterPolicy = "adaptive"
MAX_ADAPTED_QUERIES_PER_SOURCE = 2
MIN_COMPACT_RETENTION_RATIO = 0.5
MAX_OPENALEX_QUERY_LENGTH = 180
MAX_OPENALEX_TERMS = 12
MAX_SEMANTIC_SCHOLAR_QUERY_LENGTH = 180
MAX_SEMANTIC_SCHOLAR_TERMS = 10
MAX_PUBMED_QUERY_LENGTH = 180
MAX_PUBMED_TERMS = 10
MAX_ARXIV_QUERY_LENGTH = 240
MAX_ARXIV_TERMS = 6
MAX_ARXIV_DISJUNCTIVE_TERMS = 8

_ARXIV_FIELD_EXPRESSION = re.compile(
    r"(?:^|[\s(])(?:all|ti|abs|au|cat|jr|id):",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_+#.][A-Za-z0-9]+)*|[\u4e00-\u9fff]+")
_QUOTED_PHRASE_PATTERN = re.compile(r"[\"“”']([^\"“”']{2,})[\"“”']")
_ACADEMIC_QUERY_STOPWORDS = {
    "a", "about", "an", "analysed", "analyzed", "and", "any", "are",
    "attempt", "attempts", "automatic", "automatically", "based", "can",
    "case", "cases", "could", "develop", "developed", "do", "field", "for",
    "focused", "give", "have", "identifying", "in", "information", "into",
    "is", "issue", "issues", "list", "literature", "me", "of", "on", "over",
    "paper", "papers", "part", "parts", "please", "present", "proposed",
    "provide", "providing", "related", "representative", "research", "resource",
    "resources", "scenario", "scenarios", "show", "some", "suitable", "studies",
    "study", "tell", "that", "the", "there", "through", "to", "use", "used",
    "using", "what", "where", "which", "with", "works", "you", "your",
}
ACADEMIC_QUERY_STOPWORDS = frozenset(_ACADEMIC_QUERY_STOPWORDS)


def tokenize_academic_text(value: str) -> list[str]:
    """Tokenize academic search text with the production query-adapter rules."""

    return _TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", str(value)))


class AdaptedQuery(BaseModel):
    original_query: str
    source: str
    query: str
    strategy: str
    equivalent_strategies: list[str] = Field(default_factory=list)
    dropped_terms: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    original_information_terms: list[str] = Field(default_factory=list)
    retained_information_terms: list[str] = Field(default_factory=list)
    retention_ratio: float = 1.0
    protected_terms: list[str] = Field(default_factory=list)


def adapt_query_for_source(
    query: str,
    source: str,
    *,
    constraints: QueryConstraint | None = None,
    combination_mode: CombinationMode = "all",
) -> AdaptedQuery:
    """返回只做必要安全处理的原始查询，供 connector 和保底检索使用。"""

    if combination_mode == "any":
        return _safe_disjunctive(query, _normalize_source(source), constraints)
    del constraints  # 安全查询不得因推断约束而改变用户原始信息。
    return _safe_original(query, _normalize_source(source))


def adapt_queries_for_source(
    query: str,
    source: str,
    *,
    constraints: QueryConstraint | None = None,
    max_queries: int = MAX_ADAPTED_QUERIES_PER_SOURCE,
    policy: QueryAdapterPolicy = DEFAULT_QUERY_ADAPTER_POLICY,
    combination_mode: CombinationMode = "all",
) -> list[AdaptedQuery]:
    """按策略生成“安全原查询保底 + 核心查询补充”的有界变体。"""

    if policy not in ("safe_original", "hybrid", "adaptive"):
        raise ValueError(f"unsupported query adapter policy: {policy}")
    limit = max(0, min(int(max_queries), MAX_ADAPTED_QUERIES_PER_SOURCE))
    if limit == 0:
        return []

    source_key = _normalize_source(source)
    if combination_mode == "any":
        disjunctive = _safe_disjunctive(query, source_key, constraints)
        return [disjunctive] if disjunctive.query else []
    safe = _safe_original(query, source_key)
    if policy == "safe_original" or limit == 1 or not safe.query:
        return [safe]
    if source_key == "arxiv" and _ARXIV_FIELD_EXPRESSION.search(
        _normalize_space(_strip_controls(query))
    ):
        return [safe]

    compact = _compact_core(query, source_key, constraints)
    compact_unsafe = (
        not compact.query
        or compact.retention_ratio < MIN_COMPACT_RETENTION_RATIO
        or "compact_query_protected_terms_removed" in compact.warnings
    )
    if compact_unsafe and policy != "adaptive":
        warnings = _stable(
            [*safe.warnings, *compact.warnings, "compact_query_fallback_to_safe_original"]
        )
        return [safe.model_copy(update={"strategy": "fallback_original", "warnings": warnings})]

    if _query_key(compact.query) == _query_key(safe.query) and policy != "adaptive":
        return [
            safe.model_copy(
                update={
                    "equivalent_strategies": ["safe_original", "compact_core"],
                }
            )
        ]
    return [safe, compact][:limit]


def _safe_original(original_query: str, source: str) -> AdaptedQuery:
    normalized = _normalize_space(unicodedata.normalize("NFKC", _strip_controls(original_query)))
    information_terms = _information_terms(normalized)
    protected_terms = _protected_terms(normalized, None)
    warnings: list[str] = []

    if source == "arxiv" and _ARXIV_FIELD_EXPRESSION.search(normalized):
        adapted = _truncate_at_boundary(normalized, MAX_ARXIV_QUERY_LENGTH)
    else:
        cleaned = _safe_plain_text(normalized)
        max_length = _source_max_length(source)
        prefix = "all:" if source == "arxiv" and cleaned else ""
        adapted = prefix + _truncate_at_boundary(cleaned, max_length - len(prefix))
    if len(adapted) < len(("all:" if source == "arxiv" else "") + _safe_plain_text(normalized)):
        warnings.append("safe_original_truncated")
    if not adapted:
        warnings.append("empty_adapted_query")

    retained = _retained_terms(information_terms, adapted)
    return AdaptedQuery(
        original_query=original_query,
        source=source,
        query=adapted,
        strategy="safe_original",
        warnings=warnings,
        original_information_terms=information_terms,
        retained_information_terms=retained,
        retention_ratio=_retention_ratio(information_terms, retained),
        protected_terms=protected_terms,
    )


def _safe_disjunctive(
    original_query: str,
    source: str,
    constraints: QueryConstraint | None,
) -> AdaptedQuery:
    """将来源无关的 any 术语安全转换为有界 arXiv OR 表达式。"""

    terms = _logical_terms(original_query)[:MAX_ARXIV_DISJUNCTIVE_TERMS]
    required = _explicit_must_include_terms(constraints)
    required_keys = {_normalize_term(value).casefold() for value in required}
    alternatives = [
        term
        for term in terms
        if _normalize_term(term).casefold() not in required_keys
    ]
    removed_required_count = len(terms) - len(alternatives)
    warnings: list[str] = []
    if source != "arxiv":
        fallback = _safe_original(original_query, source)
        return fallback.model_copy(
            update={
                "strategy": "disjunctive_any_fallback",
                "warnings": _stable(
                    [*fallback.warnings, f"combination_mode_any_unsupported:{source}"]
                ),
                "protected_terms": _stable(
                    [*fallback.protected_terms, *required]
                ),
            }
        )

    any_clauses = [
        f"all:{value}"
        for term in alternatives
        if (value := _arxiv_value(term))
    ]
    required_clauses = [
        f"all:{value}"
        for term in required
        if (value := _arxiv_value(term))
    ]
    selected_any: list[str] = []
    for clause in any_clauses:
        candidate_any = [*selected_any, clause]
        candidate = _arxiv_disjunctive_expression(candidate_any, required_clauses)
        if len(candidate) > MAX_ARXIV_QUERY_LENGTH:
            warnings.append("disjunctive_query_truncated")
            break
        selected_any.append(clause)
    adapted = _arxiv_disjunctive_expression(selected_any, required_clauses)
    if removed_required_count:
        warnings.append("disjunctive_duplicate_required_term_removed")
    if not adapted:
        warnings.append("empty_adapted_query")

    information_terms = _stable([*alternatives, *required])
    retained = _retained_terms(information_terms, adapted)
    return AdaptedQuery(
        original_query=original_query,
        source=source,
        query=adapted,
        strategy="disjunctive_any",
        dropped_terms=alternatives[len(selected_any) :],
        warnings=_stable(warnings),
        original_information_terms=information_terms,
        retained_information_terms=retained,
        retention_ratio=_retention_ratio(information_terms, retained),
        protected_terms=_stable(required),
    )


def _compact_core(
    original_query: str,
    source: str,
    constraints: QueryConstraint | None,
) -> AdaptedQuery:
    normalized = _normalize_space(_strip_controls(original_query))
    terms, dropped = _core_terms(normalized, constraints)
    information_terms = _information_terms(normalized, constraints)
    protected_terms = _protected_terms(normalized, constraints)
    max_terms, max_length = _source_bounds(source)
    selected, omitted = _bounded_terms(terms, max_terms)

    if source == "arxiv":
        omitted.extend(selected[4:])
        term_clauses = [
            f"(ti:{value} OR abs:{value})"
            for term in selected[:4]
            if (value := _arxiv_value(term))
        ]
        candidate_clauses = [
            "(" + " AND ".join(term_clauses[index : index + 2]) + ")"
            for index in range(0, len(term_clauses), 2)
            if term_clauses[index : index + 2]
        ]
        clauses: list[str] = []
        for index, clause in enumerate(candidate_clauses):
            candidate = " OR ".join([*clauses, clause])
            if len(candidate) > max_length:
                omitted.extend(selected[index * 2 :])
                break
            clauses.append(clause)
        adapted = " OR ".join(clauses)
    else:
        adapted_terms: list[str] = []
        for term in selected:
            candidate = " ".join([*adapted_terms, term]).strip()
            if len(candidate) > max_length:
                omitted.append(term)
                continue
            adapted_terms.append(term)
        adapted = " ".join(adapted_terms)
    adapted = _truncate_at_boundary(adapted, max_length)
    retained = _retained_terms(information_terms, adapted)
    ratio = _retention_ratio(information_terms, retained)
    warnings: list[str] = []
    if omitted or len(adapted) >= max_length:
        warnings.append("compact_query_truncated")
    if not adapted:
        warnings.append("empty_adapted_query")
    if ratio < MIN_COMPACT_RETENTION_RATIO:
        warnings.append("compact_query_low_information_retention")
    if protected_terms and not _retained_terms(protected_terms, adapted):
        warnings.append("compact_query_protected_terms_removed")

    return AdaptedQuery(
        original_query=original_query,
        source=source,
        query=adapted,
        strategy="compact_core",
        dropped_terms=_stable([*dropped, *omitted]),
        warnings=warnings,
        original_information_terms=information_terms,
        retained_information_terms=retained,
        retention_ratio=ratio,
        protected_terms=protected_terms,
    )


def _core_terms(
    query: str,
    constraints: QueryConstraint | None,
) -> tuple[list[str], list[str]]:
    prioritized = _constraint_terms(constraints, explicit_only=True)
    supplemental = _constraint_terms(constraints, explicit_only=False)
    quoted = [_normalize_term(value) for value in _QUOTED_PHRASE_PATTERN.findall(query)]
    query_tokens = _TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", query))
    kept: list[str] = []
    dropped: list[str] = []
    for token in [*prioritized, *quoted, *query_tokens, *supplemental]:
        normalized = _normalize_term(token)
        if not normalized:
            continue
        if normalized.casefold() in _ACADEMIC_QUERY_STOPWORDS:
            dropped.append(normalized)
            continue
        kept.append(normalized)
    return _stable(kept), _stable(dropped)


def _logical_terms(query: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", _strip_controls(query))
    values: list[str] = []
    for match in re.finditer(r'"([^"\n]+)"|“([^”\n]+)”|([^\s]+)', normalized):
        raw = next((value for value in match.groups() if value is not None), "")
        term = _normalize_term(raw)
        if term:
            values.append(term)
    return _stable(values)


def _explicit_must_include_terms(
    constraints: QueryConstraint | None,
) -> list[str]:
    if constraints is None or "must_include_terms" not in constraints.explicit_fields:
        return []
    return _stable(_normalize_term(value) for value in constraints.must_include_terms)


def _arxiv_disjunctive_expression(
    alternatives: list[str],
    required: list[str],
) -> str:
    clauses: list[str] = []
    if alternatives:
        clauses.append("(" + " OR ".join(alternatives) + ")")
    clauses.extend(required)
    return " AND ".join(clauses)


def _information_terms(
    query: str,
    constraints: QueryConstraint | None = None,
) -> list[str]:
    terms, _ = _core_terms(query, constraints)
    return terms


def _protected_terms(
    query: str,
    constraints: QueryConstraint | None,
) -> list[str]:
    protected = _constraint_terms(constraints, explicit_only=True)
    protected.extend(_normalize_term(value) for value in _QUOTED_PHRASE_PATTERN.findall(query))
    for token in _TOKEN_PATTERN.findall(query):
        normalized = _normalize_term(token)
        if not normalized:
            continue
        has_letter = any(character.isalpha() for character in normalized)
        has_digit = any(character.isdigit() for character in normalized)
        if (
            re.fullmatch(r"[A-Z][A-Z0-9._+-]{1,}", normalized)
            or has_letter and has_digit
            or normalized.isascii()
            and len(normalized) >= 9
            and normalized.casefold() not in _ACADEMIC_QUERY_STOPWORDS
        ):
            protected.append(normalized)
    return _stable(protected)


def _constraint_terms(
    constraints: QueryConstraint | None,
    *,
    explicit_only: bool,
) -> list[str]:
    if constraints is None:
        return []
    values: list[str] = []
    fields = set(constraints.explicit_fields)
    if not explicit_only or "methods" in fields:
        values.extend(constraints.methods)
    if not explicit_only or "datasets" in fields:
        values.extend(constraints.datasets)
    if not explicit_only or "venues" in fields:
        values.extend(constraints.venues)
    if not explicit_only or "paper_types" in fields:
        values.extend(constraints.paper_types)
    if not explicit_only or "domains" in fields:
        values.extend(
            domain.replace("_", " ")
            for domain in constraints.domains
            if domain != "general_science"
        )
    if "must_include_terms" in fields:
        values.extend(constraints.must_include_terms)
    if explicit_only:
        return values
    explicit_values = {
        value.casefold()
        for value in _constraint_terms(constraints, explicit_only=True)
    }
    return [value for value in values if value.casefold() not in explicit_values]


def _source_bounds(source: str) -> tuple[int, int]:
    if source == "arxiv":
        return MAX_ARXIV_TERMS, MAX_ARXIV_QUERY_LENGTH
    if source == "semantic_scholar":
        return MAX_SEMANTIC_SCHOLAR_TERMS, MAX_SEMANTIC_SCHOLAR_QUERY_LENGTH
    if source == "pubmed":
        return MAX_PUBMED_TERMS, MAX_PUBMED_QUERY_LENGTH
    return MAX_OPENALEX_TERMS, MAX_OPENALEX_QUERY_LENGTH


def _source_max_length(source: str) -> int:
    return _source_bounds(source)[1]


def _bounded_terms(terms: list[str], limit: int) -> tuple[list[str], list[str]]:
    return list(terms[:limit]), list(terms[limit:])


def _safe_plain_text(value: str) -> str:
    text = re.sub(r"[(){}\[\]:;!?*^~\\\"'`,|&=<>]+", " ", value)
    text = re.sub(r"[^\w\u4e00-\u9fff+#./\-\s]+", " ", text, flags=re.UNICODE)
    return _normalize_space(text)


def _escape_arxiv_phrase(value: str) -> str:
    return _safe_plain_text(value).replace("/", " ")


def _arxiv_value(value: str) -> str:
    safe = _normalize_space(_escape_arxiv_phrase(value))
    if not safe:
        return ""
    return f'"{safe}"' if " " in safe else safe


def _retained_terms(original_terms: list[str], adapted_query: str) -> list[str]:
    normalized_query = _normalize_term(adapted_query).casefold()
    query_tokens = set(_TOKEN_PATTERN.findall(normalized_query))
    retained: list[str] = []
    for term in original_terms:
        normalized = _normalize_term(term).casefold()
        tokens = _TOKEN_PATTERN.findall(normalized)
        if normalized and tokens and all(token in query_tokens for token in tokens):
            retained.append(term)
    return _stable(retained)


def _retention_ratio(original: list[str], retained: list[str]) -> float:
    if not original:
        return 1.0
    return len({_normalize_term(value).casefold() for value in retained}) / len(
        {_normalize_term(value).casefold() for value in original}
    )


def _truncate_at_boundary(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    shortened = value[:max_length].rstrip()
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0].rstrip()
    return shortened


def _strip_controls(value: str) -> str:
    return "".join(character for character in str(value) if character.isprintable())


def _normalize_term(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff+#.\-/]+", " ", text, flags=re.UNICODE)
    return _normalize_space(text)


def _normalize_space(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _normalize_source(source: str) -> str:
    return source.strip().casefold().replace("-", "_").replace(" ", "_")


def _stable(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _normalize_space(value)
        key = item.casefold()
        if not item or key in seen:
            continue
        result.append(item)
        seen.add(key)
    return result


def _query_key(value: str) -> str:
    return _normalize_space(value).casefold()
