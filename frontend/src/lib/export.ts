import type {
  CitationGraph,
  CostReport,
  EvidenceItem,
  PaperIdentifiers,
  RankedPaper,
  SearchRunResultResponse,
  SynthesisOutput,
} from "@/types/api";

export function exportSearchResultAsJson(result: SearchRunResultResponse): void {
  downloadTextFile(
    `scholar-navigator-result-${safeFilename(result.run_id)}.json`,
    JSON.stringify(result, null, 2),
    "application/json;charset=utf-8",
  );
}

export function exportSearchResultAsMarkdown(result: SearchRunResultResponse): void {
  downloadTextFile(
    `scholar-navigator-result-${safeFilename(result.run_id)}.md`,
    searchResultToMarkdown(result),
    "text/markdown;charset=utf-8",
  );
}

export function searchResultToMarkdown(result: SearchRunResultResponse): string {
  const lines: string[] = [
    "# ScholarNavigator Search Result",
    "",
    `- run_id: ${result.run_id}`,
    `- status: ${result.status}`,
    `- partial: ${String(result.partial)}`,
    "",
    "## Query Analysis",
    "",
    codeBlock(result.query_analysis),
    "",
    "## Search Plan",
    "",
    `- max_rounds: ${result.search_plan.max_rounds}`,
    `- source_preferences: ${result.search_plan.source_preferences.join(", ") || "N/A"}`,
    "",
    "### Expanded Queries",
    "",
    ...listItems(result.search_plan.expanded_queries),
    "",
    "## Cost Report",
    "",
    costReportMarkdown(result.cost_report),
    "",
  ];

  if (result.synthesis) {
    lines.push(...synthesisMarkdown(result.synthesis), "");
  }

  lines.push(
    "## Highly Relevant Papers",
    "",
    ...papersMarkdown(result.highly_relevant_papers),
    "",
    "## Partially Relevant Papers",
    "",
    ...papersMarkdown(result.partially_relevant_papers),
    "",
    ...citationGraphMarkdown(result.citation_graph),
    "",
    "## Missing Evidence",
    "",
    ...listItems(result.missing_evidence),
    "",
  );

  return `${lines.join("\n").trim()}\n`;
}

function downloadTextFile(filename: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function safeFilename(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "result";
}

function codeBlock(value: unknown): string {
  return `\`\`\`json\n${JSON.stringify(value, null, 2)}\n\`\`\``;
}

function listItems(values: string[]): string[] {
  if (!values.length) {
    return ["- N/A"];
  }
  return values.map((value) => `- ${markdownText(value)}`);
}

function costReportMarkdown(costReport: CostReport): string {
  return [
    `- api_call_count: ${costReport.api_call_count}`,
    `- logical_search_call_count: ${costReport.logical_search_call_count}`,
    `- search_api_call_count: ${costReport.search_api_call_count}`,
    `- reference_api_call_count: ${costReport.reference_api_call_count}`,
    `- retry_count: ${costReport.retry_count}`,
    `- error_count: ${costReport.error_count}`,
    `- llm_call_count: ${costReport.llm_call_count}`,
    `- estimated_total_tokens: ${costReport.estimated_total_tokens}`,
    `- latency_seconds: ${costReport.latency_seconds}`,
    `- cache_hit_count: ${costReport.cache_hit_count}`,
    `- rate_limit_wait_seconds: ${costReport.rate_limit_wait_seconds}`,
    `- search_rounds: ${costReport.search_rounds}`,
    `- judged_paper_count: ${costReport.judged_paper_count}`,
    `- raw_candidate_count: ${costReport.raw_candidate_count}`,
    `- deduplicated_candidate_count: ${costReport.deduplicated_candidate_count}`,
  ].join("\n");
}

function synthesisMarkdown(synthesis: SynthesisOutput): string[] {
  return [
    "## Citation-backed Synthesis",
    "",
    `- status: ${synthesis.status}`,
    "",
    "### Summary",
    "",
    markdownText(synthesis.answer_summary),
    "",
    "### Key Findings",
    "",
    ...(synthesis.key_findings.length
      ? synthesis.key_findings.map(
          (finding) =>
            `- ${markdownText(finding.text)} ` +
            `(citations: ${finding.citation_keys.join(", ") || "N/A"}; ` +
            `confidence: ${finding.confidence})`,
        )
      : ["- N/A"]),
    "",
    "### Citation Coverage",
    "",
    codeBlock(synthesis.citation_coverage),
    "",
    "### Limitations / Warnings",
    "",
    ...listItems([...synthesis.limitations, ...synthesis.warnings]),
  ];
}

function papersMarkdown(papers: RankedPaper[]): string[] {
  if (!papers.length) {
    return ["No papers."];
  }

  return papers.flatMap((paper) => [
    `### Rank ${paper.rank}: ${markdownText(paper.paper.title)}`,
    "",
    `- authors: ${paper.paper.authors.join(", ") || "N/A"}`,
    `- year: ${paper.paper.year || "N/A"}`,
    `- venue: ${paper.paper.venue || "N/A"}`,
    `- score: ${paper.relevance_score}`,
    `- category: ${paper.category}`,
    `- sources: ${paper.paper.sources.join(", ") || "N/A"}`,
    `- identifiers: ${identifiersText(paper.paper.identifiers)}`,
    "",
    "#### Ranking Reason",
    "",
    markdownText(paper.ranking_reason),
    "",
    "#### Evidence",
    "",
    ...evidenceMarkdown(paper.evidence),
    "",
  ]);
}

function evidenceMarkdown(evidence: EvidenceItem[]): string[] {
  if (!evidence.length) {
    return ["- N/A"];
  }
  return evidence.map(
    (item) =>
      `- ${item.source} (${item.confidence}): ${markdownText(item.text)}`,
  );
}

function identifiersText(identifiers: PaperIdentifiers): string {
  const entries = Object.entries(identifiers).filter(([, value]) => Boolean(value));
  if (!entries.length) {
    return "N/A";
  }
  return entries.map(([key, value]) => `${key}=${String(value)}`).join("; ");
}

function citationGraphMarkdown(graph: CitationGraph): string[] {
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];
  if (!nodes.length && !edges.length) {
    return ["## Citation Graph", "", "No citation graph returned."];
  }

  return [
    "## Citation Graph",
    "",
    `- nodes: ${nodes.length}`,
    `- edges: ${edges.length}`,
    "",
    "### Nodes",
    "",
    ...(nodes.length
      ? nodes.map(
          (node) =>
            `- ${markdownText(node.label)} (${node.id}` +
            `${node.rank ? `; rank ${node.rank}` : ""})`,
        )
      : ["- N/A"]),
    "",
    "### Edges",
    "",
    ...(edges.length
      ? edges.map(
          (edge) =>
            `- ${edge.source} -> ${edge.target} (${edge.relation})`,
        )
      : ["- 当前无引用边/关系边。"]),
  ];
}

function markdownText(value: string): string {
  const normalized = value
    .normalize("NFKC")
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]/gu, (character) =>
      `\\u${character.codePointAt(0)?.toString(16).padStart(4, "0")}`,
    )
    .replace(/\r?\n+/g, " ")
    .trim();
  if (!normalized) {
    return "N/A";
  }
  return normalized.replace(/([\\`*_{}\[\]()<>#+.!|=-])/g, "\\$1");
}
