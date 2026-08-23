"use client";

import {
  Activity,
  AlertTriangle,
  BookOpenCheck,
  Brain,
  Clock3,
  Database,
  Download,
  ExternalLink,
  FileText,
  GitBranch,
  Moon,
  Network,
  Send,
  RefreshCw,
  Search,
  Server,
  SlidersHorizontal,
  Sparkles,
  Sun,
  Timer,
  Zap,
} from "lucide-react";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  cancelRealSearchRun,
  createRealSearchRun,
  getHealth,
  getRealSearchRun,
  getRealSearchRunResult,
  getRuntimeConfig,
  streamRealSearchRunEvents,
} from "@/lib/api";
import { exportSearchResultAsJson, exportSearchResultAsMarkdown } from "@/lib/export";
import {
  formatNumber,
  formatScore,
  formatSeconds,
  identifierEntries,
  safeExternalUrl,
} from "@/lib/format";
import { top20PaperKey } from "@/lib/top20-delivery";
import type {
  CostReport,
  RankedPaper,
  RunProfile,
  RuntimeConfigResponse,
  SearchRunCreateRequest,
  SearchRunResultResponse,
  SearchRunStatusResponse,
  StreamEvent,
  SynthesisOutput,
} from "@/types/api";
import { Badge, Button, SectionPanel, SkeletonLine } from "./ui";

const DEFAULT_QUERY =
  "请帮我搜索 2020 年以来关于 LLM reranking 在学术论文检索中的代表性论文，重点关注 ACL、EMNLP、SIGIR。";

const STAGES = [
  {
    key: "query_understanding",
    title: "理解查询",
    titleLines: ["理解", "查询"],
    icon: Brain,
  },
  {
    key: "retrieval",
    title: "检索候选",
    titleLines: ["检索", "候选"],
    icon: Database,
  },
  {
    key: "judgement",
    title: "相关性判断",
    titleLines: ["相关性", "判断"],
    icon: BookOpenCheck,
  },
  {
    key: "reranking",
    title: "重排序",
    titleLines: ["重排序"],
    icon: GitBranch,
  },
  {
    key: "synthesis",
    title: "证据归纳",
    titleLines: ["证据", "归纳"],
    icon: Sparkles,
  },
];

const STAGE_FRAME_PARTS = [
  "top-left-h",
  "top-left-v",
  "top-mid-left",
  "top-mid-right",
  "top-right-h",
  "top-right-v",
  "right-mid",
  "bottom-right-h",
  "bottom-right-v",
  "bottom-mid-right",
  "bottom-mid-left",
  "bottom-left-h",
  "bottom-left-v",
  "left-mid",
] as const;

const PROFILE_LABELS: Record<RunProfile, string> = {
  fast: "快速",
  balanced: "均衡",
  high_recall: "高召回",
  evaluation: "评测",
};

const DEFAULT_CURRENT_YEAR = new Date().getFullYear();
const YEAR_OPTIONS = Array.from(
  { length: DEFAULT_CURRENT_YEAR - 1900 + 1 },
  (_, index) => DEFAULT_CURRENT_YEAR - index,
);

type SourceMode =
  | "recommended"
  | "hybrid_local"
  | "local_hybrid"
  | "local_bm25"
  | "arxiv"
  | "semantic_scholar"
  | "pubmed"
  | "openalex"
  | "all";
type ThemeMode = "dark" | "light";
type StageLatencyItem = {
  stage: string;
  label: string;
  seconds: number;
};
type RunConfigSnapshot = {
  sourcePreferences: string[];
  runProfile: RunProfile;
  topK: number;
  enableQueryEvolution: boolean;
  enableRefchain: boolean;
  enableLlmQueryUnderstanding: boolean;
  enableLlmJudgement: boolean;
};

const SOURCE_MODE_LABELS: Record<SourceMode, string> = {
  recommended: "推荐",
  hybrid_local: "本地+外部",
  local_hybrid: "语义混合",
  local_bm25: "本地索引",
  arxiv: "arXiv",
  semantic_scholar: "Semantic Scholar",
  pubmed: "PubMed",
  openalex: "OpenAlex",
  all: "全部",
};

const SOURCE_MODE_DESCRIPTIONS: Record<SourceMode, string> = {
  recommended: "按查询领域自动选择稳定的公开来源",
  hybrid_local: "优先使用本地论文库，同时补充公开 API",
  local_hybrid: "本地 BM25 与摘要向量检索的 RRF 融合",
  local_bm25: "仅检索已配置的本地 BM25 论文库",
  arxiv: "开放预印本与学术论文库",
  semantic_scholar: "跨学科论文与引用索引",
  pubmed: "生物医学文献数据库",
  openalex: "开放学术实体与文献索引",
  all: "覆盖最大",
};

const SOURCE_MODE_ORDER: SourceMode[] = [
  "recommended",
  "local_hybrid",
  "hybrid_local",
  "local_bm25",
  "arxiv",
  "semantic_scholar",
  "openalex",
  "pubmed",
  "all",
];

const RUN_PROFILE_ORDER: RunProfile[] = ["fast", "balanced", "high_recall", "evaluation"];

const STAGE_LATENCY_LABELS: Record<string, string> = {
  query_understanding: "查询理解",
  retrieval: "候选检索",
  judgement: "相关性判断",
  reranking: "重排序",
  query_evolution: "查询演化",
  refchain: "RefChain",
  synthesis: "证据归纳",
};

const STAGE_LATENCY_ORDER = [
  "query_understanding",
  "retrieval",
  "judgement",
  "reranking",
  "query_evolution",
  "refchain",
  "synthesis",
];

export function ScholarNavigatorApp() {
  const [theme, setTheme] = useState<ThemeMode>("dark");
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [topK, setTopK] = useState(5);
  const [currentYear, setCurrentYear] = useState(DEFAULT_CURRENT_YEAR);
  const [runProfile, setRunProfile] = useState<RunProfile>("fast");
  const [sourceMode, setSourceMode] = useState<SourceMode>("recommended");
  const [enableRefchain, setEnableRefchain] = useState(false);
  const [enableQueryEvolution, setEnableQueryEvolution] = useState(false);
  const [enableLlmQueryUnderstanding, setEnableLlmQueryUnderstanding] = useState(false);
  const [enableLlmJudgement, setEnableLlmJudgement] = useState(false);
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfigResponse | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<SearchRunStatusResponse | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [result, setResult] = useState<SearchRunResultResponse | null>(null);
  const [activeRunConfig, setActiveRunConfig] = useState<RunConfigSnapshot | null>(null);
  const eventSourceCleanup = useRef<(() => void) | null>(null);
  const searchSequence = useRef(0);

  function resetRunUiState() {
    searchSequence.current += 1;
    eventSourceCleanup.current?.();
    eventSourceCleanup.current = null;
    setRunId(null);
    setStatus(null);
    setEvents([]);
    setResult(null);
    setActiveRunConfig(null);
    setIsSubmitting(false);
    setIsCancelling(false);
  }

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  useEffect(() => {
    let cancelled = false;

    async function loadRuntime() {
      try {
        await getHealth();
        const config = await getRuntimeConfig();
        if (!cancelled) {
          setRuntimeConfig(config);
          setSourceMode((current) => {
            if (sourceModeAvailable(current, config)) {
              return current;
            }
            setFormError("所选本地检索源当前未配置，已切换到推荐组合。");
            return "recommended";
          });
          setBackendError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setBackendError("后端服务不可用，请先启动后端真实检索服务。");
        }
      }
    }

    loadRuntime();
    return () => {
      cancelled = true;
      searchSequence.current += 1;
      eventSourceCleanup.current?.();
    };
  }, []);

  async function handleSearch() {
    if (!query.trim()) {
      setFormError("请输入学术查询。");
      return;
    }

    eventSourceCleanup.current?.();
    const sequence = searchSequence.current + 1;
    searchSequence.current = sequence;
    setFormError(null);
    setBackendError(null);
    setIsSubmitting(true);
    setIsCancelling(false);
    setRunId(null);
    setStatus(null);
    setEvents([]);
    setResult(null);
    const runConfigSnapshot: RunConfigSnapshot = {
      sourcePreferences: sourcePreferencesForMode(sourceMode),
      runProfile,
      topK,
      enableQueryEvolution,
      enableRefchain,
      enableLlmQueryUnderstanding,
      enableLlmJudgement,
    };
    setActiveRunConfig(runConfigSnapshot);

    try {
      const created = await createRealSearchRun({
        query,
        locale: "zh-CN",
        constraints: {
          time_range: {
            end_year: currentYear,
          },
          venues: [],
          must_have_terms: [],
          excluded_terms: [],
          datasets: [],
          paper_types: [],
        },
        source_preferences:
          sourceMode === "recommended"
            ? undefined
            : runConfigSnapshot.sourcePreferences,
        run_profile: runConfigSnapshot.runProfile,
        top_k: runConfigSnapshot.topK,
        budgets: buildBudgets(runConfigSnapshot.runProfile),
        options: {
          enable_query_evolution: runConfigSnapshot.enableQueryEvolution,
          enable_refchain: runConfigSnapshot.enableRefchain,
          enable_llm_query_understanding: runConfigSnapshot.enableLlmQueryUnderstanding,
          enable_llm_judgement: runConfigSnapshot.enableLlmJudgement,
          refchain_depth: runConfigSnapshot.enableRefchain ? 1 : 0,
          return_markdown: true,
          return_json: true,
          stream_events: true,
        },
      });

      setRunId(created.run_id);
      setStatus(buildInitialRealStatus(created.run_id, created.status));
      eventSourceCleanup.current = streamRealSearchRunEvents(
        created.run_id,
        (event) => {
          if (searchSequence.current !== sequence) {
            return;
          }
          setEvents((current) => [...current, event]);
        },
        (message) => {
          if (searchSequence.current !== sequence) {
            return;
          }
          setEvents((current) => [
            ...current,
            {
              event: "sse_error",
              payload: { message },
              receivedAt: new Date().toISOString(),
            },
          ]);
        },
      );

      await pollRealSearchRun(created.run_id, sequence);
    } catch (error) {
      if (searchSequence.current === sequence) {
        setBackendError(
          error instanceof Error
            ? error.message
            : "后端服务不可用，请先启动后端真实检索服务。",
        );
        setIsSubmitting(false);
      }
    }
  }

  async function pollRealSearchRun(runId: string, sequence: number) {
    const pollIntervalMs = 800;
    try {
      while (searchSequence.current === sequence) {
        const runStatus = await getRealSearchRun(runId);
        if (searchSequence.current !== sequence) {
          return;
        }
        setStatus(runStatus);

        if (runStatus.status === "failed") {
          // The status endpoint is the authoritative failure contract.  Use
          // it first so a failed run remains explainable even when the result
          // endpoint intentionally refuses to serve a partial/failed result.
          let message = runStatus.error_message || "真实检索失败";
          if (!runStatus.error_message) {
            try {
              await getRealSearchRunResult(runId);
            } catch (error) {
              message = error instanceof Error ? error.message : message;
            }
          }
          if (searchSequence.current === sequence) {
            setBackendError(message);
          }
          return;
        }

        if (runStatus.status === "cancelled") {
          return;
        }

        if (runStatus.status === "succeeded") {
          while (searchSequence.current === sequence) {
            try {
              const runResult = await getRealSearchRunResult(runId);
              if (searchSequence.current === sequence) {
                setResult(runResult);
              }
              return;
            } catch (error) {
              if (error instanceof ApiError && error.status === 409) {
                await sleep(pollIntervalMs);
                continue;
              }
              throw error;
            }
          }
          return;
        }

        await sleep(pollIntervalMs);
      }
    } finally {
      if (searchSequence.current === sequence) {
        setIsSubmitting(false);
      }
    }
  }

  async function handleCancelRealSearch() {
    if (!runId) {
      return;
    }

    const cancellingRunId = runId;
    setIsCancelling(true);
    setBackendError(null);
    try {
      await cancelRealSearchRun(cancellingRunId);
      resetRunUiState();
    } catch (error) {
      setBackendError(
        error instanceof Error
          ? error.message
          : "取消真实检索失败，请稍后重试。",
      );
      setIsCancelling(false);
    }
  }

  const costReport = status?.cost_report ?? result?.cost_report ?? null;

  return (
    <main className="app-shell">
      <div className="workspace space-y-6">
        <Header
          theme={theme}
          onThemeChange={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
        />

        {backendError ? <BackendWarning message={backendError} /> : null}
        {runtimeConfig ? <RuntimeReadiness config={runtimeConfig} /> : null}

        <div className="grid gap-6 xl:grid-cols-[minmax(500px,5.5fr)_minmax(0,4.5fr)]">
          <SearchWorkbench
            query={query}
            topK={topK}
            currentYear={currentYear}
            runProfile={runProfile}
            sourceMode={sourceMode}
            enableRefchain={enableRefchain}
            enableQueryEvolution={enableQueryEvolution}
            enableLlmQueryUnderstanding={enableLlmQueryUnderstanding}
            enableLlmJudgement={enableLlmJudgement}
            runtimeConfig={runtimeConfig}
            isSubmitting={isSubmitting}
            formError={formError}
            onQueryChange={setQuery}
            onTopKChange={setTopK}
            onCurrentYearChange={setCurrentYear}
            onRunProfileChange={setRunProfile}
            onSourceModeChange={setSourceMode}
            onRefchainChange={setEnableRefchain}
            onQueryEvolutionChange={setEnableQueryEvolution}
            onLlmQueryUnderstandingChange={setEnableLlmQueryUnderstanding}
            onLlmJudgementChange={setEnableLlmJudgement}
            onSearch={handleSearch}
          />

          <RunProgress
            runId={runId}
            status={status}
            events={events}
            costReport={costReport}
            runConfig={activeRunConfig}
            isSubmitting={isSubmitting}
            isCancelling={isCancelling}
            onCancelRealSearch={handleCancelRealSearch}
          />
        </div>

        <ResultsPanel result={result} isLoading={isSubmitting && !result} />
      </div>
    </main>
  );
}

function buildInitialRealStatus(
  runId: string,
  status: SearchRunStatusResponse["status"],
): SearchRunStatusResponse {
  const now = new Date().toISOString();
  return {
    run_id: runId,
    status,
    current_stage: status,
    progress: {
      completed_stages: [],
      skipped_stages: [],
      candidate_paper_count: 0,
      judged_paper_count: 0,
    },
    cost_report: emptyCostReport(),
    created_at: now,
    updated_at: now,
  };
}

function emptyCostReport(): CostReport {
  return {
    api_call_count: 0,
    logical_search_call_count: 0,
    search_api_call_count: 0,
    reference_api_call_count: 0,
    retry_count: 0,
    error_count: 0,
    llm_call_count: 0,
    llm_prompt_tokens: 0,
    llm_completion_tokens: 0,
    llm_total_tokens: 0,
    estimated_input_tokens: 0,
    estimated_output_tokens: 0,
    estimated_total_tokens: 0,
    latency_seconds: 0,
    cache_hit_count: 0,
    rate_limit_wait_seconds: 0,
    search_rounds: 0,
    judged_paper_count: 0,
    raw_candidate_count: 0,
    deduplicated_candidate_count: 0,
  };
}

function buildBudgets(runProfile: RunProfile): SearchRunCreateRequest["budgets"] {
  return {
    max_search_rounds: runProfile === "fast" ? 1 : 2,
    max_candidate_papers: runProfile === "high_recall" ? 300 : 200,
    max_llm_calls: 0,
    max_total_tokens: 0,
    max_latency_seconds: runProfile === "fast" ? 45 : 90,
  };
}

function sourcePreferencesForMode(sourceMode: SourceMode): string[] {
  if (sourceMode === "recommended") {
    return [];
  }
  if (sourceMode === "hybrid_local") {
    return ["local_bm25", "openalex", "arxiv", "semantic_scholar"];
  }
  if (sourceMode === "local_hybrid") {
    return ["local_hybrid"];
  }
  if (sourceMode === "local_bm25") {
    return ["local_bm25"];
  }
  if (sourceMode === "arxiv") {
    return ["arxiv"];
  }
  if (sourceMode === "semantic_scholar") {
    return ["semantic_scholar"];
  }
  if (sourceMode === "pubmed") {
    return ["pubmed"];
  }
  if (sourceMode === "openalex") {
    return ["openalex"];
  }
  return ["openalex", "arxiv", "semantic_scholar", "pubmed"];
}

function sourceModeRequiredConnector(sourceMode: SourceMode): string | null {
  if (sourceMode === "local_bm25" || sourceMode === "hybrid_local") {
    return "local_bm25";
  }
  if (sourceMode === "local_hybrid") {
    return "local_hybrid";
  }
  return null;
}

function sourceModeAvailable(
  sourceMode: SourceMode,
  runtimeConfig: RuntimeConfigResponse,
): boolean {
  const required = sourceModeRequiredConnector(sourceMode);
  if (!required) {
    return true;
  }
  return runtimeConfig.connectors.some(
    (connector) => connector.name === required && connector.available,
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function Header({
  theme,
  onThemeChange,
}: {
  theme: ThemeMode;
  onThemeChange: () => void;
}) {
  const switchToLight = () => {
    if (theme !== "light") {
      onThemeChange();
    }
  };
  const switchToDark = () => {
    if (theme !== "dark") {
      onThemeChange();
    }
  };

  return (
    <header className="hero-panel overflow-hidden px-5 py-6 md:px-8 lg:px-10">
      <svg
        className="pointer-events-none absolute inset-0 hidden h-full w-full text-[var(--border-strong)] lg:block"
        viewBox="0 0 1440 230"
        fill="none"
        aria-hidden="true"
      >
        <g opacity="0.12">
          <path d="M0 22H1440M0 54H1440M0 86H1440M0 118H1440M0 150H1440M0 182H1440M0 214H1440" stroke="currentColor" strokeWidth="0.8" strokeDasharray="5 8" />
          <path d="M24 0V230M64 0V230M104 0V230M144 0V230M184 0V230M224 0V230M264 0V230M304 0V230M344 0V230M384 0V230M424 0V230M464 0V230M504 0V230M544 0V230M584 0V230M624 0V230M664 0V230M704 0V230M744 0V230M784 0V230M824 0V230M864 0V230M904 0V230M944 0V230M984 0V230M1024 0V230M1064 0V230M1104 0V230M1144 0V230M1184 0V230M1224 0V230M1264 0V230M1304 0V230M1344 0V230M1384 0V230" stroke="currentColor" strokeWidth="0.8" strokeDasharray="5 8" />
        </g>
        <g opacity="0.15">
          <path d="M54 78L122 42L214 82L308 54L404 102L512 70" stroke="currentColor" strokeWidth="1.4" />
          <path d="M92 176L180 128L294 164L418 116L560 166" stroke="currentColor" strokeWidth="1.2" strokeDasharray="7 8" />
          <path d="M38 132L118 188L228 144L346 190L498 146" stroke="currentColor" strokeWidth="1.1" />
          <circle cx="54" cy="78" r="4" fill="currentColor" />
          <circle cx="122" cy="42" r="4" fill="currentColor" />
          <circle cx="214" cy="82" r="4" fill="currentColor" />
          <circle cx="308" cy="54" r="4" fill="currentColor" />
          <circle cx="404" cy="102" r="4" fill="currentColor" />
          <circle cx="512" cy="70" r="4" fill="currentColor" />
          <circle cx="92" cy="176" r="3.5" fill="currentColor" />
          <circle cx="180" cy="128" r="3.5" fill="currentColor" />
          <circle cx="294" cy="164" r="3.5" fill="currentColor" />
          <circle cx="418" cy="116" r="3.5" fill="currentColor" />
          <circle cx="560" cy="166" r="3.5" fill="currentColor" />
          <circle cx="38" cy="132" r="3" fill="currentColor" />
          <circle cx="228" cy="144" r="3" fill="currentColor" />
          <circle cx="346" cy="190" r="3" fill="currentColor" />
          <circle cx="498" cy="146" r="3" fill="currentColor" />
          <rect x="26" y="38" width="78" height="48" stroke="currentColor" strokeWidth="1.1" />
          <path d="M42 55H84M42 70H76" stroke="currentColor" strokeWidth="1.1" />
          <rect x="260" y="22" width="112" height="58" stroke="currentColor" strokeWidth="1" />
          <path d="M280 42H348M280 58H330" stroke="currentColor" strokeWidth="1" />
          <rect x="456" y="34" width="92" height="46" stroke="currentColor" strokeWidth="1.2" />
          <path d="M474 52H528M474 66H514" stroke="currentColor" strokeWidth="1.2" />
        </g>
        <g opacity="0.18">
          <path d="M584 42L656 82L656 162L584 202L512 162L512 82L584 42Z" stroke="currentColor" strokeWidth="1.2" />
          <path d="M584 42V202M512 82L656 162M656 82L512 162" stroke="currentColor" strokeWidth="1" />
          <path d="M586 118L664 92L744 130L804 98" stroke="currentColor" strokeWidth="1.3" strokeDasharray="6 7" />
          <circle cx="586" cy="118" r="4" fill="currentColor" />
          <circle cx="664" cy="92" r="4" fill="currentColor" />
          <circle cx="744" cy="130" r="4" fill="currentColor" />
          <circle cx="804" cy="98" r="4" fill="currentColor" />
          <rect x="690" y="154" width="96" height="42" stroke="currentColor" strokeWidth="1.1" />
          <path d="M708 171H764M708 184H750" stroke="currentColor" strokeWidth="1.1" />
        </g>
        <g opacity="0.3">
          <path d="M828 52L912 24L1000 68L1000 154L912 198L828 154V52Z" stroke="currentColor" strokeWidth="1.5" />
          <path d="M912 24V112M828 52L912 112L1000 68M828 154L912 112L1000 154M912 198V112" stroke="currentColor" strokeWidth="1.2" />
          <path d="M680 68L763 116L868 82L970 134L1100 78L1244 126L1374 68" stroke="currentColor" strokeWidth="2" />
          <path d="M734 172L822 134L944 178L1072 126L1190 164L1318 118" stroke="currentColor" strokeWidth="1.5" strokeDasharray="8 8" />
          <path d="M1078 44L1164 88L1164 176L1078 218L992 176L992 88L1078 44Z" stroke="currentColor" strokeWidth="1" />
          <path d="M1078 44V218M992 88L1164 176M1164 88L992 176" stroke="currentColor" strokeWidth="0.9" />
          <circle cx="680" cy="68" r="5" fill="currentColor" />
          <circle cx="763" cy="116" r="5" fill="currentColor" />
          <circle cx="868" cy="82" r="5" fill="currentColor" />
          <circle cx="970" cy="134" r="5" fill="currentColor" />
          <circle cx="1100" cy="78" r="5" fill="currentColor" />
          <circle cx="1244" cy="126" r="5" fill="currentColor" />
          <circle cx="1374" cy="68" r="5" fill="currentColor" />
          <circle cx="734" cy="172" r="4" fill="currentColor" />
          <circle cx="822" cy="134" r="4" fill="currentColor" />
          <circle cx="944" cy="178" r="4" fill="currentColor" />
          <circle cx="1072" cy="126" r="4" fill="currentColor" />
          <circle cx="1190" cy="164" r="4" fill="currentColor" />
          <circle cx="1318" cy="118" r="4" fill="currentColor" />
          <rect x="1200" y="40" width="142" height="72" stroke="currentColor" strokeWidth="1.4" />
          <path d="M1222 62H1298M1222 84H1320M1222 96H1282" stroke="currentColor" strokeWidth="1.5" />
          <rect x="1360" y="142" width="54" height="42" stroke="currentColor" strokeWidth="1.4" />
          <path d="M1372 156H1402M1372 170H1392" stroke="currentColor" strokeWidth="1.5" />
        </g>
      </svg>

      <div className="relative z-10 flex flex-col gap-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex min-w-0 items-center gap-7">
            <div
              className="inline-flex h-[124px] w-[124px] shrink-0 items-center justify-center text-[var(--foreground)]"
              aria-hidden="true"
            >
              <Image
                src="/assets/scholarnavigator-compass-logo-black.png"
                alt=""
                width={124}
                height={124}
                className="h-[124px] w-[124px] object-contain"
              />
            </div>
            <div className="min-w-0">
              <h1 className="text-4xl font-black leading-none md:text-6xl">ScholarNavigator</h1>
              <p className="mt-3 text-sm font-bold text-[var(--muted-strong)] md:text-base">
                AI-powered Academic Search & Reranking
              </p>
              <p className="mt-3 max-w-3xl text-base leading-7 text-[var(--muted)] md:text-xl">
                面向科研场景下复杂学术查询的智能论文搜索与推荐。
              </p>
              <div
                className="mt-5 inline-flex max-w-full flex-wrap items-center overflow-hidden rounded-lg border border-[color-mix(in_srgb,var(--border)_72%,var(--foreground)_18%)] bg-[color-mix(in_srgb,var(--surface)_96%,white)] shadow-[0_4px_16px_color-mix(in_srgb,var(--foreground)_8%,transparent)] dark:bg-[color-mix(in_srgb,var(--surface)_92%,white_5%)]"
                aria-label="支持的数据源"
              >
                <span className="inline-flex min-h-10 items-center gap-2 px-4 text-sm font-extrabold text-[var(--foreground)]">
                  <Image
                    src="/assets/source-icons/arxiv-logomark-cropped.png"
                    alt=""
                    width={22}
                    height={24}
                    className="h-6 w-auto shrink-0 object-contain"
                  />
                  arXiv
                </span>
                <span className="h-6 w-px bg-[color-mix(in_srgb,var(--border)_74%,var(--foreground)_12%)]" aria-hidden="true" />
                <span className="inline-flex min-h-10 items-center gap-2 px-4 text-sm font-extrabold text-[var(--foreground)]">
                  <Image
                    src="/assets/source-icons/semantic-scholar-mark.svg"
                    alt=""
                    width={28}
                    height={20}
                    className="h-5 w-7 shrink-0 object-contain"
                  />
                  Semantic Scholar
                </span>
                <span className="h-6 w-px bg-[color-mix(in_srgb,var(--border)_74%,var(--foreground)_12%)]" aria-hidden="true" />
                <span className="inline-flex min-h-10 items-center gap-2 px-4 text-sm font-extrabold text-[var(--foreground)]">
                  <Image
                    src="/assets/source-icons/openalex-icon.svg"
                    alt=""
                    width={22}
                    height={22}
                    className="h-[22px] w-[22px] shrink-0 object-contain dark:invert"
                  />
                  OpenAlex
                </span>
                <span className="h-6 w-px bg-[color-mix(in_srgb,var(--border)_74%,var(--foreground)_12%)]" aria-hidden="true" />
                <span className="inline-flex min-h-10 items-center gap-2 px-4 text-sm font-extrabold text-[var(--foreground)]">
                  <Image
                    src="/assets/source-icons/pubmed-favicon.png"
                    alt=""
                    width={22}
                    height={22}
                    className="h-[22px] w-[22px] shrink-0 object-contain"
                  />
                  PubMed
                </span>
              </div>
            </div>
          </div>

          <div
            className="inline-flex w-fit shrink-0 border-2 border-[color-mix(in_srgb,var(--foreground)_76%,var(--border-strong))] bg-[color-mix(in_srgb,var(--surface-soft)_84%,transparent)]"
            role="group"
            aria-label="主题切换"
          >
            <button
              type="button"
              className={`inline-flex min-h-11 items-center justify-center gap-2 px-4 text-sm font-extrabold text-[var(--muted-strong)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--foreground)] ${
                theme === "light" ? "bg-[var(--surface)] text-[var(--foreground)]" : ""
              }`}
              onClick={switchToLight}
              aria-pressed={theme === "light"}
            >
              <Sun className="h-4 w-4" aria-hidden="true" />
              Light
            </button>
            <button
              type="button"
              className={`inline-flex min-h-11 items-center justify-center gap-2 border-l-2 border-[color-mix(in_srgb,var(--foreground)_52%,var(--border))] px-4 text-sm font-extrabold text-[var(--muted-strong)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--foreground)] ${
                theme === "dark" ? "bg-[var(--surface)] text-[var(--foreground)]" : ""
              }`}
              onClick={switchToDark}
              aria-pressed={theme === "dark"}
            >
              <Moon className="h-4 w-4" aria-hidden="true" />
              Dark
            </button>
          </div>
        </div>

      </div>
    </header>
  );
}

function BackendWarning({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-[color-mix(in_srgb,var(--danger)_55%,var(--border))] bg-[color-mix(in_srgb,var(--danger)_12%,var(--surface))] p-4 text-sm text-[var(--foreground)]"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-[var(--danger)]" aria-hidden="true" />
        <div>
          <p className="font-semibold">{message}</p>
          <p className="mt-1 text-[var(--muted)]">
            默认地址为 http://localhost:8000，可通过 NEXT_PUBLIC_API_BASE_URL 调整。
          </p>
        </div>
      </div>
    </div>
  );
}

function RuntimeReadiness({ config }: { config: RuntimeConfigResponse }) {
  const localBm25 = config.connectors.find((connector) => connector.name === "local_bm25");
  const localHybrid = config.connectors.find((connector) => connector.name === "local_hybrid");
  const completeness = localBm25?.details?.field_completeness ?? null;
  const hybridCompleteness = localHybrid?.details?.field_completeness ?? null;
  const completenessLabels: Record<string, string> = {
    title: "标题",
    abstract: "摘要",
    authors: "作者",
    year: "年份",
    venue: "期刊/会议",
    doi: "DOI",
  };
  const availableSources = config.connectors.filter(
    (connector) => connector.available && !connector.name.startsWith("local_"),
  ).length;

  return (
    <section
      aria-label="运行能力与数据质量"
      className="rounded-lg border-2 border-[color-mix(in_srgb,var(--foreground)_32%,var(--border))] bg-[var(--surface-raised)] p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-black text-[var(--foreground)]">
            <Activity className="h-4 w-4" aria-hidden="true" />
            运行能力与数据质量
          </h2>
          <p className="mt-1 text-xs text-[var(--muted)]">
            实时展示当前环境；诊断信息不会改变检索排序。
          </p>
        </div>
        <span className="text-xs font-bold text-[var(--muted-strong)]">
          {availableSources} 个在线检索源可用
        </span>
      </div>

      <div className="mt-3 grid gap-3 md:grid-cols-3">
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
          <p className="flex items-center gap-2 text-xs font-black text-[var(--foreground)]">
            <Database className="h-4 w-4" aria-hidden="true" />本地 BM25
          </p>
          <p className="mt-1 text-xs text-[var(--muted)]">
            {localBm25?.available
              ? localBm25.reason ?? "已配置"
              : "未配置；可使用在线检索源"}
          </p>
          {completeness ? (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {Object.entries(completenessLabels).map(([field, label]) => {
                const value = completeness[field];
                return (
                  <span
                    key={field}
                    className="rounded border border-[var(--border)] px-1.5 py-0.5 text-[11px] text-[var(--muted-strong)]"
                  >
                    {label} {typeof value === "number" ? `${Math.round(value * 100)}%` : "未知"}
                  </span>
                );
              })}
            </div>
          ) : (
            <p className="mt-2 text-[11px] text-[var(--muted)]">
              字段完整度将在首次建立本地索引后显示。
            </p>
          )}
        </div>

        <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
          <p className="flex items-center gap-2 text-xs font-black text-[var(--foreground)]">
            <GitBranch className="h-4 w-4" aria-hidden="true" />本地语义混合
          </p>
          <p className="mt-1 text-xs text-[var(--muted)]">
            {localHybrid?.available
              ? localHybrid.reason ?? "已配置"
              : "未配置；需要 BM25 语料、摘要语料与向量索引"}
          </p>
          {hybridCompleteness ? (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {Object.entries(completenessLabels).map(([field, label]) => {
                const value = hybridCompleteness[field];
                return (
                  <span
                    key={field}
                    className="rounded border border-[var(--border)] px-1.5 py-0.5 text-[11px] text-[var(--muted-strong)]"
                  >
                    {label} {typeof value === "number" ? `${Math.round(value * 100)}%` : "未知"}
                  </span>
                );
              })}
            </div>
          ) : (
            <p className="mt-2 text-[11px] text-[var(--muted)]">
              字段完整度将在向量索引元数据可用后显示。
            </p>
          )}
        </div>

        <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
          <p className="flex items-center gap-2 text-xs font-black text-[var(--foreground)]">
            <Brain className="h-4 w-4" aria-hidden="true" />LLM 增强
          </p>
          <p className="mt-1 text-xs text-[var(--muted)]">
            {config.llm.available
              ? `Provider：${config.llm.provider}${config.llm.model ? ` · ${config.llm.model}` : ""}`
              : "当前关闭；失败时保留规则检索结果"}
          </p>
          <p className="mt-2 text-[11px] text-[var(--muted)]">
            {config.llm.available ? "是否启用由本次运行配置决定。" : "不会读取或展示任何凭据。"}
          </p>
        </div>
      </div>
    </section>
  );
}

function SearchWorkbench({
  query,
  topK,
  currentYear,
  runProfile,
  sourceMode,
  enableRefchain,
  enableQueryEvolution,
  enableLlmQueryUnderstanding,
  enableLlmJudgement,
  isSubmitting,
  formError,
  onQueryChange,
  onTopKChange,
  onCurrentYearChange,
  onRunProfileChange,
  onSourceModeChange,
  onRefchainChange,
  onQueryEvolutionChange,
  onLlmQueryUnderstandingChange,
  onLlmJudgementChange,
  runtimeConfig,
  onSearch,
}: {
  query: string;
  topK: number;
  currentYear: number;
  runProfile: RunProfile;
  sourceMode: SourceMode;
  enableRefchain: boolean;
  enableQueryEvolution: boolean;
  enableLlmQueryUnderstanding: boolean;
  enableLlmJudgement: boolean;
  isSubmitting: boolean;
  formError: string | null;
  onQueryChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onCurrentYearChange: (value: number) => void;
  onRunProfileChange: (value: RunProfile) => void;
  onSourceModeChange: (value: SourceMode) => void;
  onRefchainChange: (value: boolean) => void;
  onQueryEvolutionChange: (value: boolean) => void;
  onLlmQueryUnderstandingChange: (value: boolean) => void;
  onLlmJudgementChange: (value: boolean) => void;
  runtimeConfig: RuntimeConfigResponse | null;
  onSearch: () => void;
}) {
  const [hoveredRunProfileIndex, setHoveredRunProfileIndex] = useState<number | null>(null);

  const enabledAdvancedCount = [
    enableRefchain,
    enableQueryEvolution,
    enableLlmQueryUnderstanding,
    enableLlmJudgement,
  ].filter(Boolean).length;
  const llmAvailable = runtimeConfig?.llm.available ?? false;

  const handleTopKStep = (delta: number) => {
    onTopKChange(Math.min(100, Math.max(1, topK + delta)));
  };

  const runProfileIndex = Math.max(0, RUN_PROFILE_ORDER.indexOf(runProfile));
  const previewRunProfileIndex = hoveredRunProfileIndex ?? runProfileIndex;
  const runProfileSelectedTransform = `translateX(${runProfileIndex * 100}%)`;
  const runProfilePreviewTransform = `translateX(${previewRunProfileIndex * 100}%)`;

  return (
    <SectionPanel aria-labelledby="search-workbench-title" className="search-workbench-panel h-fit">
      <div className="space-y-6">
        <div className="ow-search">
          <label id="search-workbench-title" className="ow-search__label" htmlFor="query">
            学术检索
          </label>
          <div className="ow-search__field">
            <svg className="ow-search__icon" viewBox="0 0 256 256" aria-hidden="true">
              <path d="M229.66,218.34l-50.07-50.06a88.11,88.11,0,1,0-11.31,11.31l50.06,50.07a8,8,0,0,0,11.32-11.32ZM40,112a72,72,0,1,1,72,72A72.08,72.08,0,0,1,40,112Z" />
            </svg>
            <textarea
              id="query"
              value={query}
              onChange={(event) => onQueryChange(event.target.value)}
              className="ow-search__input"
              placeholder="输入复杂学术查询..."
            />
            <button
              type="button"
              className="ow-search__send"
              onClick={onSearch}
              disabled={isSubmitting}
              aria-label="发送检索请求"
            >
              {isSubmitting ? (
                <RefreshCw className="h-5 w-5 motion-safe:animate-spin" aria-hidden="true" />
              ) : (
                <Send className="h-6 w-6" aria-hidden="true" />
              )}
            </button>
          </div>
          {formError ? <p className="mt-2 px-2 text-sm text-[var(--danger)]">{formError}</p> : null}
        </div>

        <div className="space-y-5">
          <div>
            <h3 className="mb-3 text-sm font-black uppercase tracking-[0.08em] text-[var(--foreground)]">
              检索源
            </h3>
            <div role="radiogroup" aria-label="选择检索数据源" className="source-fancy-row">
              {SOURCE_MODE_ORDER.map((mode) => {
                const selected = sourceMode === mode;
                const available = !runtimeConfig || sourceModeAvailable(mode, runtimeConfig);
                const unavailableReason =
                  mode === "local_hybrid"
                    ? "需要已配置的本地语义索引"
                    : mode === "local_bm25" || mode === "hybrid_local"
                      ? "需要已配置的本地 BM25 语料"
                      : "";
                return (
                  <button
                    key={mode}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    onClick={() => {
                      if (available) onSourceModeChange(mode);
                    }}
                    disabled={!available}
                    aria-disabled={!available}
                    title={available ? SOURCE_MODE_DESCRIPTIONS[mode] : unavailableReason}
                    data-tooltip={available ? SOURCE_MODE_DESCRIPTIONS[mode] : unavailableReason}
                    className={`source-fancy ${selected ? "source-fancy--selected" : ""} ${!available ? "source-fancy--unavailable" : ""}`}
                  >
                    <span className="source-fancy__top-key" aria-hidden="true" />
                    <span className="source-fancy__text">
                      {mode === "recommended" ? "推荐组合" : SOURCE_MODE_LABELS[mode]}
                    </span>
                    <span className="source-fancy__bottom-key-1" aria-hidden="true" />
                    <span className="source-fancy__bottom-key-2" aria-hidden="true" />
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <h3 className="mb-3 text-sm font-black uppercase tracking-[0.08em] text-[var(--foreground)]">
              检索参数
            </h3>
            <div className="flex flex-wrap items-center justify-between gap-x-8 gap-y-4">
              <div className="flex items-center gap-2">
                <span className="text-sm font-black text-[var(--muted-strong)]">返回数量</span>
                <div className="inline-flex min-h-10 overflow-hidden border-2 border-[color-mix(in_srgb,var(--foreground)_72%,var(--border))] bg-[var(--surface)]">
                  <button
                    type="button"
                    onClick={() => handleTopKStep(-1)}
                    disabled={topK <= 1}
                    className="w-10 border-r-2 border-[color-mix(in_srgb,var(--foreground)_46%,var(--border))] text-lg font-black text-[var(--foreground)] transition hover:bg-[var(--surface-soft)] disabled:cursor-not-allowed disabled:opacity-40"
                    aria-label="减少返回数量"
                  >
                    -
                  </button>
                  <input
                    aria-label="返回数量"
                    type="number"
                    min={1}
                    max={100}
                    value={topK}
                    onChange={(event) => onTopKChange(Number(event.target.value))}
                    className="h-10 w-14 border-0 bg-[var(--surface)] p-0 text-center text-sm font-black tabular-nums text-[var(--foreground)] outline-none [appearance:textfield] focus:bg-[var(--surface-soft)] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                  />
                  <button
                    type="button"
                    onClick={() => handleTopKStep(1)}
                    disabled={topK >= 100}
                    className="w-10 border-l-2 border-[color-mix(in_srgb,var(--foreground)_46%,var(--border))] text-lg font-black text-[var(--foreground)] transition hover:bg-[var(--surface-soft)] disabled:cursor-not-allowed disabled:opacity-40"
                    aria-label="增加返回数量"
                  >
                    +
                  </button>
                </div>
              </div>

              <div className="ml-auto flex min-w-0 flex-wrap items-center gap-2">
                <span className="text-sm font-black text-[var(--muted-strong)]">运行模式</span>
                <div
                  role="radiogroup"
                  aria-label="选择运行模式"
                  className="run-profile-radio-inputs"
                  onMouseLeave={() => setHoveredRunProfileIndex(null)}
                  onBlur={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget)) {
                      setHoveredRunProfileIndex(null);
                    }
                  }}
                >
                  <span
                    aria-hidden="true"
                    className="run-profile-bar"
                    style={{ transform: runProfileSelectedTransform }}
                  />
                  <span
                    aria-hidden="true"
                    className="run-profile-slidebar"
                    style={{ transform: runProfilePreviewTransform }}
                  />
                  {RUN_PROFILE_ORDER.map((profile, index) => {
                    const previewed = previewRunProfileIndex === index;
                    return (
                      <label
                        key={profile}
                        className="run-profile-radio"
                        onMouseEnter={() => setHoveredRunProfileIndex(index)}
                        onFocus={() => setHoveredRunProfileIndex(index)}
                      >
                        <input
                          type="radio"
                          name="run-profile"
                          value={profile}
                          checked={runProfile === profile}
                          onChange={() => onRunProfileChange(profile)}
                        />
                        <span
                          className={`run-profile-name ${
                            previewed ? "run-profile-name--previewed" : ""
                          }`}
                        >
                          {PROFILE_LABELS[profile]}模式
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <details className="details-card border-2 border-[color-mix(in_srgb,var(--foreground)_68%,var(--border))] bg-[var(--surface-raised)] p-0 shadow-[2px_2px_0_color-mix(in_srgb,var(--foreground)_14%,transparent)]">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-3 text-sm font-black text-[var(--foreground)] [&::-webkit-details-marker]:hidden">
              <span className="inline-flex items-center gap-2">
                <span className="advanced-details-arrow" aria-hidden="true" />
                高级能力配置
              </span>
              <span className="border border-[color-mix(in_srgb,var(--foreground)_58%,var(--border))] bg-[var(--surface)] px-2 py-1 text-xs font-black text-[var(--muted-strong)]">
                已启用 {enabledAdvancedCount} / 4
              </span>
            </summary>
            <div className="space-y-4 border-t-2 border-[color-mix(in_srgb,var(--foreground)_36%,var(--border))] p-4">
              <div className="flex flex-wrap items-center gap-3">
                <label htmlFor="current-year" className="text-sm font-black text-[var(--muted-strong)]">
                  年份
                </label>
                <YearPicker value={currentYear} onChange={onCurrentYearChange} />
              </div>
              <div className="advanced-toggle-grid">
                <ToggleControl
                  label="RefChain 引用扩展"
                  description="沿高相关论文做单层引用扩展"
                  checked={enableRefchain}
                  onChange={onRefchainChange}
                />
                <ToggleControl
                  label="查询演化"
                  description="基于初始结果生成补充检索式"
                  checked={enableQueryEvolution}
                  onChange={onQueryEvolutionChange}
                />
                <ToggleControl
                  label="LLM 查询理解"
                  description={llmAvailable ? "增强查询解析" : "当前未配置 Provider，保持关闭并使用规则解析"}
                  checked={enableLlmQueryUnderstanding}
                  onChange={onLlmQueryUnderstandingChange}
                  disabled={!llmAvailable}
                />
                <ToggleControl
                  label="LLM 相关性判断"
                  description={llmAvailable ? "判断更强" : "当前未配置 Provider，保持关闭并使用规则判断"}
                  checked={enableLlmJudgement}
                  onChange={onLlmJudgementChange}
                  disabled={!llmAvailable}
                />
              </div>
            </div>
          </details>
        </div>
      </div>
    </SectionPanel>
  );
}

function ToggleControl({
  label,
  description,
  checked,
  onChange,
  disabled = false,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label
      className={`advanced-toggle-card ${checked ? "advanced-toggle-card--checked" : ""}`}
      data-tooltip={description}
    >
      <span className="advanced-toggle-card__content">
        <span className="advanced-toggle-card__label">{label}</span>
      </span>
      <input
        type="checkbox"
        className="advanced-toggle-card__input"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        aria-label={label}
      />
    </label>
  );
}

function YearPicker({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      return;
    }

    function handlePointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="year-picker">
      <button
        id="current-year"
        type="button"
        className="year-select"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        {value}
      </button>
      {open ? (
        <div className="year-picker__menu" role="listbox" aria-label="选择当前年份">
          {YEAR_OPTIONS.map((year) => (
            <button
              key={year}
              type="button"
              role="option"
              aria-selected={year === value}
              className={`year-picker__option ${year === value ? "year-picker__option--selected" : ""}`}
              onClick={() => {
                onChange(year);
                setOpen(false);
              }}
            >
              {year}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function RunProgress({
  runId,
  status,
  events,
  costReport,
  runConfig,
  isSubmitting,
  isCancelling,
  onCancelRealSearch,
}: {
  runId: string | null;
  status: SearchRunStatusResponse | null;
  events: StreamEvent[];
  costReport: CostReport | null;
  runConfig: RunConfigSnapshot | null;
  isSubmitting: boolean;
  isCancelling: boolean;
  onCancelRealSearch: () => void;
}) {
  const completedStages = new Set(status?.progress.completed_stages ?? []);
  events.forEach((event) => {
    if (event.event === "stage_completed" && typeof event.payload.stage === "string") {
      completedStages.add(event.payload.stage);
    }
  });
  if (status?.status === "succeeded") {
    completedStages.add("synthesis");
  }
  const statusClass = `run-status-badge--${status?.status ?? (isSubmitting ? "running" : "idle")}`;
  const canCancelRealSearch =
    Boolean(runId) &&
    Boolean(status && ["queued", "running"].includes(status.status));

  return (
    <SectionPanel aria-labelledby="run-progress-title" className="run-progress-panel">
      <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 id="run-progress-title" className="text-2xl font-black">
            检索运行状态
          </h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {canCancelRealSearch ? (
            <Button
              type="button"
              variant="secondary"
              className="run-cancel-button"
              onClick={onCancelRealSearch}
              disabled={isCancelling}
            >
              {isCancelling ? (
                <RefreshCw className="h-4 w-4 motion-safe:animate-spin" aria-hidden="true" />
              ) : (
                <AlertTriangle className="h-4 w-4" aria-hidden="true" />
              )}
              取消检索
            </Button>
          ) : null}
          <Badge className={`run-status-badge ${statusClass}`}>
            {status ? statusLabel(status.status) : isSubmitting ? "运行中" : "待启动"}
          </Badge>
        </div>
      </div>

      <div className="run-stage-grid">
        {STAGES.map((stage, index) => {
          const Icon = stage.icon;
          const done = completedStages.has(stage.key);
          const active = events.some(
            (event) =>
              event.payload.stage === stage.key &&
              (event.event === "stage_started" || event.event === `${stage.key}_started`),
          );
          const state = done ? "done" : active ? "active" : "pending";
          const connected = done && index < STAGES.length - 1;
          return (
            <div
              key={stage.key}
              className={`run-stage-card run-stage-card--${state} ${
                connected ? "run-stage-card--connected" : ""
              }`}
            >
              <span className="run-stage-card__orbit" aria-hidden="true" />
              <span className="run-stage-card__shine" aria-hidden="true" />
              {STAGE_FRAME_PARTS.map((part) => (
                <span
                  key={part}
                  className={`run-stage-card__frame run-stage-card__frame--${part}`}
                  aria-hidden="true"
                />
              ))}
              <div className="run-stage-card__content">
                <Icon className="run-stage-card__icon" aria-hidden="true" />
                <p className="run-stage-card__title" aria-label={stage.title}>
                  {stage.titleLines.map((line) => (
                    <span key={line}>{line}</span>
                  ))}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {runConfig ? <CompactRunConfig runConfig={runConfig} /> : null}

      <details className="details-card run-diagnostics mt-5">
        <summary className="run-diagnostics__summary">
          运行诊断 / 调试信息
        </summary>
        <div className="mt-4 space-y-5">
          <CostMetrics costReport={costReport} />
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
            <div className="run-diagnostic-panel">
              {status ? (
                <StatusSummaryPanel status={status} />
              ) : (
                <EmptyBlock lines={3} />
              )}
            </div>

            <div className="run-diagnostic-panel">
              <div className="mb-3 flex items-center gap-2">
                <Clock3 className="run-diagnostic-panel__icon" aria-hidden="true" />
                <h3 className="font-semibold">真实检索事件</h3>
              </div>
              {events.length ? (
                <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                  {events.map((event, index) => (
                    <RunEventCard key={`${event.event}-${index}`} event={event} />
                  ))}
                </div>
              ) : (
                <EmptyBlock lines={4} />
              )}
            </div>
          </div>
        </div>
      </details>
    </SectionPanel>
  );
}

function CompactRunConfig({ runConfig }: { runConfig: RunConfigSnapshot }) {
  const chips = [
    `数据源：${runConfig.sourcePreferences.join(" / ")}`,
    `模式：${PROFILE_LABELS[runConfig.runProfile]}`,
    `top_k：${formatNumber(runConfig.topK)}`,
    `查询演化：${formatBoolean(runConfig.enableQueryEvolution)}`,
    `RefChain：${formatBoolean(runConfig.enableRefchain)}`,
    `LLM 查询理解：${formatBoolean(runConfig.enableLlmQueryUnderstanding)}`,
    `LLM 判断：${formatBoolean(runConfig.enableLlmJudgement)}`,
  ];

  return (
    <div className="run-config-strip">
      <div className="mb-3 flex items-center gap-2">
        <SlidersHorizontal className="run-config-strip__icon" aria-hidden="true" />
        <h3 className="font-semibold">本次运行配置</h3>
      </div>
      <div className="flex flex-wrap gap-2">
        {chips.map((chip) => (
          <Badge key={chip} className="run-config-chip">
            <span>{chip}</span>
          </Badge>
        ))}
      </div>
    </div>
  );
}

function CostMetrics({ costReport }: { costReport: CostReport | null }) {
  const metrics = [
    {
      label: "API 调用",
      value: costReport ? formatNumber(costReport.api_call_count) : "--",
      icon: Server,
    },
    {
      label: "估算 Token",
      value: costReport ? formatNumber(costReport.estimated_total_tokens) : "--",
      icon: Zap,
    },
    {
      label: "延迟",
      value: costReport ? formatSeconds(costReport.latency_seconds) : "--",
      icon: Timer,
    },
    {
      label: "缓存命中",
      value: costReport ? formatNumber(costReport.cache_hit_count) : "--",
      icon: Database,
    },
  ];

  return (
    <div className="run-cost-grid">
      {metrics.map((metric) => {
        const Icon = metric.icon;
        return (
          <div key={metric.label} className="run-cost-card">
            <div className="run-cost-card__header">
              <span className="run-cost-card__label">{metric.label}</span>
              <Icon className="run-cost-card__icon" aria-hidden="true" />
            </div>
            <p className="run-cost-card__value metric-value">{metric.value}</p>
          </div>
        );
      })}
    </div>
  );
}

function ResultsPanel({
  result,
  isLoading,
}: {
  result: SearchRunResultResponse | null;
  isLoading: boolean;
}) {
  const visiblePaperCount = result
    ? result.highly_relevant_papers.length + result.partially_relevant_papers.length
    : 0;
  const hasDiagnosticsWithoutCandidates =
    Boolean(result) && visiblePaperCount === 0 && Boolean(result?.missing_evidence.length);

  return (
    <SectionPanel aria-labelledby="results-title">
      <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="mb-2 text-sm font-semibold text-[var(--primary)]">检索结果</p>
          <h2 id="results-title" className="text-2xl font-black">
            论文与证据
          </h2>
          <p className="mt-1 text-sm leading-6 text-[var(--muted)]">
            优先展示可读论文卡片；运行耗时、成本、检索源错误和原始 warning 已收进诊断折叠区。
          </p>
        </div>
        {result ? (
          <div className="flex flex-col gap-3 md:items-end">
            <div className="flex flex-wrap gap-2 md:justify-end">
              <Badge>{result.highly_relevant_papers.length} 篇高度相关</Badge>
              <Badge>{result.partially_relevant_papers.length} 篇部分相关</Badge>
              <Badge>{result.search_plan.source_preferences.join(" / ")}</Badge>
            </div>
            <ResultExportActions result={result} />
          </div>
        ) : null}
      </div>

      {isLoading ? <LoadingResults /> : null}
      {!isLoading && !result ? <EmptyResults /> : null}
      {result ? (
        <div className="space-y-6">
          {hasDiagnosticsWithoutCandidates ? <SourceDiagnosticNotice result={result} /> : null}

          {result.synthesis ? <SynthesisPanel synthesis={result.synthesis} /> : null}

          <CitationGraphPanel result={result} />

          <QuerySummary result={result} />

          <PaperSection
            title="高度相关论文"
            papers={result.highly_relevant_papers}
          />

          <PaperSection
            title="部分相关论文"
            papers={result.partially_relevant_papers}
          />

          <div className="grid gap-4 lg:grid-cols-3">
            <MethodClusters result={result} />
            <Timeline result={result} />
          </div>

          <details className="details-card rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-4">
            <summary className="cursor-pointer text-sm font-bold text-[var(--foreground)]">
              技术诊断 / 调试信息
            </summary>
            <div className="mt-4 space-y-5">
              <StageLatencyPanel result={result} />
              <CostEfficiencyPanel result={result} />
              <RetrievalDiagnosticsPanel result={result} />
              <MissingEvidence result={result} />
            </div>
          </details>
        </div>
      ) : null}
    </SectionPanel>
  );
}

function ResultExportActions({ result }: { result: SearchRunResultResponse }) {
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--surface-raised)] p-3">
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="secondary"
          onClick={() => exportSearchResultAsJson(result)}
          aria-label="导出当前结果为 JSON"
        >
          <Download className="h-4 w-4" aria-hidden="true" />
          导出 JSON
        </Button>
        <Button
          type="button"
          variant="secondary"
          onClick={() => exportSearchResultAsMarkdown(result)}
          aria-label="导出当前结果为 Markdown"
        >
          <FileText className="h-4 w-4" aria-hidden="true" />
          导出 Markdown
        </Button>
      </div>
      <p className="mt-2 max-w-sm text-xs leading-5 text-[var(--muted)]">
        导出内容来自当前页面 result，不会重新检索，也不会上传到后端。
      </p>
    </div>
  );
}

function SourceDiagnosticNotice({ result }: { result: SearchRunResultResponse }) {
  const sourceCount = result.retrieval_diagnostics?.source_stats?.length ?? 0;
  return (
    <div
      role="status"
      className="rounded-lg border border-[color-mix(in_srgb,var(--warning)_60%,var(--border))] bg-[color-mix(in_srgb,var(--warning)_12%,var(--surface))] p-4"
    >
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-[var(--warning)]" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold text-[var(--foreground)]">检索源失败/无候选</h3>
          <p className="mt-1 text-sm text-[var(--muted)]">
            返回结构有效，但当前没有可展示论文。原始错误、限流、超时和检索源诊断已放入下方“技术诊断 / 调试信息”折叠区。
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge>{formatNumber(sourceCount)} 个检索源记录</Badge>
            <Badge>{formatNumber(result.missing_evidence.length)} 条诊断</Badge>
          </div>
        </div>
      </div>
    </div>
  );
}

function StageLatencyPanel({ result }: { result: SearchRunResultResponse }) {
  const latencies = parseStageLatencies(result.missing_evidence);
  if (!latencies.length) {
    return null;
  }

  const maxSeconds = Math.max(...latencies.map((item) => item.seconds), 0.001);
  const totalSeconds = latencies.reduce((total, item) => total + item.seconds, 0);

  return (
    <section
      aria-labelledby="stage-latency-title"
      className="rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-5 shadow-sm"
    >
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <Timer className="h-5 w-5 text-[var(--primary)]" aria-hidden="true" />
              <h3 id="stage-latency-title" className="text-lg font-bold">
              阶段耗时
            </h3>
          </div>
          <p className="text-sm leading-6 text-[var(--muted)]">
            用于定位真实检索 pipeline 中耗时较高的阶段。
          </p>
        </div>
        <Badge>总计 {formatDetailedSeconds(totalSeconds)}</Badge>
      </div>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {latencies.map((item) => {
          const width = `${Math.max(4, Math.round((item.seconds / maxSeconds) * 100))}%`;
          return (
            <div
              key={item.stage}
              className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3"
            >
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-[var(--foreground)]">
                    {item.label}
                  </p>
                  <p className="mt-1 font-mono text-xs text-[var(--muted)]">
                    {item.stage}
                  </p>
                </div>
                <span className="font-mono text-sm font-semibold text-[var(--primary)]">
                  {formatDetailedSeconds(item.seconds)}
                </span>
              </div>
              <div
                className="h-2 overflow-hidden rounded-full bg-[var(--surface-soft)]"
                aria-hidden="true"
              >
                <div
                  className="h-full rounded-full bg-[var(--primary)]"
                  style={{ width }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function CostEfficiencyPanel({ result }: { result: SearchRunResultResponse }) {
  const costReport = result.cost_report;
  const metrics = [
    {
      label: "总 API 调用",
      value: formatNumber(costValue(costReport, "api_call_count")),
      icon: Server,
    },
    {
      label: "检索 API 调用",
      value: formatNumber(costValue(costReport, "search_api_call_count")),
      icon: Search,
    },
    {
      label: "缓存命中",
      value: formatNumber(costValue(costReport, "cache_hit_count")),
      icon: Database,
    },
    {
      label: "LLM 调用",
      value: formatNumber(costValue(costReport, "llm_call_count")),
      icon: Brain,
    },
    {
      label: "LLM 输入 Token",
      value: formatNumber(costValue(costReport, "llm_prompt_tokens")),
      icon: Zap,
    },
    {
      label: "LLM 输出 Token",
      value: formatNumber(costValue(costReport, "llm_completion_tokens")),
      icon: Zap,
    },
    {
      label: "LLM 总 Token",
      value: formatNumber(costValue(costReport, "llm_total_tokens")),
      icon: Zap,
    },
    {
      label: "估算输入 Token",
      value: formatNumber(costValue(costReport, "estimated_input_tokens")),
      icon: Timer,
    },
    {
      label: "估算输出 Token",
      value: formatNumber(costValue(costReport, "estimated_output_tokens")),
      icon: Timer,
    },
    {
      label: "估算总 Token",
      value: formatNumber(costValue(costReport, "estimated_total_tokens")),
      icon: Timer,
    },
  ];

  return (
    <section
      aria-labelledby="cost-efficiency-title"
      className="rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-5 shadow-sm"
    >
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <Zap className="h-5 w-5 text-[var(--primary)]" aria-hidden="true" />
            <h3 id="cost-efficiency-title" className="text-lg font-bold">
              成本与效率
            </h3>
          </div>
          <p className="text-sm leading-6 text-[var(--muted)]">
            展示 API 调用、缓存命中与 LLM token 统计；不包含任何 API key。
          </p>
        </div>
        <Badge>延迟 {formatDetailedSeconds(costValue(costReport, "latency_seconds"))}</Badge>
      </div>

      <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {metrics.map((metric) => {
          const Icon = metric.icon;
          return (
            <div
              key={metric.label}
              className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3"
            >
              <dt className="flex min-h-10 items-start justify-between gap-2 text-xs font-semibold uppercase text-[var(--muted)]">
                <span className="break-words">{metric.label}</span>
                <Icon className="h-4 w-4 shrink-0 text-[var(--primary)]" aria-hidden="true" />
              </dt>
              <dd className="metric-value mt-2 text-lg font-bold text-[var(--foreground)]">
                {metric.value}
              </dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}

function RetrievalDiagnosticsPanel({ result }: { result: SearchRunResultResponse }) {
  const diagnostics = result.retrieval_diagnostics;
  const sourceStats = diagnostics?.source_stats ?? [];
  const hasCounts =
    typeof diagnostics?.raw_count === "number" ||
    typeof diagnostics?.deduplicated_count === "number";

  if (!hasCounts && sourceStats.length === 0) {
    return null;
  }

  return (
    <section
      aria-labelledby="retrieval-diagnostics-title"
      className="rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-5 shadow-sm"
    >
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2">
            <Database className="h-5 w-5 text-[var(--primary)]" aria-hidden="true" />
            <h3 id="retrieval-diagnostics-title" className="text-lg font-bold">
              检索诊断
            </h3>
          </div>
          <p className="text-sm leading-6 text-[var(--muted)]">
            候选规模与检索源状态来自后端输出，用于观察跨源召回、去重和缓存命中情况。
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:min-w-64">
          <DiagnosticMetric label="原始候选" value={formatNumber(diagnostics?.raw_count ?? 0)} />
          <DiagnosticMetric
            label="去重后"
            value={formatNumber(diagnostics?.deduplicated_count ?? 0)}
          />
        </div>
      </div>

      {sourceStats.length ? (
        <div className="grid gap-3 lg:grid-cols-2">
          {sourceStats.map((stat, index) => (
            <div
              key={`${stat.source}-${index}`}
              className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-4"
            >
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <p className="font-semibold text-[var(--foreground)]">{stat.source}</p>
                  <p className="mt-1 font-mono text-xs text-[var(--muted)]">
                    {formatDetailedSeconds(stat.latency_seconds)}
                  </p>
                </div>
                <div className="flex flex-wrap justify-end gap-2">
                  <Badge>{formatNumber(stat.returned_count)} 条返回</Badge>
                  <Badge>{stat.cache_hit ? "缓存命中" : "未命中缓存"}</Badge>
                </div>
              </div>
              {stat.error_message ? (
                <div className="mt-3 rounded-md border border-[color-mix(in_srgb,var(--warning)_55%,var(--border))] bg-[color-mix(in_srgb,var(--warning)_10%,var(--surface))] px-3 py-2 text-sm leading-5 text-[var(--muted-strong)]">
                  {stat.error_message}
                </div>
              ) : (
                <p className="mt-3 text-sm text-[var(--muted)]">无连接器错误。</p>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--muted)]">
          后端未返回检索源统计。
        </p>
      )}
    </section>
  );
}

function DiagnosticMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2">
      <span className="block text-xs font-semibold uppercase text-[var(--muted)]">
        {label}
      </span>
      <span className="mt-1 block font-mono text-lg font-semibold text-[var(--foreground)]">
        {value}
      </span>
    </div>
  );
}

function SynthesisPanel({ synthesis }: { synthesis: SynthesisOutput }) {
  const coverage = synthesis.citation_coverage;
  const evidenceRows = synthesis.evidence_table.slice(0, 6);
  const limitationItems = [...synthesis.limitations, ...synthesis.warnings];
  const coverageMetrics = [
    {
      label: "排序论文",
      value: formatNumber(coverage.ranked_paper_count),
    },
    {
      label: "引用论文",
      value: formatNumber(coverage.cited_paper_count),
    },
    {
      label: "证据行",
      value: formatNumber(coverage.evidence_row_count),
    },
    {
      label: "覆盖率",
      value: formatScore(coverage.coverage_ratio),
    },
    {
      label: "源错误",
      value: formatNumber(coverage.source_error_count),
    },
  ];

  return (
    <section
      aria-labelledby="synthesis-title"
      className="rounded-lg border border-[color-mix(in_srgb,var(--accent)_45%,var(--border))] bg-[color-mix(in_srgb,var(--accent)_7%,var(--surface-raised))] p-5 shadow-sm"
    >
      <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Sparkles className="h-5 w-5 text-[var(--accent)]" aria-hidden="true" />
            <h3 id="synthesis-title" className="text-lg font-bold">
              引文支撑归纳
            </h3>
            <Badge>{synthesis.status}</Badge>
          </div>
          <p className="text-sm leading-6 text-[var(--muted-strong)]">
            {synthesis.answer_summary}
          </p>
          <p className="mt-2 text-xs leading-5 text-[var(--muted)]">
            规则版元数据与证据行归纳；当前 MVP 不代表系统已读取全文 PDF。
          </p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {coverageMetrics.map((metric) => (
          <div
            key={metric.label}
            className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3"
          >
            <p className="text-xs font-semibold uppercase text-[var(--muted)]">
              {metric.label}
            </p>
            <p className="metric-value mt-1 text-lg font-bold text-[var(--foreground)]">
              {metric.value}
            </p>
          </div>
        ))}
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_0.9fr]">
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-4">
          <div className="mb-3 flex items-center gap-2">
            <BookOpenCheck className="h-4 w-4 text-[var(--primary)]" aria-hidden="true" />
            <h4 className="font-semibold">关键发现</h4>
          </div>
          {synthesis.key_findings.length ? (
            <div className="space-y-3">
              {synthesis.key_findings.map((finding, index) => (
                <div
                  key={`${finding.text}-${index}`}
                  className="rounded-md border border-[var(--border)] bg-[var(--surface-raised)] p-3"
                >
                  <p className="text-sm leading-6 text-[var(--muted-strong)]">{finding.text}</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {finding.citation_keys.map((key) => (
                      <Badge key={key}>{key}</Badge>
                    ))}
                    <Badge>{formatScore(finding.confidence)}</Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-[var(--muted)]">暂无可引用 finding。</p>
          )}
        </div>

        <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-4">
          <div className="mb-3 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-[var(--warning)]" aria-hidden="true" />
            <h4 className="font-semibold">限制与提示</h4>
          </div>
          {limitationItems.length ? (
            <div className="space-y-2">
              {limitationItems.slice(0, 8).map((item) => (
                <div
                  key={item}
                  className="rounded-md border border-[var(--border)] bg-[var(--surface-raised)] px-3 py-2 text-sm text-[var(--muted-strong)]"
                >
                  {item}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-[var(--muted)]">当前归纳未返回额外限制。</p>
          )}
        </div>
      </div>

      <div className="mt-5 rounded-md border border-[var(--border)] bg-[var(--surface)] p-4">
        <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h4 className="font-semibold">证据表</h4>
            <p className="text-sm text-[var(--muted)]">展示前 {evidenceRows.length} 条证据行。</p>
          </div>
          <Badge>{formatNumber(synthesis.evidence_table.length)} 行</Badge>
        </div>
        {evidenceRows.length ? (
          <div className="grid gap-3">
            {evidenceRows.map((row) => (
              <div
                key={row.row_id}
                className="rounded-md border border-[var(--border)] bg-[var(--surface-raised)] p-3"
              >
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <Badge>{row.citation_key}</Badge>
                  <Badge>第 {row.rank} 名</Badge>
                  {row.year ? <Badge>{row.year}</Badge> : null}
                  <Badge>{row.evidence_source}</Badge>
                </div>
                <p className="font-semibold leading-snug text-[var(--foreground)]">
                  {row.paper_title}
                </p>
                <p className="mt-2 text-sm leading-6 text-[var(--muted-strong)]">
                  {row.evidence_text}
                </p>
              </div>
            ))}
          </div>
        ) : (
        <p className="text-sm text-[var(--muted)]">暂无证据行。</p>
        )}
      </div>
    </section>
  );
}

function CitationGraphPanel({ result }: { result: SearchRunResultResponse }) {
  const nodes = result.citation_graph?.nodes ?? [];
  const edges = result.citation_graph?.edges ?? [];

  if (!nodes.length && !edges.length) {
    return null;
  }

  return (
    <section
      aria-labelledby="citation-graph-title"
      className="rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-5 shadow-sm"
    >
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Network className="h-5 w-5 text-[var(--primary)]" aria-hidden="true" />
            <h3 id="citation-graph-title" className="text-lg font-bold">
              引用图谱
            </h3>
            <Badge>{formatNumber(nodes.length)} 个节点</Badge>
            <Badge>{formatNumber(edges.length)} 条边</Badge>
          </div>
          <p className="max-w-4xl text-sm leading-6 text-[var(--muted)]">
            当前图谱只展示后端返回的 citation_graph / RefChain metadata；前端不推断未返回的引用关系。
          </p>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h4 className="font-semibold">节点</h4>
            <Badge>{formatNumber(nodes.length)}</Badge>
          </div>
          <div className="max-h-80 space-y-2 overflow-y-auto pr-1">
            {nodes.map((node) => (
              <div
                key={node.id}
                className="rounded-md border border-[var(--border)] bg-[var(--surface-raised)] p-3"
              >
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  {node.rank ? <Badge>第 {node.rank} 名</Badge> : <Badge>未排序</Badge>}
                  <Badge>节点</Badge>
                </div>
                <p className="break-words text-sm font-semibold leading-5 text-[var(--foreground)]">
                  {node.label}
                </p>
                <p className="mt-2 break-all font-mono text-xs leading-5 text-[var(--muted)]">
                  {node.id}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h4 className="font-semibold">关系边</h4>
            <Badge>{formatNumber(edges.length)}</Badge>
          </div>
          {edges.length ? (
            <div className="max-h-80 space-y-2 overflow-y-auto pr-1">
              {edges.map((edge, index) => (
                <div
                  key={`${edge.source}-${edge.target}-${edge.relation}-${index}`}
                  className="rounded-md border border-[var(--border)] bg-[var(--surface-raised)] p-3"
                >
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <Badge>{edge.relation}</Badge>
                    <Badge>第 {index + 1} 条边</Badge>
                  </div>
                  <div className="grid gap-2 text-xs">
                    <GraphEndpoint label="来源节点" value={edge.source} />
                    <GraphEndpoint label="目标节点" value={edge.target} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-md border border-dashed border-[var(--border)] bg-[var(--surface-raised)] p-4">
              <p className="text-sm font-semibold text-[var(--foreground)]">
                当前无引用边/关系边
              </p>
              <p className="mt-1 text-sm leading-6 text-[var(--muted)]">
                后端返回了图谱节点，但未返回引用关系边。真实检索在未启用
                RefChain 或没有可用引用元数据时可能出现这种状态。
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function parseStageLatencies(missingEvidence: string[]): StageLatencyItem[] {
  const byStage = new Map<string, number>();
  missingEvidence.forEach((item) => {
    const match = /^stage_latency:([^:]+):(.+)$/.exec(item.trim());
    if (!match) {
      return;
    }
    const stage = match[1];
    const seconds = Number(match[2]);
    if (!stage || !Number.isFinite(seconds) || seconds < 0) {
      return;
    }
    byStage.set(stage, seconds);
  });

  return Array.from(byStage.entries())
    .map(([stage, seconds]) => ({
      stage,
      label: STAGE_LATENCY_LABELS[stage] ?? stage,
      seconds,
    }))
    .sort((left, right) => {
      const leftIndex = STAGE_LATENCY_ORDER.indexOf(left.stage);
      const rightIndex = STAGE_LATENCY_ORDER.indexOf(right.stage);
      const normalizedLeft = leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex;
      const normalizedRight = rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex;
      if (normalizedLeft !== normalizedRight) {
        return normalizedLeft - normalizedRight;
      }
      return left.stage.localeCompare(right.stage);
    });
}

function formatDetailedSeconds(seconds: number): string {
  if (seconds < 1) {
    return `${seconds.toFixed(3)}s`;
  }
  if (seconds < 10) {
    return `${seconds.toFixed(2)}s`;
  }
  return `${seconds.toFixed(1)}s`;
}

function GraphEndpoint({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2">
      <span className="block text-xs font-semibold uppercase text-[var(--muted)]">
        {label}
      </span>
      <span className="mt-1 block break-all font-mono text-xs leading-5 text-[var(--foreground)]">
        {value}
      </span>
    </div>
  );
}

function QuerySummary({ result }: { result: SearchRunResultResponse }) {
  const constraints = result.query_analysis.constraints;
  const timeRange = asRecord(constraints.time_range);
  const constraintGroups = [
    ["方法", asStringArray(constraints.methods)],
    ["数据集", asStringArray(constraints.datasets)],
    ["必须包含", asStringArray(constraints.must_have_terms)],
    ["排除", asStringArray(constraints.excluded_terms)],
    ["领域", asStringArray(constraints.domains)],
    ["论文类型", asStringArray(constraints.paper_types)],
    ["venue", asStringArray(constraints.venues)],
  ].filter(([, values]) => values.length > 0) as Array<[string, string[]]>;
  const facets = result.search_plan.query_planning.facets;

  return (
    <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
      <div className="panel-soft rounded-lg p-4">
        <h3 className="mb-3 font-semibold">查询理解</h3>
        <div className="flex flex-wrap gap-2">
          <Badge>{result.query_analysis.intent_type}</Badge>
          <Badge>{result.query_analysis.domain}</Badge>
          {result.query_analysis.research_topics.map((topic) => (
            <Badge key={topic}>{topic}</Badge>
          ))}
        </div>
        <div className="mt-4 rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--muted)]">
            已解析约束
          </p>
          <div className="space-y-2 text-sm">
            {timeRange ? (
              <ConstraintRow
                label="时间"
                value={
                  String(timeRange.label ||
                    [timeRange.start_year, timeRange.end_year].filter(Boolean).join("–") ||
                    "已指定")
                }
              />
            ) : null}
            {constraintGroups.map(([label, values]) => (
              <ConstraintRow key={label} label={label} value={values.join(" · ")} />
            ))}
            {!timeRange && constraintGroups.length === 0 ? (
              <p className="text-[var(--muted)]">未识别到显式约束，使用主题相关性检索。</p>
            ) : null}
          </div>
        </div>
      </div>
      <div className="panel-soft rounded-lg p-4">
        <h3 className="mb-3 font-semibold">扩展检索式</h3>
        <div className="space-y-2">
          {result.search_plan.expanded_queries.map((expandedQuery, index) => (
            <div key={`${expandedQuery}-${index}`} className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3 text-sm">
              <span className="mr-2 font-semibold text-[var(--primary)]">{index + 1}</span>
              {expandedQuery}
            </div>
          ))}
        </div>
        {facets.length ? (
          <details className="mt-3 rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
            <summary className="cursor-pointer text-sm font-semibold">规划依据（{facets.length} 个 facets）</summary>
            <div className="mt-3 space-y-2">
              {facets.map((facet, index) => (
                <div key={`${facet.facet_type}-${index}`} className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge>{facet.facet_type}</Badge>
                  <span className="text-[var(--muted-strong)]">{facet.terms.join(" · ")}</span>
                  <span className="text-xs text-[var(--muted)]">
                    {facet.source} · 置信度 {facet.confidence.toFixed(2)}{facet.required ? " · 必需" : ""}
                  </span>
                </div>
              ))}
            </div>
          </details>
        ) : null}
      </div>
    </div>
  );
}

function ConstraintRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-3">
      <span className="w-16 shrink-0 font-semibold text-[var(--muted)]">{label}</span>
      <span className="break-words text-[var(--muted-strong)]">{value}</span>
    </div>
  );
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function PaperSection({
  title,
  description,
  papers,
}: {
  title: string;
  description?: string;
  papers: RankedPaper[];
}) {
  return (
    <section aria-label={title}>
      <div className="mb-3">
        <h3 className="text-lg font-bold">{title}</h3>
        {description ? <p className="text-sm text-[var(--muted)]">{description}</p> : null}
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {papers.map((paper) => (
          <PaperCard key={top20PaperKey(paper)} paper={paper} />
        ))}
      </div>
    </section>
  );
}

function PaperCard({ paper }: { paper: RankedPaper }) {
  const identifiers = identifierEntries(paper.paper.identifiers);

  return (
    <article className="card paper-card result-paper-card">
      <div className="mb-2 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge>第 {paper.rank} 名</Badge>
              <Badge>{paper.paper.year || "年份未知"}</Badge>
              {paper.paper.venue ? <Badge>{paper.paper.venue}</Badge> : null}
              <Badge>相关性 {formatScore(paper.relevance_score)}</Badge>
              <Badge>{categoryLabel(paper.category)}</Badge>
            </div>
            <PaperActionLinks urls={paper.paper.urls} />
          </div>
          <h4 className="card__title">
            <span className="card__title-text">{paper.paper.title}</span>
          </h4>
          <p className="card__content mt-1">
            {paper.paper.authors.length ? paper.paper.authors.join(", ") : "作者信息暂缺"}
          </p>
        </div>
      </div>

      <details className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3">
        <summary className="cursor-pointer text-sm font-bold text-[var(--foreground)]">
          摘要
        </summary>
        <p className="mt-3 text-sm leading-6 text-[var(--muted-strong)]">
          {paper.paper.abstract || "当前结果未返回摘要。"}
        </p>
      </details>

      <details className="mt-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3">
        <summary className="cursor-pointer text-sm font-bold text-[var(--foreground)]">
          排序依据
        </summary>
        <p className="mt-3 text-sm leading-6 text-[var(--muted-strong)]">
          {paper.ranking_reason || "当前结果未提供排序说明。"}
        </p>
        {paper.matched_constraints.length ? (
          <div className="mt-3 flex flex-wrap gap-2">
            {paper.matched_constraints.map((constraint) => (
              <Badge key={constraint}>{constraint}</Badge>
            ))}
          </div>
        ) : null}
      </details>

      {paper.quality_policy ? (
        <div className="mt-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 text-sm">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <p className="font-semibold">质量信号</p>
            <Badge>{paper.quality_policy}</Badge>
            {paper.quality_score != null ? (
              <Badge>质量分 {formatScore(paper.quality_score)}</Badge>
            ) : null}
            {paper.quality_contribution != null ? (
              <Badge>贡献 {paper.quality_contribution.toFixed(4)}</Badge>
            ) : null}
          </div>
          <p className="text-[var(--muted)]">
            {paper.quality_rank_change_reason ||
              "质量信号仅作受限、可解释的辅助，不代表相关性或撤稿结论。"}
          </p>
          {paper.quality_signals?.length ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {paper.quality_signals.map((signal) => (
                <Badge key={signal.name}>
                  {signal.name}: {signal.state}
                </Badge>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {paper.evidence.length ? (
        <div className="mt-4 space-y-2">
          <p className="text-sm font-semibold">证据</p>
          {paper.evidence.map((item) => (
            <div key={`${item.source}-${item.text}`} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 text-sm">
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <Badge>{item.source}</Badge>
                <Badge>置信度 {formatScore(item.confidence)}</Badge>
              </div>
              <p className="text-[var(--muted)]">{item.text}</p>
            </div>
          ))}
        </div>
      ) : null}

      {paper.paper.full_text_evidence.length ? (
        <FullTextEvidenceSection documents={paper.paper.full_text_evidence} />
      ) : null}

      {identifiers.length ? (
        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {identifiers.map(([label, value]) => (
            <div key={`${label}-${value}`} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs">
              <span className="block font-semibold text-[var(--muted)]">{label}</span>
              <span className="mt-1 block break-words text-[var(--foreground)]">{value}</span>
            </div>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function FullTextEvidenceSection({
  documents,
}: {
  documents: RankedPaper["paper"]["full_text_evidence"];
}) {
  return (
    <div className="mt-4 space-y-2" aria-label="全文证据">
      <p className="text-sm font-semibold">全文证据与定位</p>
      {documents.map((document, documentIndex) => {
        const sourceUrl = safeExternalUrl(document.source_url);
        return (
          <details
            key={`${document.content_sha256}-${documentIndex}`}
            className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3"
          >
            <summary className="cursor-pointer text-sm font-bold text-[var(--foreground)]">
              证据文档 {documentIndex + 1} · {document.paragraphs.length} 个段落
            </summary>
            <div className="mt-3 space-y-3 text-sm">
              <dl className="grid gap-2 sm:grid-cols-2">
                <div>
                  <dt className="font-semibold text-[var(--muted)]">许可</dt>
                  <dd className="mt-1 break-words text-[var(--foreground)]">{document.license_id}</dd>
                </div>
                <div>
                  <dt className="font-semibold text-[var(--muted)]">内容 SHA-256</dt>
                  <dd className="mt-1 break-all font-mono text-xs text-[var(--foreground)]">
                    {document.content_sha256}
                  </dd>
                </div>
              </dl>
              <div>
                <p className="font-semibold text-[var(--muted)]">来源</p>
                {sourceUrl ? (
                  <a
                    href={sourceUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-1 inline-flex break-all text-[var(--primary)] underline-offset-2 hover:underline"
                  >
                    {sourceUrl}
                  </a>
                ) : (
                  <p className="mt-1 text-[var(--muted)]">来源地址未通过安全校验</p>
                )}
              </div>
              <div className="space-y-2">
                {document.paragraphs.map((paragraph) => (
                  <div
                    key={paragraph.evidence_id}
                    className="rounded-md border border-[var(--border)] bg-[var(--background)] p-3"
                  >
                    <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-[var(--muted)]">
                      <Badge>段落 {paragraph.paragraph_index + 1}</Badge>
                      <span>字符 {paragraph.start_char}–{paragraph.end_char}</span>
                      <span className="font-mono">段落 SHA-256 {paragraph.text_sha256}</span>
                    </div>
                    <p className="whitespace-pre-wrap leading-6 text-[var(--muted-strong)]">
                      {paragraph.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </details>
        );
      })}
    </div>
  );
}

function PaperActionLinks({ urls }: { urls: RankedPaper["paper"]["urls"] }) {
  const landingPage = safeExternalUrl(urls.landing_page);
  const pdf = safeExternalUrl(urls.pdf);
  if (!landingPage && !pdf) {
    return null;
  }

  return (
    <div className="paper-action-row">
      {landingPage ? (
        <a
          href={landingPage}
          target="_blank"
          rel="noreferrer"
          className="paper-action-link"
        >
          <ExternalLink className="h-4 w-4" aria-hidden="true" />
          打开论文页
        </a>
      ) : null}
      {pdf ? (
        <a
          href={pdf}
          target="_blank"
          rel="noreferrer"
          className="paper-action-link"
        >
          <FileText className="h-4 w-4" aria-hidden="true" />
          PDF
        </a>
      ) : null}
    </div>
  );
}

function MethodClusters({ result }: { result: SearchRunResultResponse }) {
  return (
    <div className="panel-soft rounded-lg p-4">
      <h3 className="mb-3 font-semibold">方法聚类</h3>
      <div className="space-y-3">
        {result.method_clusters.map((cluster) => (
          <div key={cluster.name} className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
            <p className="font-semibold">{cluster.name}</p>
            <p className="mt-1 text-sm text-[var(--muted)]">{cluster.summary}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {cluster.paper_ranks.map((rank) => (
                <Badge key={rank}>第 {rank} 名</Badge>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Timeline({ result }: { result: SearchRunResultResponse }) {
  return (
    <div className="panel-soft rounded-lg p-4">
      <h3 className="mb-3 font-semibold">时间线</h3>
      <div className="space-y-3">
        {result.timeline.map((item) => (
          <div key={item.year} className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
            <p className="metric-value text-lg font-bold text-[var(--primary)]">{item.year}</p>
            <p className="mt-1 text-sm text-[var(--muted)]">{item.summary}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {item.paper_ranks.map((rank) => (
                <Badge key={rank}>第 {rank} 名</Badge>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MissingEvidence({ result }: { result: SearchRunResultResponse }) {
  return (
    <div className="panel-soft rounded-lg p-4">
      <h3 className="mb-3 font-semibold">原始提示与证据缺口</h3>
      <p className="mb-3 text-sm leading-6 text-[var(--muted)]">
        这里集中展示 503、429、timeout、cooldown、source_error 等后端原始诊断，默认折叠以避免干扰主要阅读。
      </p>
      <div className="space-y-2">
        {result.missing_evidence.map((item) => (
          <div key={item} className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3 text-sm text-[var(--muted)]">
            {item}
          </div>
        ))}
      </div>
    </div>
  );
}

type RunEventSummary = {
  title: string;
  description: string;
  chips: string[];
};

function RunEventCard({ event }: { event: StreamEvent }) {
  const summary = describeRunEvent(event);

  return (
    <div className="run-event-card">
      <div className="run-event-card__header">
        <div className="run-event-card__copy">
          <p className="run-event-card__title">{summary.title}</p>
          <p className="run-event-card__description">{summary.description}</p>
        </div>
        <div className="run-event-card__chips" aria-label="事件标签">
          {summary.chips.map((chip) => (
            <Badge key={chip}>{chip}</Badge>
          ))}
        </div>
      </div>
      <details className="run-event-card__raw">
        <summary>原始数据</summary>
        <pre>{JSON.stringify(event.payload, null, 2)}</pre>
      </details>
    </div>
  );
}

function describeRunEvent(event: StreamEvent): RunEventSummary {
  const payload = event.payload;
  const stage = readString(payload.stage);
  const source = readString(payload.source) ?? readString(payload.connector);

  if (event.event === "run_started") {
    const query = readString(payload.query);
    return {
      title: "检索任务已创建",
      description: query ? `已提交查询：${truncateText(query, 72)}` : "已提交真实检索任务，等待后端执行。",
      chips: ["排队中"],
    };
  }

  if (event.event === "stage_started" || event.event === `${stage}_started`) {
    const stageName = stageDisplayLabel(stage);
    return {
      title: `${stageName}开始`,
      description: `系统正在执行“${stageName}”阶段。`,
      chips: ["阶段开始", stageName],
    };
  }

  if (event.event === "stage_completed" || event.event === `${stage}_completed`) {
    return describeStageCompletedEvent(payload, stage);
  }

  if (event.event.endsWith("_skipped")) {
    const stageName = stageDisplayLabel(stage);
    return {
      title: `${stageName}已跳过`,
      description: `本次运行未执行“${stageName}”阶段。`,
      chips: ["阶段跳过", stageName],
    };
  }

  if (event.event === "connector_completed") {
    const sourceName = sourceDisplayLabel(source);
    const returnedCount = readNumber(payload.returned_count);
    const latency = readNumber(payload.latency_seconds);
    const cacheHit = readBoolean(payload.cache_hit);
    const hasError = Boolean(readString(payload.error_message));
    const parts = [
      returnedCount !== null ? `返回 ${formatNumber(returnedCount)} 篇候选` : "检索源已返回",
      latency !== null ? `耗时 ${formatDetailedSeconds(latency)}` : null,
      cacheHit === true ? "命中缓存" : cacheHit === false ? "未命中缓存" : null,
    ].filter(Boolean);

    return {
      title: `${sourceName} 检索完成`,
      description: hasError
        ? `${sourceName} 返回了诊断信息，本次结果可能不完整；详情可展开原始数据查看。`
        : parts.join("，") || `${sourceName} 已完成候选检索。`,
      chips: [
        sourceName,
        returnedCount !== null ? `${formatNumber(returnedCount)} 篇` : "已返回",
        ...(cacheHit ? ["缓存命中"] : []),
        ...(hasError ? ["有诊断"] : []),
      ],
    };
  }

  if (event.event === "warning") {
    const message = readString(payload.message);
    return {
      title: "运行提示",
      description: warningDescription(message),
      chips: ["提示"],
    };
  }

  if (event.event === "cost_updated") {
    const costReport = readRecord(payload.cost_report);
    const apiCalls = readNumber(costReport?.api_call_count);
    const cacheHits = readNumber(costReport?.cache_hit_count);
    const totalTokens = readNumber(costReport?.llm_total_tokens) ?? readNumber(costReport?.estimated_total_tokens);
    const parts = [
      apiCalls !== null ? `API 调用 ${formatNumber(apiCalls)} 次` : null,
      cacheHits !== null ? `缓存命中 ${formatNumber(cacheHits)} 次` : null,
      totalTokens !== null ? `Token ${formatNumber(totalTokens)}` : null,
    ].filter(Boolean);

    return {
      title: "成本统计已更新",
      description: parts.join("，") || "本次运行的调用次数和成本统计已更新。",
      chips: ["成本"],
    };
  }

  if (event.event === "run_completed") {
    const status = readString(payload.status);
    const statusText = status ? runStatusText(status) : "已结束";
    return {
      title: `检索任务${statusText}`,
      description: status === "failed"
        ? "任务执行失败，后端错误详情可展开原始数据查看。"
        : status === "cancelled"
          ? "任务已取消，前端会回到可重新检索的状态。"
          : "真实检索流程已结束，可以查看结果与诊断信息。",
      chips: [statusText],
    };
  }

  if (event.event === "error") {
    return {
      title: "运行出错",
      description: "后端返回错误，详情可展开原始数据查看。",
      chips: ["错误"],
    };
  }

  if (event.event === "sse_error") {
    return {
      title: "事件连接异常",
      description: "前端事件流连接出现异常，检索任务本身可能仍在后端运行。",
      chips: ["连接"],
    };
  }

  return {
    title: eventNameLabel(event.event),
    description: "收到一条运行事件，详情可展开原始数据查看。",
    chips: [eventNameLabel(event.event)],
  };
}

function describeStageCompletedEvent(
  payload: Record<string, unknown>,
  stage: string | null,
): RunEventSummary {
  const stageName = stageDisplayLabel(stage);
  const chips = ["阶段完成", stageName];

  if (stage === "retrieval") {
    const candidateCount = readNumber(payload.candidate_paper_count);
    const searchApiCalls = readNumber(payload.search_api_call_count);
    const parts = [
      candidateCount !== null ? `保留 ${formatNumber(candidateCount)} 篇候选论文` : "候选检索已完成",
      searchApiCalls !== null ? `检索 API 调用 ${formatNumber(searchApiCalls)} 次` : null,
    ].filter(Boolean);
    return {
      title: "候选检索完成",
      description: parts.join("，") || "候选检索阶段已完成。",
      chips,
    };
  }

  if (stage === "judgement") {
    const judgedCount = readNumber(payload.judged_paper_count);
    return {
      title: "相关性判断完成",
      description: judgedCount !== null
        ? `系统已完成 ${formatNumber(judgedCount)} 篇论文的相关性判断。`
        : "系统已完成候选论文的相关性判断。",
      chips,
    };
  }

  if (stage === "reranking") {
    const topK = readNumber(payload.top_k);
    return {
      title: "重排序完成",
      description: topK !== null ? `系统已完成排序，并按 top_k=${formatNumber(topK)} 输出结果。` : "系统已完成候选论文重排序。",
      chips,
    };
  }

  if (stage === "synthesis") {
    return {
      title: "证据归纳完成",
      description: "系统已完成结构化证据归纳和结果整理。",
      chips,
    };
  }

  if (stage === "query_understanding") {
    return {
      title: "查询理解完成",
      description: "系统已完成查询解析，准备进入候选检索。",
      chips,
    };
  }

  return {
    title: `${stageName}完成`,
    description: `“${stageName}”阶段已完成。`,
    chips,
  };
}

function stageDisplayLabel(stage: string | null): string {
  if (!stage) {
    return "未知阶段";
  }
  return STAGE_LATENCY_LABELS[stage] ?? stage;
}

function sourceDisplayLabel(source: string | null): string {
  const labels: Record<string, string> = {
    arxiv: "arXiv",
    semantic_scholar: "Semantic Scholar",
    openalex: "OpenAlex",
    pubmed: "PubMed",
    local_hybrid: "语义混合",
    local_bm25: "本地索引",
  };
  return source ? labels[source] ?? source : "检索源";
}

function warningDescription(message: string | null): string {
  if (!message) {
    return "后端返回一条运行提示，详情可展开原始数据查看。";
  }
  if (message.includes("llm_query_understanding_used")) {
    return "本次运行使用了 LLM 查询理解。";
  }
  if (message.includes("llm_judgement_used")) {
    return "本次运行使用了 LLM 相关性判断。";
  }
  if (message.includes("llm_query_understanding_disabled")) {
    return "LLM 查询理解未启用，系统使用规则版查询解析。";
  }
  if (message.includes("llm_judgement_disabled")) {
    return "LLM 相关性判断未启用，系统使用规则版相关性判断。";
  }
  if (message.includes("llm_query_understanding_failed")) {
    return "LLM 查询理解失败，系统已回退到规则版查询解析。";
  }
  if (message.includes("llm_judgement_failed")) {
    return "LLM 相关性判断失败，系统已回退到规则版相关性判断。";
  }
  if (message.includes("source_cooldown_skip")) {
    return "某个检索源处于短暂冷却期，本次已跳过该源。";
  }
  if (message.includes("subquery_skipped_by_limit")) {
    return "快速模式下跳过了部分扩展查询，以控制请求次数和延迟。";
  }
  if (message.includes("stage_latency")) {
    return "后端记录了一条阶段耗时诊断。";
  }
  return "后端返回一条运行提示，详情可展开原始数据查看。";
}

function runStatusText(status: string): string {
  const labels: Record<string, string> = {
    queued: "排队中",
    running: "运行中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
  };
  return labels[status] ?? status;
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function truncateText(value: string, maxLength: number): string {
  return value.length > maxLength ? `${value.slice(0, maxLength)}…` : value;
}

function StatusSummaryPanel({ status }: { status: SearchRunStatusResponse }) {
  const items = [
    {
      label: "当前阶段",
      value: status.current_stage,
    },
    {
      label: "候选数",
      value: formatNumber(status.progress.candidate_paper_count),
    },
    {
      label: "已判断论文",
      value: formatNumber(status.progress.judged_paper_count),
    },
    {
      label: "完成阶段数",
      value: formatNumber(status.progress.completed_stages.length),
    },
  ];

  return (
    <div className="status-summary-card">
      <div className="status-summary-card__header">
        <Activity className="status-summary-card__header-icon" aria-hidden="true" />
        <h3>状态摘要</h3>
      </div>
      <dl className="status-summary-card__grid">
        {items.map((item) => (
          <div
            key={item.label}
            className={`status-summary-card__metric${
              item.label === "当前阶段" ? " status-summary-card__metric--stage" : ""
            }`}
          >
            <dt>{item.label}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <dt className="text-xs font-semibold uppercase text-[var(--muted)]">{label}</dt>
      <dd className="metric-value mt-1 text-base font-bold text-[var(--foreground)]">{value}</dd>
    </div>
  );
}

function formatBoolean(value: boolean): string {
  return value ? "开启" : "关闭";
}

function statusLabel(status: SearchRunStatusResponse["status"]): string {
  const labels: Record<SearchRunStatusResponse["status"], string> = {
    queued: "排队中",
    running: "运行中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
  };
  return labels[status] ?? status;
}

function eventNameLabel(eventName: string): string {
  const labels: Record<string, string> = {
    run_started: "任务开始",
    stage_started: "阶段开始",
    stage_completed: "阶段完成",
    connector_completed: "检索源完成",
    connector_started: "检索源开始",
    budget_stop: "预算停止",
    run_cancelled: "任务取消",
    warning: "提示",
    cost_updated: "成本更新",
    error: "错误",
    run_completed: "任务结束",
    sse_error: "事件连接异常",
  };
  return labels[eventName] ?? eventName;
}

function categoryLabel(category: RankedPaper["category"]): string {
  const labels: Record<RankedPaper["category"], string> = {
    highly_relevant: "高度相关",
    partially_relevant: "部分相关",
    weakly_relevant: "弱相关",
    irrelevant: "不相关",
    insufficient_evidence: "证据不足",
  };
  return labels[category] ?? category;
}

function costValue(
  costReport: CostReport | null | undefined,
  key: keyof CostReport,
): number {
  const value = costReport?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function EmptyBlock({ lines }: { lines: number }) {
  return (
    <div className="space-y-3" aria-hidden="true">
      {Array.from({ length: lines }).map((_, index) => (
        <SkeletonLine key={index} className={index === lines - 1 ? "w-2/3" : "w-full"} />
      ))}
    </div>
  );
}

function EmptyResults() {
  return (
    <div className="rounded-lg border border-dashed border-[var(--border)] bg-[var(--surface-raised)] p-8 text-center">
      <FileText className="mx-auto mb-3 h-8 w-8 text-[var(--primary)]" aria-hidden="true" />
      <h3 className="text-lg font-bold">暂无检索结果</h3>
      <p className="mx-auto mt-2 max-w-xl text-sm text-[var(--muted)]">
        创建真实检索任务后，这里会展示高度相关论文、部分相关论文、方法聚类、时间线和证据缺口。
      </p>
    </div>
  );
}

function LoadingResults() {
  return (
    <div className="grid gap-4 md:grid-cols-2" aria-label="结果加载中">
      {Array.from({ length: 4 }).map((_, index) => (
        <div key={index} className="rounded-lg border border-[var(--border)] bg-[var(--surface-raised)] p-5">
          <SkeletonLine className="mb-4 w-1/3" />
          <SkeletonLine className="mb-3 w-full" />
          <SkeletonLine className="mb-3 w-5/6" />
          <SkeletonLine className="w-2/3" />
        </div>
      ))}
    </div>
  );
}
